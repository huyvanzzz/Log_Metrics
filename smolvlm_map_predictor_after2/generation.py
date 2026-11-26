"""Greedy decoding with one predictor selection at the post-block-two boundary."""
import torch
from transformers.cache_utils import DynamicCache
from smolvlm_topk_deferred.generation import DeferredGreedyDecoder, _mask
from smolvlm_topk_benchmark.generation import compact_prompt, decode_sequences, greedy_config
from .model import topk_indices
from .prefix import fuse_inputs, run_prefix, predictor_inputs


class PredictorGreedyDecoder(DeferredGreedyDecoder):
    def __init__(self, backbone, predictor, keep_tokens=32, selection_fn=None):
        super().__init__(backbone, prune_after_layer_index=2)
        if type(keep_tokens) is not int or not 0 < keep_tokens <= backbone.visual_token_len:
            raise ValueError('invalid keep_tokens')
        self.predictor,self.keep_tokens=predictor,keep_tokens
        self.selection_fn=selection_fn

    def _suffix(self,fused,prefix,selections,cache):
        compact=compact_prompt(dict(fused,inputs_embeds=prefix['hidden']),selections)
        hidden,valid,pos=compact['inputs_embeds'],compact['attention_mask'].bool(),compact['prompt_positions']
        masks=list(prefix['prefix_masks']); lengths=list(prefix['lengths'])
        causal_mask = _mask(valid, hidden.shape[1], dtype=hidden.dtype)
        rotary = self.text.rotary_emb(hidden, pos)
        slots = torch.arange(hidden.shape[1], device=hidden.device)
        for layer in self.text.layers[3:]:
            masks.append(valid); lengths.append(hidden.shape[1])
            hidden=layer(hidden,attention_mask=causal_mask,
                position_ids=pos,position_embeddings=rotary,past_key_value=cache,
                use_cache=cache is not None,cache_position=slots)[0]
        return self.model.native.lm_head(self.text.norm(hidden[:,-1:]))[:,0].float(),masks,lengths

    def _step(self, token, active, positions, cache, prefixes):
        hidden = self.model.native.get_input_embeddings()(token[:, None])
        rotary = self.text.rotary_emb(hidden, positions[:, None])
        groups = {}
        for i, layer in enumerate(self.text.layers):
            past = cache.get_seq_length(i)
            previous = prefixes[i]
            if previous.shape[1] != past:
                raise ValueError('layer cache length differs from its padding mask')
            # Only reuse masks with the SAME tensor identity and physical slots,
            # not merely equal lengths: the prefix and suffix have different keys.
            key = (id(previous), past)
            if key not in groups:
                valid = torch.cat((previous, active[:, None]), 1)
                groups[key] = (valid, _mask(valid, 1, past, hidden.dtype),
                               torch.arange(past, past + 1, device=hidden.device))
            valid, attention, slots = groups[key]
            prefixes[i] = valid
            hidden = layer(hidden, attention_mask=attention,
                position_ids=positions[:, None], position_embeddings=rotary,
                past_key_value=cache, use_cache=True, cache_position=slots)[0]
        return self.model.native.lm_head(self.text.norm(hidden))[:, 0].float()

    @torch.inference_mode()
    def generate(self,batch,max_new_tokens=512,debug=False):
        return self._decode(batch,max_new_tokens,debug,False)

    @torch.inference_mode()
    def generate_reference(self,batch,max_new_tokens=512,debug=False):
        return self._decode(batch,max_new_tokens,debug,True)

    def _decode(self,batch,max_new_tokens,debug,reference):
        self.last_debug={}
        if self.model.training:
            raise ValueError('generation requires an eval backbone')
        fused=fuse_inputs(self.model,batch)
        config=greedy_config(self.model,max_new_tokens)
        # Validate before performing any decoder computation.
        self._validate(fused,[None]*fused['input_ids'].shape[0],config)
        if self.predictor.training:
            raise ValueError('generation requires an eval predictor')
        cache=None if reference else DynamicCache()
        prefix=run_prefix(self.model,fused,cache)
        inputs=predictor_inputs(prefix)
        scores=self.predictor(**inputs)
        chosen=(topk_indices(scores,self.keep_tokens) if self.selection_fn is None else
                self.selection_fn(scores,inputs['visual'],self.keep_tokens))
        if chosen.shape != (scores.shape[0],self.keep_tokens) or chosen.dtype != torch.long:
            raise ValueError('Selection must return Long indices [B,keep_tokens]')
        selections=chosen.cpu().tolist()
        logits,masks,lengths=self._suffix(fused,prefix,selections,cache)
        source=dict(fused)
        finished=torch.zeros(logits.shape[0],device=logits.device,dtype=torch.bool)
        eos=config.eos_token_id
        eos=[] if eos is None else eos if isinstance(eos,(list,tuple)) else [eos]
        eos=torch.tensor(eos,device=logits.device,dtype=torch.long)
        output=[]
        for step in range(max_new_tokens):
            if step:
                active=~finished
                positions=fused['original_lengths']+step-1
                if reference:
                    source=dict(source,input_ids=torch.cat((source['input_ids'],output[-1][:,None]),1),
                        inputs_embeds=torch.cat((source['inputs_embeds'],self.model.native.get_input_embeddings()(output[-1][:,None])),1),
                        attention_mask=torch.cat((source['attention_mask'],active[:,None]),1),
                        position_ids=torch.cat((source['position_ids'],positions[:,None]),1))
                    # Recompute history using the fixed original selection; never rescore.
                    current=run_prefix(self.model,source)
                    logits,_,_=self._suffix(source,current,selections,None)
                else:
                    logits=self._step(output[-1],active,positions,cache,masks)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Generation logits contain NaN/Inf')
            token=logits.argmax(-1).masked_fill(finished,config.pad_token_id)
            output.append(token)
            finished |= (token[:,None] == eos[None,:]).any(1)
            if bool(finished.all()): break
        records=decode_sequences(self.model.processor.tokenizer,torch.stack(output,1),config.eos_token_id,max_new_tokens)
        for record,selected in zip(records,selections): record['visual_indices']=selected
        if debug:
            count=logits.shape[0]
            self.last_debug=dict(predictor_calls=1,prefix_calls=1,prune_after_layer_index=2,
                visual_counts_by_layer=[[self.model.visual_token_len]*count]*3+[[self.keep_tokens]*count]*(len(self.text.layers)-3),
                prefill_cache_lengths=lengths,final_cache_lengths=[] if reference else [cache.get_seq_length(i) for i in range(len(self.text.layers))],
                selected_source_indices=selections,last_query_columns=prefix['last_query_columns'].detach().cpu(),
                query_source_indices=prefix['pos'].gather(1,prefix['last_query_columns'][:,None]).detach().cpu())
        return records
