"""Frozen predictor selection, one mass-preserving layer, then hard gather."""
import torch
from transformers.cache_utils import DynamicCache
from smolvlm_topk_deferred.generation import _mask
from smolvlm_topk_benchmark.generation import compact_prompt,decode_sequences,greedy_config
from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
from smolvlm_map_predictor_after2.prefix import fuse_inputs,predictor_inputs
from smolvlm_map_predictor.model import gather_questions,topk_indices
from .attention import intervention


class Decoder(PredictorGreedyDecoder):
    def __init__(self,backbone,predictor,variant,keep_tokens=32):
        super().__init__(backbone,predictor,keep_tokens)
        if variant not in ('before','after2'):
            raise ValueError('variant must be before or after2')
        self.variant=variant
        self.index=0 if variant=='before' else 3
        if len(self.text.layers)<=self.index:
            raise ValueError('intervention layer missing')

    def _scores(self,fused,hidden,batch):
        image=fused['image_positions']; pos=fused['position_ids']
        if self.variant=='after2':
            valid=fused['attention_mask'].bool()
            columns=torch.where(valid,torch.arange(valid.shape[1],device=valid.device)[None],-1).max(-1).values
            return self.predictor(**predictor_inputs(dict(hidden=hidden,image_positions=image,pos=pos,last_query_columns=columns)))
        visual=hidden.gather(1,image[:,:,None].expand(-1,-1,hidden.shape[-1]))
        question,mask,positions=gather_questions(hidden,pos,batch)
        return self.predictor(visual=visual,question=question,question_mask=mask,
            visual_positions=pos.gather(1,image),question_positions=positions)

    def _affected(self,batch,fused):
        device=fused['input_ids'].device
        if 'question_columns' in batch:
            columns=batch['question_columns'].to(device)
            qvalid=batch.get('question_valid',batch.get('question_mask')).to(device).bool()
            start=columns.masked_fill(~qvalid,fused['input_ids'].shape[1]).min(-1).values
        else:
            question=batch['question_mask'].to(device).bool()
            start=torch.where(question,torch.arange(question.shape[1],device=device)[None],question.shape[1]).min(-1).values
        if bool((start>=fused['input_ids'].shape[1]).any()):
            raise ValueError('question span empty')
        return fused['attention_mask'].bool() & (torch.arange(fused['input_ids'].shape[1],device=device)[None]>=start[:,None])

    def _mass_prefill(self,fused,batch,cache,affected,selections=None,debug=False):
        hidden=fused['inputs_embeds']; valid=fused['attention_mask'].bool(); pos=fused['position_ids']
        masks=[]; lengths=[]; counts=[]; mass={}
        if selections is None and self.variant=='before':
            selections=topk_indices(self._scores(fused,hidden,batch),self.keep_tokens).cpu().tolist()
        for i,layer in enumerate(self.text.layers):
            masks.append(valid); lengths.append(hidden.shape[1])
            counts.append([self.model.visual_token_len if i<=self.index else self.keep_tokens]*hidden.shape[0])
            args=dict(attention_mask=_mask(valid,hidden.shape[1],dtype=hidden.dtype),position_ids=pos,
                position_embeddings=self.text.rotary_emb(hidden,pos),past_key_value=cache,use_cache=cache is not None,
                cache_position=torch.arange(hidden.shape[1],device=hidden.device))
            if i==self.index:
                selected=torch.tensor(selections,device=hidden.device,dtype=torch.long)
                with intervention(layer.self_attn,fused['image_positions'],selected,affected,mass if debug else None):
                    hidden=layer(hidden,**args)[0]
                compact=compact_prompt(dict(fused,inputs_embeds=hidden),selections)
                hidden,valid,pos=compact['inputs_embeds'],compact['attention_mask'].bool(),compact['prompt_positions']
            else:
                hidden=layer(hidden,**args)[0]
            if i==2 and self.variant=='after2' and selections is None:
                selections=topk_indices(self._scores(fused,hidden,batch),self.keep_tokens).cpu().tolist()
        logits=self.model.native.lm_head(self.text.norm(hidden[:,-1:]))[:,0].float()
        return logits,masks,selections,dict(visual_counts_by_layer=counts,prefill_cache_lengths=lengths,
            selected_source_indices=selections,intervention_layer_index=self.index,prune_after_layer_index=self.index,
            predictor_calls=1,mass=mass)

    @torch.inference_mode()
    def generate(self,batch,max_new_tokens=512,debug=False):
        return self._decode_mass(batch,max_new_tokens,debug,False)

    @torch.inference_mode()
    def generate_reference(self,batch,max_new_tokens=512,debug=False):
        return self._decode_mass(batch,max_new_tokens,debug,True)

    def _decode_mass(self,batch,max_new_tokens,debug,reference):
        self.last_debug={}
        if self.model.training or self.predictor.training:
            raise ValueError('generation requires eval backbone and predictor')
        if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
            raise ValueError('generation requires autocast disabled')
        if any(p.dtype != torch.float32 for p in self.predictor.parameters()):
            raise ValueError('generation requires FP32 predictor')
        fused=fuse_inputs(self.model,batch)
        config=greedy_config(self.model,max_new_tokens)
        self._validate(fused,[None]*fused['input_ids'].shape[0],config)
        affected=self._affected(batch,fused)
        cache=None if reference else DynamicCache()
        logits,masks,selections,info=self._mass_prefill(fused,batch,cache,affected,debug=debug)
        source=dict(fused)
        finished=torch.zeros(logits.shape[0],device=logits.device,dtype=torch.bool)
        eos=config.eos_token_id
        eos=[] if eos is None else eos if isinstance(eos,(list,tuple)) else [eos]
        eos=torch.tensor(eos,device=logits.device,dtype=torch.long)
        output=[]
        for step in range(max_new_tokens):
            if step:
                active=~finished; positions=fused['original_lengths']+step-1
                if reference:
                    source=dict(source,input_ids=torch.cat((source['input_ids'],output[-1][:,None]),1),
                        inputs_embeds=torch.cat((source['inputs_embeds'],self.model.native.get_input_embeddings()(output[-1][:,None])),1),
                        attention_mask=torch.cat((source['attention_mask'],active[:,None]),1),
                        position_ids=torch.cat((source['position_ids'],positions[:,None]),1))
                    affected=torch.cat((affected,torch.zeros_like(active[:,None])),1)
                    logits,_,_,_=self._mass_prefill(source,batch,None,affected,selections)
                else:
                    logits=self._step(output[-1],active,positions,cache,masks)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Generation logits contain NaN/Inf')
            token=logits.argmax(-1).masked_fill(finished,config.pad_token_id)
            output.append(token)
            finished |= (token[:,None]==eos[None]).any(1)
            if bool(finished.all()): break
        records=decode_sequences(self.model.processor.tokenizer,torch.stack(output,1),config.eos_token_id,max_new_tokens)
        for record,selection in zip(records,selections): record['visual_indices']=selection
        if debug:
            info['final_cache_lengths']=[] if reference else [cache.get_seq_length(i) for i in range(len(self.text.layers))]
            self.last_debug=info
        return records
