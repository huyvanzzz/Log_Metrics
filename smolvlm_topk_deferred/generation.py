"""Fixed global selections gathered after a native decoder block, with layer-local KV."""
import torch
from transformers.cache_utils import DynamicCache
from smolvlm_topk_benchmark.generation import compact_prompt, decode_sequences, greedy_config


def _mask(valid, count, past=0, dtype=torch.float32):
    causal = torch.arange(valid.shape[1], device=valid.device)[None, :] <= past + torch.arange(count, device=valid.device)[:, None]
    allowed = causal[None, None] & valid[:, None, None, :]
    allowed |= ~allowed.any(-1, keepdim=True)
    return torch.zeros(allowed.shape, device=valid.device, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)


class DeferredGreedyDecoder:
    def __init__(self, model, prune_after_layer_index=2):
        self.model = model
        self.text = model.native.model.text_model
        if type(prune_after_layer_index) is not int or not 0 <= prune_after_layer_index < len(self.text.layers):
            raise ValueError('Invalid prune-after layer index')
        self.index = prune_after_layer_index
        self.last_debug = {}

    def _validate(self, fused, selections, config):
        if self.model.training or any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('generation requires an eval FP32 model')
        if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
            raise ValueError('generation requires autocast disabled')
        if type(config.max_new_tokens) is not int or config.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        expected = greedy_config(self.model, config.max_new_tokens).to_dict()
        actual = config.to_dict()
        # Generation bookkeeping does not affect logits. Every other changed option
        # must be implemented explicitly rather than silently ignored.
        ignored = {'eos_token_id', 'transformers_version', '_from_model_config', 'return_dict_in_generate', 'output_scores', 'output_logits', 'output_attentions', 'output_hidden_states'}
        if any(actual.get(k) != v for k, v in expected.items() if k not in ignored):
            raise ValueError('Only neutral greedy generation settings are supported')
        if config.pad_token_id is None:
            raise ValueError('Native PAD token is required')
        if bool((fused['original_lengths'] + config.max_new_tokens > self.text.config.max_position_embeddings).any()):
            raise ValueError('Original prompt plus generation exceeds native context budget')
        compact_prompt(fused, selections)  # Validate sorted, unique source selections.

    def _prefill(self, fused, selections, cache):
        hidden, valid, pos = fused['inputs_embeds'], fused['attention_mask'].bool(), fused['position_ids']
        prefixes, lengths, counts = [], [], []
        selected_counts = [self.model.visual_token_len if s is None else len(s) for s in selections]
        for i, layer in enumerate(self.text.layers):
            prefixes.append(valid.clone())
            lengths.append(hidden.shape[1])
            counts.append([self.model.visual_token_len] * hidden.shape[0] if i <= self.index else selected_counts)
            rotary = self.text.rotary_emb(hidden, pos)
            hidden = layer(hidden, attention_mask=_mask(valid, hidden.shape[1], dtype=hidden.dtype),
                position_ids=pos, position_embeddings=rotary, past_key_value=cache,
                use_cache=cache is not None, cache_position=torch.arange(hidden.shape[1], device=hidden.device))[0]
            if i == self.index:
                current = dict(fused, inputs_embeds=hidden)
                compact = compact_prompt(current, selections)
                hidden, valid, pos = compact['inputs_embeds'], compact['attention_mask'].bool(), compact['prompt_positions']
        logits = self.model.native.lm_head(self.text.norm(hidden[:, -1:, :]))[:, 0].float()
        return logits, prefixes, dict(visual_counts_by_layer=counts, prefill_cache_lengths=lengths,
            selected_source_indices=[list(range(self.model.visual_token_len)) if s is None else list(s) for s in selections])

    def _step(self, token, active, positions, cache, prefixes):
        hidden = self.model.native.get_input_embeddings()(token[:, None])
        for i, layer in enumerate(self.text.layers):
            past = cache.get_seq_length(i)
            prefixes[i] = torch.cat((prefixes[i], active[:, None]), 1)
            hidden = layer(hidden, attention_mask=_mask(prefixes[i], 1, past, hidden.dtype),
                position_ids=positions[:, None], position_embeddings=self.text.rotary_emb(hidden, positions[:, None]),
                past_key_value=cache, use_cache=True,
                cache_position=torch.arange(past, past + 1, device=hidden.device))[0]
        return self.model.native.lm_head(self.text.norm(hidden))[:, 0].float()

    @torch.inference_mode()
    def generate(self, fused, selections, generation_config, debug=False):
        return self._generate(fused, selections, generation_config, debug, False)

    @torch.inference_mode()
    def generate_reference(self, fused, selections, generation_config, debug=False):
        return self._generate(fused, selections, generation_config, debug, True)

    def _generate(self, fused, selections, config, debug, reference):
        self.last_debug = {}
        self._validate(fused, selections, config)
        cache = None if reference else DynamicCache()
        logits, prefixes, info = self._prefill(fused, selections, cache)
        source = dict(fused)
        finished = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.bool)
        eos = config.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos, (list, tuple)) else [eos]
        eos = torch.tensor(eos, device=logits.device, dtype=torch.long)
        output = []
        for step in range(config.max_new_tokens):
            if step:
                active = ~finished
                positions = fused['original_lengths'] + step - 1
                if reference:
                    source = dict(source,
                        input_ids=torch.cat((source['input_ids'], output[-1][:, None]), 1),
                        inputs_embeds=torch.cat((source['inputs_embeds'], self.model.native.get_input_embeddings()(output[-1][:, None])), 1),
                        attention_mask=torch.cat((source['attention_mask'], active[:, None]), 1),
                        position_ids=torch.cat((source['position_ids'], positions[:, None]), 1))
                    logits, _, _ = self._prefill(source, selections, None)
                else:
                    logits = self._step(output[-1], active, positions, cache, prefixes)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Generation logits contain NaN/Inf')
            token = logits.argmax(-1).masked_fill(finished, config.pad_token_id)
            output.append(token)
            finished |= (token[:, None] == eos[None, :]).any(1)
            if bool(finished.all()):
                break
        if debug:
            info['prune_after_layer_index'] = self.index
            info['final_cache_lengths'] = [] if reference else [cache.get_seq_length(i) for i in range(len(self.text.layers))]
            self.last_debug = info
        return decode_sequences(self.model.processor.tokenizer, torch.stack(output, 1), config.eos_token_id, config.max_new_tokens)
