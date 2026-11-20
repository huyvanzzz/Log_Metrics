"""Native fusion followed by physical prompt compaction and shared text decoding."""
import torch
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel


@torch.inference_mode()
def fuse_batch(model, batch):
    if model.training or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('generation requires an eval FP32 model')
    device = next(model.parameters()).device
    ids = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    if not mask.bool().any(1).all():
        raise ValueError('empty prompt')
    projected = model._project(batch['visual_features'], ids.shape[0])
    embeddings = model.native.model.inputs_merger(input_ids=ids,
        inputs_embeds=model.native.get_input_embeddings()(ids), image_hidden_states=projected)
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
    image = [torch.where((row == model.native.config.image_token_id) & m.bool())[0] for row, m in zip(ids, mask)]
    if any(len(x) != model.visual_token_len for x in image):
        raise ValueError('unexpected visual token count')
    return dict(input_ids=ids, inputs_embeds=embeddings, attention_mask=mask,
                position_ids=positions, original_lengths=mask.sum(-1).long(),
                image_positions=torch.stack(image),pad_token_id=model.processor.tokenizer.pad_token_id)


def compact_prompt(fused, selections):
    ids = fused['input_ids']
    if len(selections) != ids.shape[0]:
        raise ValueError('one selection per batch row required')
    indices = []
    for row, selection in enumerate(selections):
        visual = fused['image_positions'][row]
        keep = fused['attention_mask'][row].bool().clone()
        if selection is not None:
            selected = list(selection)
            if selected != sorted(set(selected)) or any(not isinstance(x, int) or x < 0 or x >= len(visual) for x in selected):
                raise ValueError('visual indices must be sorted, unique, and in range')
            keep[visual] = False
            keep[visual[selected]] = True
        indices.append(torch.where(keep)[0])
    width = max(map(len, indices))
    out = dict(input_ids=ids.new_full((len(indices), width),fused.get('pad_token_id',0)),
               inputs_embeds=fused['inputs_embeds'].new_zeros((len(indices), width, fused['inputs_embeds'].shape[-1])),
               attention_mask=fused['attention_mask'].new_zeros((len(indices), width)),
               prompt_positions=fused['position_ids'].new_zeros((len(indices), width)),
               original_lengths=fused['original_lengths'])
    for row, index in enumerate(indices):
        start = width - len(index)
        for dest, source in [('input_ids', 'input_ids'), ('inputs_embeds', 'inputs_embeds'), ('prompt_positions', 'position_ids')]:
            out[dest][row, start:] = fused[source][row, index]
        out['attention_mask'][row, start:] = 1
    return out


class SharedCausalLM(LlamaForCausalLM):
    """GenerationMixin adapter referencing the already loaded native weights."""
    def __init__(self, native):
        LlamaPreTrainedModel.__init__(self, native.config.text_config)
        self.model = native.model.text_model
        self.lm_head = native.lm_head
        self.vocab_size = native.config.text_config.vocab_size
        self.eval()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
            attention_mask=None, inputs_embeds=None, cache_position=None,
            original_lengths=None, prompt_positions=None, **kwargs):
        prepared = super().prepare_inputs_for_generation(input_ids,
            past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
        if cache_position[0] == 0:
            prepared['position_ids'] = prompt_positions
        else:
            prepared['position_ids'] = original_lengths[:, None] + cache_position[None, :] - prompt_positions.shape[1]
        return prepared


def greedy_config(model, max_new_tokens=512):
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    native = getattr(model, 'native', model)
    source = native.generation_config
    return GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        num_return_sequences=1, use_cache=True, repetition_penalty=1.0,
        encoder_repetition_penalty=1.0, no_repeat_ngram_size=0,
        bos_token_id=source.bos_token_id, eos_token_id=source.eos_token_id,
        pad_token_id=source.pad_token_id)


def decode_sequences(tokenizer, continuation, eos_ids, limit):
    eos = set(eos_ids or []) if not isinstance(eos_ids, int) else {eos_ids}
    records = []
    for row in continuation.detach().cpu().tolist():
        tokens = row[:limit]
        stop = 'max_new_tokens'
        for index, token in enumerate(tokens):
            if token in eos:
                tokens = tokens[:index + 1]
                stop = 'eos'
                break
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        records.append(dict(generated_ids=tokens, generated_token_count=len(tokens),
                            stop_reason=stop, raw_prediction=raw, prediction=raw.strip()))
    return records


@torch.inference_mode()
def generate_compacted(adapter, compact, tokenizer, generation_config):
    if adapter.training or any(p.dtype != torch.float32 for p in adapter.parameters()):
        raise ValueError('generation requires an eval FP32 model')
    if (compact['original_lengths'] + generation_config.max_new_tokens > adapter.config.max_position_embeddings).any():
        raise ValueError('original prompt plus generation exceeds native context budget')
    sequences = adapter.generate(**compact, generation_config=generation_config)
    if hasattr(sequences, 'sequences'):
        sequences = sequences.sequences
    continuation = sequences[:, compact['input_ids'].shape[1]:]
    return decode_sequences(tokenizer, continuation, generation_config.eos_token_id, generation_config.max_new_tokens)
