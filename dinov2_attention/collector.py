"""Native T5 decoder-start cross-attention, without encoder attention matrices."""
import torch


@torch.inference_mode()
def pad_cross_forward(model, batch):
    """Return CPU FP32 ``[B,layers,visual_keys]`` and first-token logits."""
    if any(module.training for module in model.modules()):
        raise ValueError('Attention measurement requires eval mode')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError('Attention measurement requires FP32 model weights')
    device = next(model.parameters()).device
    ids = batch['input_ids'].to(device=device, dtype=torch.long)
    mask = batch['attention_mask'].to(device=device, dtype=torch.long)
    features = batch['visual_features'].to(device=device, dtype=torch.float32)
    if ids.ndim != 2 or ids.shape != mask.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Invalid prompt token IDs/attention mask')
    embeds, encoder_mask, *_ = model._native_inputs(ids, mask, features)
    native = model.text_model
    encoded = native.encoder(inputs_embeds=embeds, attention_mask=encoder_mask,
                             output_attentions=False, output_hidden_states=False, return_dict=True)
    start = native.config.decoder_start_token_id
    if start != native.config.pad_token_id or start != 0:
        raise ValueError('Expected T5 decoder-start pad token ID 0')
    decoder_ids = torch.full((ids.shape[0], 1), start, dtype=torch.long, device=device)
    outputs = native(encoder_outputs=encoded, attention_mask=encoder_mask,
                     decoder_input_ids=decoder_ids,
                     decoder_attention_mask=torch.ones_like(decoder_ids),
                     output_attentions=True, output_hidden_states=False,
                     use_cache=False, return_dict=True)
    cross = outputs.cross_attentions
    if cross is None or len(cross) != native.config.num_decoder_layers:
        raise ValueError('Missing native T5 cross-attention tensors')
    visual = model.visual_token_len
    values = torch.stack([attention[:, :, 0, :visual].float().mean(1)
                          for attention in cross], dim=1)
    if not torch.isfinite(values).all() or (values < 0).any() or (values.sum(-1) > 1.0001).any():
        raise ValueError('Invalid native cross-attention probabilities')
    return values.cpu(), outputs.logits[:, 0].float().cpu()


def collect_pad_cross_attention(model, batch):
    return pad_cross_forward(model, batch)[0]
