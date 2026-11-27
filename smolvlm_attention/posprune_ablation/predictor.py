"""Load the frozen after-block-index-2 MAP scorer for a paired PosPrune run."""

import torch


def load_after2_predictor(path, backbone, checkpoint_sha256, prepared_hashes, device):
    from smolvlm_map_predictor_after2.model import METHOD_ID, MapPredictor

    payload = torch.load(path, map_location='cpu', weights_only=True)
    identity = payload.get('identity', {})
    method = dict(id=METHOD_ID, input_layer_index=2, prune_after_layer_index=2, teacher_layer_index=22,
                  query_policy='last_valid_native_generation_prompt_token')
    expected = dict(checkpoint_sha256=checkpoint_sha256, model=backbone.architecture_signature, method=method,
                    predictor_input_boundary='decoder_block_output', predictor_input_layer_index=2,
                    production_prune_after_layer_index=2, predictor_query_policy=method['query_policy'])
    if (payload.get('format') != 'map_predictor_checkpoint_v1' or
            any(identity.get(key) != value for key, value in expected.items()) or
            any(identity.get('prepared_hashes', {}).get(key) != value for key, value in prepared_hashes.items())):
        raise ValueError('Use the trained after2 predictor for this backbone and prepared dataset')
    signature = payload.get('architecture_signature', {})
    scorer = MapPredictor(backbone.native.model.text_model.config, hidden_dim=signature['hidden_dim'],
                          projection_dim=signature['projection_dim'], visual_tokens=backbone.visual_token_len)
    if scorer.architecture_signature != signature:
        raise ValueError('The predictor architecture signature does not match the loaded model')
    scorer.load_state_dict(payload['model_state_dict'], strict=True)
    return scorer.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
