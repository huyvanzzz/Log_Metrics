"""Single native prompt query over post-block-two visual states."""
import torch
from smolvlm_map_predictor.model import MapPredictor as OriginalMapPredictor, kl_per_sample, topk_indices

METHOD_ID = 'map_after2_lastprompt_direct_v1'


class MapPredictor(OriginalMapPredictor):
    @property
    def architecture_signature(self):
        return dict(super().architecture_signature, architecture=METHOD_ID, method=METHOD_ID,
                    boundary='raw_output_after_decoder_block_2', query='last_valid_native_prompt_token',
                    pruning='full_visual_blocks_0_1_2_then_topk_visual_keep_all_text',
                    normalization='direct_visual_log_softmax')

    def forward(self, visual, last_query, visual_positions, query_positions):
        if last_query.ndim != 3 or last_query.shape[1] != 1:
            raise ValueError('exactly one last native prompt query required')
        mask = torch.ones(last_query.shape[:2], dtype=torch.bool, device=last_query.device)
        return super().forward(visual, last_query, mask, visual_positions, query_positions)
