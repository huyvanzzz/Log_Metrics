"""Single native prompt query over post-block-two visual states."""
import torch
from smolvlm_map_predictor.model import MapPredictor as OriginalMapPredictor, kl_per_sample, topk_indices

METHOD_ID = 'map_after2_pos_tagging_direct_v1'


class MapPredictor(OriginalMapPredictor):
    @property
    def architecture_signature(self):
        return dict(super().architecture_signature, architecture=METHOD_ID, method=METHOD_ID,
                    boundary='raw_output_after_decoder_block_2', query='pos_tagged_tokens',
                    pruning='full_visual_blocks_0_1_2_then_topk_visual_keep_all_text',
                    normalization='direct_visual_log_softmax')

    def forward(self, visual, question, question_mask, visual_positions, question_positions):
        return super().forward(visual, question, question_mask, visual_positions, question_positions)
