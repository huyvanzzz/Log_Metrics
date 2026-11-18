"""Offline matching, geometry, and denominator checks for the new probe."""
import importlib
import unittest

from tokenizers import Tokenizer, decoders, models, pre_tokenizers


def target(text="White sedan", tag="one", camera="Front", bbox=None):
    return {"object_tag": tag, "camera": camera,
            "bbox": [0, 0, 200, 112.5] if bbox is None else bbox,
            "image_size": [1600, 900],
            "reference_span": {"start": 0, "end": len(text), "text": text}}


class SubwordTokenizer:
    """Real local WordPiece decoder, with two spellings of the same text."""
    all_special_ids = [0]

    def __init__(self):
        vocab = {piece: i for i, piece in enumerate(
            ["[PAD]", "[UNK]", "White", "Wh", "##ite", "sedan", "se", "##dan",
             ".", "white", "sedans", "and", "Red", "light", "truck"])}
        self.backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
        self.backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.backend.decoder = decoders.WordPiece(cleanup=False)
        self.backend.add_special_tokens(["[PAD]"])

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return self.backend.decode(ids, skip_special_tokens=True)

    def encode(self, text):
        return self.backend.encode(text).ids


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.loc = importlib.import_module("smolvlm_probing.localization")
        except ModuleNotFoundError:
            self.loc = None
        self.assertIsNotNone(self.loc, "localization implementation is required")
        self.tokenizer = SubwordTokenizer()

    def match(self, text, targets=None, **kwargs):
        return self.loc.match_targets(self.tokenizer, self.tokenizer.encode(text),
                                      targets or [target()], **kwargs)

    def test_real_subwords_align_original_noncanonical_ids(self):
        ids = [3, 4, 6, 7]
        self.assertNotEqual(ids, self.tokenizer.encode("White sedan"))
        decision = self.loc.match_targets(self.tokenizer, ids, [target()])[0]
        self.assertEqual(decision["status"], "eligible")
        self.assertEqual(decision["token_indices"], [0, 1, 2, 3])
        self.assertEqual(decision["matched_span"], {"start": 0, "end": 11, "text": "White sedan"})

    def test_special_token_in_middle_keeps_original_offsets(self):
        decision = self.loc.match_targets(self.tokenizer, [3, 4, 0, 6, 7], [target()])[0]
        self.assertEqual(decision["token_indices"], [0, 1, 3, 4])

    def test_case_and_word_boundaries_are_exact(self):
        for text in ["white sedan", "White sedans", "Red light"]:
            self.assertEqual(self.match(text)[0]["status"], "missing_phrase")
        self.assertEqual(self.match("White sedan.")[0]["status"], "eligible")

    def test_internal_whitespace_and_punctuation_are_literal(self):
        self.assertEqual(self.match("White sedan", [target("White  sedan")])[0]["status"], "missing_phrase")
        self.assertEqual(self.match("White sedan.", [target("White, sedan")])[0]["status"], "missing_phrase")

    def test_repeated_phrase_is_excluded(self):
        decision = self.match("White sedan and White sedan")[0]
        self.assertEqual(decision["status"], "repeated_phrase")
        self.assertEqual(decision["token_indices"], [])

    def test_overlapping_repeated_occurrences_are_also_excluded(self):
        decision = self.match("White White White", [target("White White")])[0]
        self.assertEqual(decision["status"], "repeated_phrase")

    def test_overlapping_bindings_exclude_both_targets(self):
        decisions = self.match("White sedan", [target(), target("sedan", "two")])
        self.assertEqual([d["status"] for d in decisions], ["ambiguous_binding"] * 2)
        self.assertTrue(all(not d["token_indices"] for d in decisions))

    def test_same_mention_cannot_bind_different_objects(self):
        self.assertEqual([d["status"] for d in self.match("White sedan", [target(), target(tag="two")])],
                         ["ambiguous_binding"] * 2)

    def test_duplicate_target_rows_do_not_share_one_mention(self):
        self.assertEqual([d["status"] for d in self.match("White sedan", [target(), target()])],
                         ["ambiguous_binding"] * 2)

    def test_unstable_prefix_decoder_is_audited(self):
        class Unstable:
            all_special_ids = []
            def decode(self, ids, **kwargs):
                return {(): "", (1,): "WhitX", (1, 2): "White sedan"}[tuple(ids)]
        decision = self.loc.match_targets(Unstable(), [1, 2], [target()])[0]
        self.assertEqual(decision["status"], "unalignable_tokens")
        self.assertEqual(decision["token_indices"], [])

    def test_empty_invalid_and_length_limited_generations(self):
        self.assertEqual(self.loc.match_targets(self.tokenizer, [0], [target()])[0]["status"], "empty_generation")
        for status in ["invalid_multimodal_generation", "invalid"]:
            self.assertEqual(self.match("White sedan", generation_status=status)[0]["status"], "invalid_generation")
        self.assertEqual(self.match("White sedan", generation_status="length")[0]["status"], "eligible")

    def test_missing_phrase_never_selects_entire_response(self):
        decisions = self.match("Red light", [target(), target("truck", "two")])
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(d["status"] == "missing_phrase" and d["token_indices"] == [] for d in decisions))

    def test_geometry_camera_boundary_and_row_major_corners(self):
        expected = {0: ("Front", [100, 56.25]), 7: ("Front", [1500, 56.25]),
                    56: ("Front", [100, 843.75]), 63: ("Front", [1500, 843.75]),
                    64: ("Front-Left", [100, 56.25]), 383: ("Back-Right", [1500, 843.75])}
        for peak, (camera, center) in expected.items():
            d = self.match("White sedan", [target(camera=camera, bbox=[0, 0, 1600, 900])])
            m = [[0.0] * 384 for _ in range(6)]
            for row in m:
                row[peak] = 1.0
            result = self.loc.evaluate_targets(d, [m])[0]["group_results"][0]
            self.assertEqual(result["peak_camera"], camera)
            self.assertEqual(result["peak_center"], center)
            self.assertTrue(result["correct"])

    def test_global_peak_other_camera_is_incorrect(self):
        maps = [[0.0] * 384 for _ in range(6)]
        for row in maps:
            row[0], row[64] = 0.5, 1.0
        result = self.loc.evaluate_targets(self.match("White sedan"), [maps])[0]
        self.assertTrue(all(not g["correct"] for g in result["group_results"]))

    def test_bbox_boundaries_are_inclusive_and_unreachable_is_retained(self):
        self.assertTrue(self.loc.bbox_contains_grid_center(target(bbox=[100, 56.25, 101, 57])))
        tiny = target(bbox=[101, 57, 102, 58])
        self.assertFalse(self.loc.bbox_contains_grid_center(tiny))
        result = self.loc.evaluate_targets(self.match("White sedan", [tiny]), [[[0.0] * 384 for _ in range(6)]])[0]
        self.assertEqual(result["status"], "eligible")
        self.assertFalse(result["bbox_contains_grid_center"])
        self.assertTrue(all(not g["correct"] for g in result["group_results"]))

    def test_ties_choose_smallest_index_and_report_zero_map(self):
        result = self.loc.evaluate_targets(self.match("White sedan"), [[[0.0] * 384 for _ in range(6)]])[0]
        for group in result["group_results"]:
            self.assertEqual(group["peak_index"], 0)
            self.assertEqual(group["tied_peak_count"], 384)
            self.assertTrue(group["zero_map"])

    def test_aggregation_uses_same_eligible_denominator_and_empty_partition(self):
        decisions = self.match("White sedan", [target(), target("truck", "two")])
        maps = [[0.0] * 384 for _ in range(6)]
        maps[0][0] = 1.0
        for row in maps[1:]:
            row[64] = 1.0
        audits = self.loc.evaluate_targets(decisions, [maps])
        summary = self.loc.aggregate_localization([{"sample_id": "test", "partition": "with_coordinates", "targets": audits}])
        group = summary["with_coordinates"]
        self.assertEqual((group["total_targets"], group["eligible_targets"], group["coverage"]), (2, 1, .5))
        self.assertEqual(group["reasons"]["missing_phrase"], 1)
        self.assertEqual([g["total"] for g in group["groups"]], [1] * 6)
        self.assertEqual([g["accuracy"] for g in group["groups"]], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        empty = summary["without_coordinates"]
        self.assertIsNone(empty["coverage"])
        self.assertTrue(all(g["accuracy"] is None for g in empty["groups"]))

    def test_aggregation_reports_unreachable_total_and_eligible(self):
        decisions = self.match("White sedan", [target(bbox=[1, 1, 2, 2]), target("truck", "two", bbox=[1, 1, 2, 2])])
        audits = self.loc.evaluate_targets(decisions, [[[0.0] * 384 for _ in range(6)]])
        result = self.loc.aggregate_localization([{"sample_id": "a", "partition": "without_coordinates", "targets": audits}])["without_coordinates"]
        self.assertEqual((result["unreachable_total"], result["unreachable_eligible"]), (2, 1))

    def test_nonfinite_maps_and_missing_eligible_map_fail(self):
        decisions = self.match("White sedan")
        with self.assertRaises(ValueError):
            self.loc.evaluate_targets(decisions, [])
        maps = [[0.0] * 384 for _ in range(6)]
        maps[0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.loc.evaluate_targets(decisions, [maps])


if __name__ == "__main__":
    unittest.main()
