import unittest
from unittest.mock import patch
from smolvlm_topk_benchmark import scoring


class ScoringTests(unittest.TestCase):
    def rows(self):
        return [dict(sample_id='a', eval_id=1, reference='Cat\n||| Dog', prediction=''),
                dict(sample_id='b', eval_id=2, reference='a red car', prediction='a red car', generated_tokens=512)]

    def test_all_rows_and_original_text_reach_scorers(self):
        captured = {}
        def compute(gts, res, identity):
            captured.update(gts=gts, res=res)
            return dict.fromkeys(scoring.METRICS, .25)
        with patch.object(scoring, '_identity', return_value={}), patch.object(scoring, '_compute', side_effect=compute):
            result = scoring.score_rows(self.rows(), expected_count=2)
        self.assertEqual(captured['gts'][1], ['Cat\n||| Dog'])
        self.assertEqual(captured['res'][1], [''])
        self.assertEqual(result['diagnostics']['prediction_count'], 2)
        self.assertEqual(result['metrics_display']['CIDEr'], 25)

    def test_invalid_count_and_duplicates_fail(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            scoring.score_rows(rows)
        rows[1]['sample_id'] = 'a'
        with self.assertRaises(ValueError):
            scoring.score_rows(rows, expected_count=2)

    def test_meteor_protocol_sanitization_only(self):
        self.assertEqual(scoring._meteor_text('A\r\n||| B'), 'A   B')

    def test_real_seven_metric_fixture(self):
        try:
            scoring._identity()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        rows = [dict(sample_id='a', eval_id=1, reference='a red car drives down the road', prediction='a red car drives down the road'),
                dict(sample_id='b', eval_id=2, reference='a blue bus stops near a tree', prediction='a blue bus stops near a tree')]
        result = scoring.score_rows(rows, expected_count=2)
        self.assertEqual(set(result['metrics']), set(scoring.METRICS))
        self.assertAlmostEqual(result['metrics']['ROUGE-L'], 1)
        self.assertGreater(result['metrics']['METEOR'], .9)
        self.assertGreater(result['metrics']['CIDEr'], 9)

    def test_real_meteor_empty_and_protocol_characters(self):
        try:
            identity = scoring._identity()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        value = scoring._meteor({1: ['Cat\r\n||| Dog'], 2: ['a blue car']},
                                {1: [''], 2: ['a\n||| blue car']}, identity)
        self.assertGreaterEqual(value, 0)

    def test_scorer_failure_propagates_without_fallback(self):
        with patch.object(scoring, '_identity', return_value={}), patch.object(scoring, '_compute', side_effect=RuntimeError('broken scorer')):
            with self.assertRaisesRegex(RuntimeError, 'broken scorer'):
                scoring.score_rows(self.rows(), expected_count=2)

if __name__ == '__main__':
    unittest.main()
