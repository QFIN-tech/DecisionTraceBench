"""Public CLI tests use constructed fixtures only; no corpus or network."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
from decisiontracebench import scoring as score
from test_evaluation import simple_case


class ExternalTests(unittest.TestCase):
    def cli(self, *args, cwd=None):
        return subprocess.run([sys.executable, '-B', '-m', 'decisiontracebench', 'score', *map(str, args)],
                              cwd=cwd, env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True, text=True, encoding='utf-8')

    def test_raw_versus_reviewed_fixture(self):
        case = score.read_json(ROOT/'assets/examples/case.json')
        before = copy.deepcopy(case)
        self.assertEqual(score.score_case(case, 'raw')['f1'], 0)
        self.assertEqual(score.score_case(case, 'reviewed')['f1'], 1)
        self.assertEqual(case, before)

    def test_optimized_python_cannot_skip_scoring_assertions(self):
        for mode in ('raw', 'reviewed'):
            p = subprocess.run([sys.executable, '-O', '-B', '-m', 'decisiontracebench', 'score',
                                '--mode', mode, '--case', str(ROOT/'assets/examples/case.json')],
                               env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(p.returncode, 2, p.stdout)
            self.assertIn('without -O', p.stderr)
            self.assertEqual(p.stdout, '')

    def test_unreviewed_cannot_be_final(self):
        case = score.read_json(ROOT/'assets/examples/case.json')
        for k in score.DECISIONS:
            case.pop(k, None)
        result = score.score_case(case, 'reviewed')
        self.assertIsNone(result['f1'])
        self.assertIsNone(score.aggregate([result])['dcist_percent'])

    def test_macro_not_micro(self):
        def row(tp, gold, pred):
            f1 = 2*tp/(gold+pred)
            return {'status': 'complete', 'f1': f1, 'mode': 'raw', 'time_enabled': True,
                    'score': {'tp': tp, 'gold': gold, 'pred': pred}}
        out = score.aggregate([row(1, 1, 1), row(0, 9, 9)])
        self.assertEqual(out['dcist_percent'], 50)
        self.assertEqual(out['tp'], 1)

    def test_mixed_time_or_mode_rejected(self):
        a = score.score_case(simple_case(), 'raw')
        b = dict(a, time_enabled=False)
        with self.assertRaisesRegex(ValueError, 'time'): score.aggregate([a, b])
        with self.assertRaisesRegex(ValueError, 'modes'): score.aggregate([a, dict(a, mode='reviewed')])

    def test_no_answer_requires_explicit_policy(self):
        case = simple_case(); case.update(inference_status='no_answer', prediction=None)
        with self.assertRaisesRegex(ValueError, 'No answer'): score.score_case(case, 'raw')
        result = score.score_case(case, 'reviewed', 'zero')
        self.assertEqual(result['f1'], 0)
        self.assertEqual(score.aggregate([result])['n_no_answer'], 1)

    def test_no_answer_cannot_discard_existing_answer(self):
        case = simple_case(); case['inference_status'] = 'no_answer'
        with self.assertRaisesRegex(ValueError, 'discard'): score.score_case(case, 'raw', 'zero')

    def test_pending_never_zero(self):
        case = simple_case(); case['inference_status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'scoreable'): score.score_case(case, 'raw', 'zero')

    def test_separate_files_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            p = self.cli('--mode', 'reviewed', '--gold', ROOT/'assets/examples/gold.json',
                         '--prediction', ROOT/'assets/examples/prediction.json', '--reviews', ROOT/'assets/examples/reviews.json',
                         '--time', 'on', cwd=temp)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)['summary']['dcist_percent'], 100)

    def test_manifest_resolves_relative_to_its_file(self):
        with tempfile.TemporaryDirectory() as temp:
            p = self.cli('--mode', 'reviewed', '--manifest', ROOT/'assets/examples/manifest.json', cwd=temp)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)['summary']['n'], 1)

    def test_duplicate_samples_rejected(self):
        item = {'sample_id': 'same', 'gold': 'unused', 'prediction': 'unused', 'time_enabled': True}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'manifest.json'
            path.write_text(json.dumps({'samples': [item, item]}), encoding='utf-8')
            p = self.cli('--mode', 'raw', '--manifest', path)
        self.assertEqual(p.returncode, 2)
        self.assertIn('Duplicate', p.stderr)

    def test_completed_malformed_answer_zero(self):
        case = simple_case(); case['prediction'] = 'not JSON'
        out = score.score_case(case, 'reviewed')
        self.assertEqual(out['status'], 'completed_format_error')
        self.assertEqual(out['f1'], 0)

    def test_cli_refuses_overwriting_input(self):
        p = self.cli('--case', ROOT/'assets/examples/case.json', '--mode', 'raw', '--output', ROOT/'assets/examples/case.json')
        self.assertEqual(p.returncode, 2)
        self.assertIn('overwrite an input', p.stderr)

    def test_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            p = self.cli('--case', ROOT/'assets/examples/case.json', '--mode', 'raw', '--output', path)
            self.assertEqual(p.returncode, 0, p.stderr)
            before = path.read_bytes()
            p = self.cli('--case', ROOT/'assets/examples/case.json', '--mode', 'reviewed', '--output', path)
            self.assertEqual(p.returncode, 2)
            self.assertEqual(path.read_bytes(), before)

    def test_review_file_cannot_override_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'review.json'; path.write_text('{"gold": {}}', encoding='utf-8')
            spec = dict(gold=str(ROOT/'assets/examples/gold.json'), prediction=str(ROOT/'assets/examples/prediction.json'),
                        reviews=str(path), time_enabled=True)
            with self.assertRaisesRegex(ValueError, 'cannot replace'): score.load_external(spec, ROOT, 'reviewed')


if __name__ == '__main__':
    unittest.main()
