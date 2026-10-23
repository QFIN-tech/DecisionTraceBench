"""Offline DCIST scoring for external gold/prediction files. No model calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import statistics

from . import evaluation as adapter

DECISIONS = ('semantic_decisions', 'node_decisions', 'track_decisions')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_case(case, mode, no_answer_policy='error'):
    """Score one already-loaded case; keep originals and review records intact."""
    require(mode in ('raw', 'reviewed'), 'mode must be raw or reviewed')
    require(no_answer_policy in ('error', 'zero'), 'Invalid no-answer policy')
    require(isinstance(case, dict), 'Case must be a JSON object')
    work = copy.deepcopy(case)
    require(type(work.get('time_enabled')) is bool, 'Explicit time_enabled is required')
    status = work.get('inference_status')
    if status == 'no_answer':
        require(no_answer_policy == 'zero', 'No answer: explicitly select --no-answer-policy zero or stop')
        require(work.get('prediction') is None, 'no_answer cannot discard an existing prediction')
        # Validate gold before applying the cohort's no-answer policy.
        probe = {'inference_status': 'completed', 'time_enabled': work['time_enabled'],
                 'gold': work['gold'], 'prediction': {'tracks': []}}
        adapter.evaluate(probe)
        selected = {'tp': 0, 'gold': len(work['gold']['tracks']), 'pred': 0,
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        return {'status': 'no_answer_counted_zero', 'mode': mode, 'score': selected,
                'f1': 0.0, 'dcist_percent': 0.0, 'time_enabled': work['time_enabled'],
                'no_answer_policy': 'zero', 'extra_node_ratio': None,
                'note': 'Explicit cohort-level policy; not a semantic-error diagnosis.'}
    require(status == 'completed', 'Only completed answers or explicitly declared no_answer are scoreable')
    if mode == 'raw':
        for key in DECISIONS:
            work.pop(key, None)
    result = adapter.evaluate(work)
    if mode == 'raw':
        selected = result['raw_score']
        public_status = ('completed_format_error' if result['status'] == 'completed_format_error'
                         else 'complete')
    else:
        selected = (result['raw_score'] if result['status'] == 'completed_format_error'
                    else result.get('reviewed_score'))
        public_status = result['status']
    return {'status': public_status, 'mode': mode, 'score': selected,
            'f1': selected['f1'] if selected is not None else None,
            'dcist_percent': 100 * selected['f1'] if selected is not None else None,
            'time_enabled': work['time_enabled'], 'no_answer_policy': no_answer_policy,
            'extra_node_ratio': result.get('extra_node_ratio') if mode == 'reviewed' else None,
            'details': result}


def aggregate(results):
    require(bool(results), 'Empty cohort')
    require(len({r['mode'] for r in results}) == 1, 'Do not mix scoring modes')
    require(len({r['time_enabled'] for r in results}) == 1, 'Do not mix time conditions')
    pending = sum(r['f1'] is None for r in results)
    values = [r['f1'] for r in results if r['f1'] is not None]
    return {'status': 'review_required' if pending else 'complete', 'n': len(results),
            'n_scoreable': len(values), 'n_pending_review': pending,
            'n_no_answer': sum(r['status'] == 'no_answer_counted_zero' for r in results),
            'n_format_error': sum(r['status'] == 'completed_format_error' for r in results),
            'mode': results[0]['mode'], 'time_enabled': results[0]['time_enabled'],
            'recording_macro_f1': None if pending else statistics.mean(values),
            'dcist_percent': None if pending else 100 * statistics.mean(values),
            'tp': None if pending else sum(r['score']['tp'] for r in results),
            'gold': sum(r['score']['gold'] for r in results if r['score'] is not None),
            'gold_count_complete': not pending,
            'aggregation': 'recording-macro F1, not pooled micro-F1'}


def load_external(spec, base, mode):
    require(isinstance(spec, dict), 'Manifest item must be an object')
    require(type(spec.get('time_enabled')) is bool, 'Each item needs time_enabled')
    files = {}

    def locate(key):
        value = spec.get(key)
        require(isinstance(value, str) and value, f'Missing path: {key}')
        path = Path(value)
        path = path if path.is_absolute() else base / path
        require(path.is_file(), f'Missing input file for {key}')
        files[key] = path.resolve()
        return path

    case = {'gold': read_json(locate('gold')), 'time_enabled': spec['time_enabled'],
            'inference_status': spec.get('inference_status', 'completed')}
    if case['inference_status'] == 'no_answer':
        require(not spec.get('prediction'), 'Do not discard a prediction by marking no_answer')
        case['prediction'] = None
    else:
        # Keep complete malformed answers as model outcomes rather than input-file errors.
        case['prediction'] = locate('prediction').read_text(encoding='utf-8-sig')
    if spec.get('reviews'):
        require(mode == 'reviewed', 'Use --mode reviewed with a reviews file')
        reviews = read_json(locate('reviews'))
        require(isinstance(reviews, dict), 'Reviews must be a JSON object')
        require(set(reviews) <= set(DECISIONS), 'Unknown review keys; reviews cannot replace gold or settings')
        for key, value in reviews.items():
            require(isinstance(value, list), f'{key} must be an array')
            case[key] = value
    return case, files


def compute(args):
    protected = set()
    if args.manifest:
        require(not any((args.gold, args.prediction, args.reviews, args.case, args.time, args.inference_status)),
                '--manifest cannot be combined with single-case inputs/settings')
        manifest = read_json(args.manifest)
        require(isinstance(manifest, dict), 'Manifest must be an object')
        specs = manifest.get('samples')
        require(isinstance(specs, list) and specs, 'Manifest needs a nonempty samples array')
        require(all(isinstance(s, dict) for s in specs), 'Invalid manifest item')
        ids = [s.get('sample_id') for s in specs]
        require(all(isinstance(i, str) and i for i in ids), 'Every sample needs an ID')
        require(len(set(ids)) == len(ids), 'Duplicate sample_id')
        protected.add(args.manifest.resolve())
        items = []
        for spec in specs:
            case, files = load_external(spec, args.manifest.resolve().parent, args.mode)
            protected.update(files.values())
            items.append((spec['sample_id'], case, files))
    elif args.case:
        require(not any((args.gold, args.prediction, args.reviews, args.time, args.inference_status)),
                '--case already contains inputs and settings')
        protected.add(args.case.resolve())
        items = [('case', read_json(args.case), {'case': args.case.resolve()})]
    else:
        require(args.gold is not None and args.time is not None, 'Provide --gold and --time')
        spec = {'gold': str(args.gold), 'time_enabled': args.time == 'on',
                'inference_status': args.inference_status or 'completed'}
        if args.prediction:
            spec['prediction'] = str(args.prediction)
        if args.reviews:
            spec['reviews'] = str(args.reviews)
        case, files = load_external(spec, Path.cwd(), args.mode)
        protected.update(files.values())
        items = [('case', case, files)]
    if args.output:
        require(args.output.resolve() not in protected, 'Output must not overwrite an input')
        require(not args.output.exists(), 'Refusing to overwrite an existing output')
    before = {p: digest(p) for p in protected}
    results = []
    for sid, case, files in items:
        result = score_case(case, args.mode, args.no_answer_policy)
        result.update(sample_id=sid)
        results.append(result)
    require(all(digest(p) == h for p, h in before.items()), 'Input changed during scoring')
    return {'summary': aggregate(results), 'samples': results,
            'inputs_unchanged': True, 'model_calls': 0}
