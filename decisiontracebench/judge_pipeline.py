"""Prepare evidence, validate model adjudications, and invoke the DCIST scorer."""
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock

from . import evaluation as adapter, matching as core, schema, structure, money
from . import scoring as score
from .judge_client import ReviewError, require, response_document, strict_json

ROOT = Path(__file__).resolve().parent.parent
DECISION_KEYS = ('semantic_decisions', 'node_decisions', 'track_decisions')


def read_json(path):
    return strict_json(path.read_text(encoding='utf-8-sig'))


def write_once(path, value):
    """Reuse identical artifacts, but never replace a different request or decision."""
    rendered = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(read_json(path) == strict_json(rendered), 'Existing artifact differs; use a new run directory')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(rendered)


def code_snapshot():
    files = list((ROOT / 'decisiontracebench').glob('*.py')) + [ROOT / 'score.py', ROOT / 'judge.py']
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8-sig') for p in sorted(files)}


def clean_case(case):
    return {key: copy.deepcopy(case[key]) for key in ('inference_status', 'time_enabled', 'gold', 'prediction')}


def empty_reviews():
    return {key: [] for key in DECISION_KEYS}


def with_reviews(case, reviews):
    return dict(clean_case(case), **copy.deepcopy(reviews))


def prepare(manifest_path, run_dir):
    require(__debug__, 'Run Python without -O so scoring assertions remain enabled')
    require(not (run_dir / 'plan.json').exists(), 'Plan already exists; run it or use a new directory')
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and isinstance(manifest.get('samples'), list)
            and manifest['samples'], 'Manifest needs a nonempty samples array')
    base = manifest_path.resolve().parent
    samples, seen = [], set()
    for index, spec in enumerate(manifest['samples']):
        require(isinstance(spec, dict), 'Sample must be an object')
        sid = spec.get('sample_id')
        require(isinstance(sid, str) and sid.strip() and sid not in seen, 'Sample IDs must be unique')
        seen.add(sid)
        require(not spec.get('reviews'), 'Automatic review starts from predictions, not previous decisions')

        def text_file(key, required=True):
            name = spec.get(key)
            if not name and not required:
                return None
            require(isinstance(name, str) and name, f'Missing path: {key}')
            path = Path(name)
            path = path if path.is_absolute() else base / path
            require(path.is_file(), f'Missing file: {key}')
            content = path.read_text(encoding='utf-8-sig')
            require(bool(content.strip()), f'Empty file: {key}')
            return content

        case = {'inference_status': spec.get('inference_status', 'completed'),
                'time_enabled': spec.get('time_enabled'), 'gold': strict_json(text_file('gold')),
                'prediction': text_file('prediction')}
        baseline = adapter.evaluate(case)  # Validates GT even if the answer is malformed.
        task = text_file('generation_query')
        actual = text_file('actual_input', False)
        reference = text_file('reference', False)
        note = spec.get('actual_input_note', '')
        require(actual is not None or (reference is not None and isinstance(note, str) and note.strip()),
                'Supply actual_input, or reference plus an explicit actual_input_note for audio-only generation')
        if baseline['status'] != 'completed_format_error':
            case['prediction'] = schema.parse_json(case['prediction'])
        samples.append({'id': f'sample_{index + 1:04d}', 'sample_id': sid, 'case': case,
                        'generation_query': task, 'actual_input': actual, 'reference': reference,
                        'actual_input_note': note, 'format_error': baseline['status'] == 'completed_format_error'})
    require(len({s['case']['time_enabled'] for s in samples}) == 1, 'Use one time condition per run')
    plan = {'format': 'decisiontracebench.auto_review.v1', 'samples': samples,
            'prompts': {name: (ROOT / 'prompts' / (name + '.txt')).read_text(encoding='utf-8-sig')
                        for name in ('semantic_equivalence', 'extra_nodes')},
            'scoring_protocol': (ROOT / 'docs' / 'SCORING.md').read_text(encoding='utf-8-sig'),
            'engine_sources': code_snapshot()}
    write_once(run_dir / 'plan.json', plan)
    counts = []
    for sample in samples:
        if sample['format_error']:
            counts.append({'sample_id': sample['sample_id'], 'format_error': True, 'initial_requests': 0})
            continue
        keys, events = candidates(sample['case'])
        counts.append({'sample_id': sample['sample_id'], 'track_key_candidates': len(keys),
                       'event_candidates_upper_bound': len(events), 'initial_requests': len(keys),
                       'prediction_tracks': len(sample['case']['prediction']['tracks'])})
    report = {'status': 'prepared', 'network_calls': 0, 'samples': counts,
              'note': 'One request per semantic candidate; then one per track needing extra-node review. '
                      'Cross-field event requests depend on approved field correspondences.'}
    write_once(run_dir / 'preparation.json', report)
    return report


def value_possible(g, p):
    if core.same(money.normalize(g), money.normalize(p)):
        return True
    if g is None or p is None:
        return False
    if all(isinstance(v, dict) and set(v) == {'number', 'unit'} for v in (g, p)):
        return False
    return True


def candidate_id(dim, gi, pi, coords):
    suffix = ''.join(f'_{k}{v}' for k, v in coords.items())
    return f'{dim}_g{gi}_p{pi}{suffix}'


def candidates(case):
    """Enumerate occurrence-bound candidates under the protected event gates."""
    keys, events = [], []
    gold, pred, timed = case['gold'], case['prediction'], case['time_enabled']
    for gi, gt in enumerate(gold['tracks']):
        for pi, pt in enumerate(pred['tracks']):
            if structure.structural_violations(pt) or len(gt['cycles']) != len(pt['cycles']):
                continue
            is_same_key = core.key(gt) == core.key(pt)
            hard_pairs = []
            for ci, (gc, pc) in enumerate(zip(gt['cycles'], pt['cycles'])):
                pairs = [(gei, pei, ge, pe) for gei, ge in enumerate(gc['events'])
                         for pei, pe in enumerate(pc['events'])
                         if core.hard_event_equal(ge, pe, timed) and value_possible(ge['value'], pe['value'])]
                hard_pairs.append(pairs)
            # Cross-field recovery must admit an ordered complete required chain.
            full_possible = True
            for gc, pairs in zip(gt['cycles'], hard_pairs):
                cursor = -1
                for gei in range(len(gc['events'])):
                    options = [pei for gidx, pei, _, _ in pairs if gidx == gei and pei > cursor]
                    if not options:
                        full_possible = False
                        break
                    cursor = min(options)
            if not is_same_key and not full_possible:
                continue

            def entry(dim, gv, pv, coords):
                return dict(candidate_id=candidate_id(dim, gi, pi, coords), dimension=dim,
                            gold_track_index=gi, prediction_track_index=pi,
                            gold=copy.deepcopy(gv), prediction=copy.deepcopy(pv), **coords)

            if not is_same_key:
                keys.append(entry('track_key', list(core.key(gt)), list(core.key(pt)), {}))
            for ci, pairs in enumerate(hard_pairs):
                for gei, pei, ge, pe in pairs:
                    coords = dict(cycle_index=ci, gold_event_index=gei, prediction_event_index=pei)
                    if ge['scope_id'] != pe['scope_id']:
                        events.append(entry('scope', ge['scope_id'], pe['scope_id'], coords))
                    if not core.same(money.normalize(ge['value']), money.normalize(pe['value'])):
                        events.append(entry('value', ge['value'], pe['value'], coords))
    return keys, events


def source_packet(sample):
    return {'sample_id': sample['sample_id'], 'time_enabled': sample['case']['time_enabled'],
            'actual_input': sample['actual_input'], 'actual_input_note': sample['actual_input_note'],
            'reference': sample['reference'], 'gold': sample['case']['gold'],
            'prediction': sample['case']['prediction']}


def messages_for(plan, sample, kind, packets, reviews):
    template = plan['prompts'][kind]
    values = {'generation_query_and_taxonomy': sample['generation_query'],
              'scoring_protocol': json.dumps({'time_enabled': sample['case']['time_enabled'],
                                             'rules': plan['scoring_protocol']}, ensure_ascii=False),
              'candidate_packets': json.dumps(packets, ensure_ascii=False),
              'review_packets': json.dumps(packets, ensure_ascii=False),
              'approved_mapping_manifest': json.dumps([d for d in reviews['semantic_decisions']
                                                       if d['verdict'] == 'equivalent'], ensure_ascii=False)}
    marker = '【本次输入：由调用方完整填入，不把占位符发给模型】'
    require(marker in template, 'Review template is missing its input boundary')
    instruction, inputs = template.split(marker, 1)
    # Substitute only template tokens; strings inside the evidence are never reinterpreted.
    def replace(match):
        require(match.group(1) in values, 'Unknown template placeholder')
        return values[match.group(1)]
    rendered = re.sub(r'\{\{([a-z_]+)\}\}', replace, inputs)
    return [{'role': 'system', 'content': instruction.strip()},
            {'role': 'user', 'content': marker + rendered}]


def semantic_job(plan, sample, candidate, reviews):
    packets = dict(source_packet(sample), candidates=[candidate], quantity_output_contract={
        'note': 'Only when quantity representation moves: certify equality and provide complete '
                'path-bound pairs. This does not authorize changing numerical facts.',
        'optional_fields': ['quantity_equivalence_verified', 'quantity_equivalence_reason', 'quantity_alignment'],
        'alignment_entry': {'gold': {'path': ['object_key', 0], 'number': 'decimal string', 'unit': 'unit string'},
                            'prediction': {'path': ['object_key'], 'number': 'decimal string', 'unit': 'unit string',
                                           'surface': 'exact number+unit substring, for text endpoints only'}}})
    return {'id': sample['id'] + '__' + candidate['candidate_id'], 'kind': 'semantic',
            'candidate': candidate,
            'messages': messages_for(plan, sample, 'semantic_equivalence', packets, reviews)}


def require_text(value, name):
    require(isinstance(value, str) and value.strip(), f'Missing {name}')


def semantic_decision(document, candidate, sample, reviews):
    require_text(document.get('reviewer'), 'reviewer')
    rows = document.get('decisions')
    require(isinstance(rows, list) and len(rows) == 1, 'Expected one semantic decision')
    row = rows[0]
    require(isinstance(row, dict) and row.get('candidate_id') == candidate['candidate_id'],
            'Semantic candidate ID mismatch')
    require(row.get('verdict') in {'equivalent', 'not_equivalent', 'uncertain'}, 'Invalid semantic verdict')
    require_text(row.get('reason'), 'reason')
    require_text(row.get('evidence_quote'), 'evidence_quote')
    decision = dict(copy.deepcopy(candidate), verdict=row['verdict'], reason=row['reason'],
                    evidence_quote=row['evidence_quote'])
    for key in ('quantity_equivalence_verified', 'quantity_equivalence_reason', 'quantity_alignment'):
        if key in row:
            decision[key] = copy.deepcopy(row[key])
    trial = with_reviews(sample['case'], reviews)
    trial['semantic_decisions'].append(decision)
    adapter.approved_mappings(trial, trial['gold'], trial['prediction'], trial['time_enabled'])
    return decision


def extra_jobs(plan, sample, reviews):
    case = with_reviews(sample['case'], reviews)
    gold, pred, timed = case['gold'], case['prediction'], case['time_enabled']
    allows = adapter.approved_mappings(case, gold, pred, timed)
    metadata = adapter.base_sample(gold, pred, timed)['row']['tracks']
    already_matched = {m['prediction_track_index'] for m in adapter.evaluate(case).get('matches', [])}
    jobs = []
    for pi, pt in enumerate(pred['tracks']):
        meta = metadata[pi]
        if meta['structural_violations']:
            continue
        # A new raw-key track requires every node reviewed before whole-track exemption.
        track_required = not meta['gold_present'] and pi not in already_matched
        unresolved = set(adapter.positions(pt)) if track_required else set()
        alignments = []
        for gi, gt in enumerate(gold['tracks']):
            if core.key(gt) != core.key(pt) and ('case', gi, pi) not in allows['track_key']:
                continue
            alignment = core.align(gt, pt, timed, True, allows, 'case', gi, pi, 'joint')
            alignments.append({'gold_track_index': gi, 'alignment': alignment})
            anchors = {(c['cycle_index'], pei) for c in alignment['cycles'] for _, pei in c['matched']}
            unresolved.update(adapter.positions(pt) - anchors)
            if core.key(gt) == core.key(pt):
                base = core.align(gt, pt, timed, True, allows, 'case', gi, pi, 'baseline_money')
                if base['covered']:
                    anchors = {(c['cycle_index'], pei) for c in base['cycles'] for _, pei in c['matched']}
                    unresolved.update(adapter.positions(pt) - anchors)
        if not alignments:
            unresolved.update(adapter.positions(pt))
        if not unresolved and not track_required:
            continue
        rid = f'case/T{pi}'
        positions = [dict(cycle_index=ci, event_index=ei) for ci, ei in sorted(unresolved)]
        packet = dict(source_packet(sample), review_id=rid, prediction_track_index=pi,
                      gold_present=not track_required, raw_gold_present=meta['gold_present'], prediction_track=pt,
                      mechanical={'unresolved_nodes': positions, 'candidate_alignments': alignments,
                                  'structural_violations': meta['structural_violations']})
        jobs.append({'id': sample['id'] + f'__extra_p{pi}', 'kind': 'extra', 'pi': pi,
                     'review_id': rid, 'positions': sorted(unresolved),
                     'track_required': track_required,
                     'messages': messages_for(plan, sample, 'extra_nodes', packet, reviews)})
    return jobs


def extra_decisions(document, job, sample, reviews):
    require_text(document.get('reviewer'), 'reviewer')
    rows = document.get('decisions')
    require(isinstance(rows, list), 'Missing node decisions')
    expected = set(map(tuple, job['positions']))
    seen, nodes = set(), []
    for row in rows:
        require(isinstance(row, dict) and row.get('review_id') == job['review_id'], 'Review ID mismatch')
        ci, ei = row.get('cycle_index'), row.get('event_index')
        require(type(ci) is int and type(ei) is int, 'Node indices must be integers')
        pos = ci, ei
        require(pos in expected and pos not in seen, 'Unexpected or duplicate node decision')
        seen.add(pos)
        require(row.get('label') in adapter.NODE_LABELS, 'Invalid node label')
        for name in ('reason', 'rule_quote', 'evidence_quote'):
            require_text(row.get(name), name)
        nodes.append(dict(prediction_track_index=job['pi'], cycle_index=ci, event_index=ei,
                          **{k: row[k] for k in ('label', 'reason', 'rule_quote', 'evidence_quote')}))
    require(seen == expected, 'Missing node decisions; unreviewed is not uncertain')
    tracks = document.get('track_decisions', [])
    require(isinstance(tracks, list) and len(tracks) == int(job['track_required']),
            'Missing or unexpected track decision')
    imported_tracks = []
    if tracks:
        row = tracks[0]
        require(isinstance(row, dict) and row.get('review_id') == job['review_id'], 'Track ID mismatch')
        require(row.get('label') in adapter.TRACK_LABELS, 'Invalid track label')
        require_text(row.get('reason'), 'track reason')
        imported_tracks.append(dict(prediction_track_index=job['pi'], label=row['label'], reason=row['reason']))
    trial = with_reviews(sample['case'], reviews)
    trial['node_decisions'].extend(nodes)
    trial['track_decisions'].extend(imported_tracks)
    metadata = adapter.base_sample(trial['gold'], trial['prediction'], trial['time_enabled'])
    adapter.extra_reviews(trial, metadata, trial['prediction'])
    return {'node_decisions': nodes, 'track_decisions': imported_tracks}


class RequestStore:
    """Persist request attempts; never repeat successful or ambiguous calls implicitly."""
    def __init__(self, run_dir, client, max_requests, retry_failed=False):
        self.root, self.client = run_dir, client
        self.limit, self.retry_failed = max_requests, retry_failed
        self.calls, self.lock = 0, Lock()

    def execute(self, job, validate):
        directory = self.root / 'requests' / job['id']
        payload = self.client.payload(job['messages'])
        write_once(directory / 'request.json', payload)
        attempts = sorted(directory.glob('attempt_*'))
        if attempts:
            last = attempts[-1]
            if (last / 'validated.json').exists():
                value = validate(response_document(read_json(last / 'response.json')))
                require(value == read_json(last / 'validated.json'), 'Saved adjudication changed; refusing reuse')
                return value
            if not self.retry_failed:
                raise ReviewError('Previous attempt failed or has unknown outcome; explicit --retry-failed required')
        with self.lock:
            require(self.calls < self.limit, 'Request limit reached; resume explicitly to continue')
            self.calls += 1
        attempt = directory / f'attempt_{len(attempts) + 1:04d}'
        write_once(attempt / 'started.json', {'status': 'started', 'usage': 'unknown',
                                             'started_at': datetime.now(timezone.utc).isoformat()})
        try:
            envelope = self.client.complete(payload)
            write_once(attempt / 'response.json', envelope)
            decision = validate(response_document(envelope))
            write_once(attempt / 'validated.json', decision)
            return decision
        except Exception as exc:
            # Diagnostics never include transport URLs, headers or arbitrary exception reprs.
            reason = self.client.redact(str(exc)) if isinstance(exc, ReviewError) else 'Response failed schema or scoring validation'
            write_once(attempt / 'error.json', {'status': 'failed', 'reason': reason,
                                               'usage': 'see response.json if present; otherwise unknown'})
            raise ReviewError(reason) from None


def execute_phase(items, store, workers):
    def execute(item):
        state, job = item
        try:
            if job['kind'] == 'semantic':
                value = store.execute(job, lambda d: semantic_decision(d, job['candidate'], state['sample'], state['reviews']))
            else:
                value = store.execute(job, lambda d: extra_decisions(d, job, state['sample'], state['reviews']))
            return state, value, None
        except (ValueError, KeyError, TypeError, AssertionError, OSError):
            return state, None, job['id']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(execute, items))
    for state, value, failure in outcomes:
        if failure:
            state['failed_jobs'].append(failure)
        elif isinstance(value, dict) and 'candidate_id' in value:
            state['reviews']['semantic_decisions'].append(value)
        else:
            for key in ('node_decisions', 'track_decisions'):
                state['reviews'][key].extend(value[key])


def run_pipeline(run_dir, client, workers=4, max_requests=100, retry_failed=False):
    require(__debug__, 'Run Python without -O so scoring assertions remain enabled')
    plan = read_json(run_dir / 'plan.json')
    require(plan.get('engine_sources') == code_snapshot(), 'Code changed since preparation; create a new run')
    write_once(run_dir / 'model_settings.json', client.settings)
    store = RequestStore(run_dir, client, max_requests, retry_failed)
    states = [{'sample': s, 'reviews': empty_reviews(), 'failed_jobs': []} for s in plan['samples']]
    items = []
    for state in states:
        if state['sample']['format_error']:
            continue
        keys, events = candidates(state['sample']['case'])
        state['event_candidates'] = events
        items.extend((state, semantic_job(plan, state['sample'], c, state['reviews'])) for c in keys)
    execute_phase(items, store, workers)
    items = []
    for state in states:
        if state['sample']['format_error'] or state['failed_jobs']:
            continue
        approved = {(d['gold_track_index'], d['prediction_track_index']) for d in state['reviews']['semantic_decisions']
                    if d['dimension'] == 'track_key' and d['verdict'] == 'equivalent'}
        case = state['sample']['case']
        for c in state['event_candidates']:
            gi, pi = c['gold_track_index'], c['prediction_track_index']
            if core.key(case['gold']['tracks'][gi]) == core.key(case['prediction']['tracks'][pi]) or (gi, pi) in approved:
                items.append((state, semantic_job(plan, state['sample'], c, state['reviews'])))
    execute_phase(items, store, workers)
    items = []
    for state in states:
        if not state['sample']['format_error'] and not state['failed_jobs']:
            items.extend((state, job) for job in extra_jobs(plan, state['sample'], state['reviews']))
    execute_phase(items, store, workers)
    results, pending = [], []
    for state in states:
        sample = state['sample']
        if state['failed_jobs']:
            pending.append({'sample_id': sample['sample_id'], 'failed_or_pending_jobs': state['failed_jobs']})
            continue
        case = with_reviews(sample['case'], state['reviews'])
        result = score.score_case(case, 'reviewed')
        if result['f1'] is None:
            pending.append({'sample_id': sample['sample_id'], 'missing_reviews': result['details']['missing_reviews']})
            continue
        result['sample_id'] = sample['sample_id']
        write_once(run_dir / 'reviews' / (sample['id'] + '.json'), state['reviews'])
        write_once(run_dir / 'scoring_cases' / (sample['id'] + '.json'), case)
        write_once(run_dir / 'scores' / (sample['id'] + '.json'), result)
        results.append(result)
    summary = (score.aggregate(results) if not pending else
               {'status': 'review_required', 'n': len(states), 'n_complete': len(results),
                'n_pending': len(pending), 'dcist_percent': None, 'recording_macro_f1': None})
    output = {'summary': summary, 'samples': results, 'pending': pending, 'requests_this_run': store.calls}
    if not pending:
        write_once(run_dir / 'results.json', {'summary': summary, 'samples': results})
    return output
