"""Trajectory matching with occurrence-bound semantic equivalence.

Approved value, scope and track-key correspondences create matching edges
subject to action, speaker, cycle, structure and temporal constraints.
"""
import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
import rescore as legacy
import money_equivalent_values as money

official = legacy.official
MODES = ('baseline_money', 'value_only', 'label_only', 'joint')
DIMENSIONS = {'value', 'scope', 'track_key'}


def read(path):
    p=Path(path)
    return json.loads((p if p.is_absolute() else ROOT/p).read_text(encoding='utf-8-sig'))


def sha(path):
    p=Path(path)
    return hashlib.sha256((p if p.is_absolute() else ROOT/p).read_bytes()).hexdigest()


def same(a, b):
    return official.freeze(a) == official.freeze(b)


def key(track):
    return track['entity_id'], track['field_id']


def key_payload(value):
    if isinstance(value, dict):
        return value['entity_id'], value['field_id']
    assert isinstance(value, (list, tuple)) and len(value) == 2
    return tuple(value)


def approved_for_mode(mode):
    return {'baseline_money': set(), 'value_only': {'value'},
            'label_only': {'scope', 'track_key'}, 'joint': DIMENSIONS}[mode]


def hard_event_equal(g, p, time_enabled):
    if any(g.get(k) != p.get(k) for k in ('action', 'speaker_id', 'result')):
        return False
    if not time_enabled:
        return True
    try:
        return legacy.time_match(p.get('start_ms'), p.get('end_ms'),
                                 g.get('start_ms'), g.get('end_ms'))['pass']
    except (ValueError, TypeError, KeyError):
        return False


def align(g, p, time_enabled, monetary, allows, audit_id, gi, pi, mode):
    allowed = approved_for_mode(mode)
    track_coord = (audit_id, gi, pi)
    if key(g) != key(p) and not ('track_key' in allowed and track_coord in allows['track_key']):
        return {'covered': False, 'reason': 'field_mismatch', 'cycles': []}
    if len(g['cycles']) != len(p['cycles']):
        return {'covered': False, 'reason': 'cycle_mismatch', 'cycles': []}
    norm = money.normalize if monetary else official.normalize
    cycles = []
    for ci, (gc, pc) in enumerate(zip(g['cycles'], p['cycles'])):
        gold, pred = gc['events'], pc['events']

        def event_ok(a, b):
            ge, pe = gold[a], pred[b]
            coord = (audit_id, gi, pi, ci, a, b)
            if not hard_event_equal(ge, pe, time_enabled):
                return False
            if ge.get('scope_id') != pe.get('scope_id') and not (
                    'scope' in allowed and coord in allows['scope']):
                return False
            if not same(norm(ge['value']), norm(pe['value'])) and not (
                    'value' in allowed and coord in allows['value']):
                return False
            return True

        @lru_cache(None)
        def visit(a, b):
            if a == len(gold) or b == len(pred):
                return ()
            candidates = [visit(a + 1, b), visit(a, b + 1)]
            if event_ok(a, b):
                candidates.append(((a, b),) + visit(a + 1, b + 1))
            return min(candidates, key=lambda x: (-len(x), x))

        pairs = list(visit(0, 0))
        matched_p = {b for a, b in pairs}
        cycles.append({'cycle_index': ci, 'gold_count': len(gold),
                       'prediction_count': len(pred), 'matched': pairs,
                       'unmatched_prediction': [i for i in range(len(pred)) if i not in matched_p]})
    return {'covered': all(len(c['matched']) == c['gold_count'] for c in cycles), 'cycles': cycles}


def load_legacy_verdicts(manifest):
    nd, td = {}, {}
    for source, destination, coord in [
            (manifest['verdicts']['node'], nd,
             lambda d: (d['review_id'], d['cycle_index'], d['event_index'])),
            (manifest['verdicts']['track'], td, lambda d: d['review_id'])]:
        for item in source:
            ident = coord(item)
            if ident in destination:
                assert destination[ident] == item, ('conflicting_historical_review', ident)
            destination[ident] = item
    return nd, td


def effective_node_label(nd, rid, ci, ei):
    label = nd.get((rid, ci, ei), {}).get('label')
    return 'exempt_extra' if label == 'evidence_uncertain' else label


def original_exemption(track, nd, td):
    if track.get('gold_present') or track.get('structural_violations'):
        return False
    rid = track['review_id']
    known_wrong = any(effective_node_label(nd, rid, e['cycle_index'], e['event_index'])
                      in ('invalid_extra', 'replacement_error') for e in track.get('unresolved_nodes', []))
    label = td.get(rid, {}).get('label')
    if label == 'exempt_new_track':
        assert not known_wrong, ('conflicting_original_exemption', rid)
    return not known_wrong and label in ('exempt_new_track', 'evidence_uncertain')


def candidate_gate(alignment, track, nd, exempt_original):
    if track.get('structural_violations'):
        return 'wrong_query_structure', []
    if not alignment['covered']:
        return 'mandatory_gold_not_covered', []
    missing, wrong = [], []
    for cycle in alignment['cycles']:
        for ei in cycle['unmatched_prediction']:
            ci = cycle['cycle_index']
            label = effective_node_label(nd, track['review_id'], ci, ei)
            if label in ('invalid_extra', 'replacement_error'):
                wrong.append({'cycle_index': ci, 'event_index': ei, 'label': label})
            elif label != 'exempt_extra' and not exempt_original:
                missing.append({'cycle_index': ci, 'event_index': ei})
    if wrong:
        return 'wrong_additional_node', wrong
    if missing:
        return 'displaced_anchor_requires_review', missing
    return 'matched', []


def min_cost_maximum_matching(adjacency, gold_count, original_exempt):
    """Maximum TP, then minimum reinstated exemptions, with deterministic ties."""
    npred = len(adjacency)
    source, sink = npred + gold_count, npred + gold_count + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(a, b, capacity, cost):
        graph[a].append([b, len(graph[b]), capacity, cost])
        graph[b].append([a, len(graph[a]) - 1, 0, -cost])

    refs = []
    for pi, choices in enumerate(adjacency):
        edge(source, pi, 1, int(pi in original_exempt))
        for gi in sorted(choices):
            refs.append((pi, gi, len(graph[pi])))
            edge(pi, npred + gi, 1, 0)
    for gi in range(gold_count):
        edge(npred + gi, sink, 1, 0)
    while True:
        distance, parent = [float('inf')] * len(graph), [None] * len(graph)
        distance[source] = 0
        queue, queued = deque([source]), {source}
        while queue:
            a = queue.popleft()
            queued.discard(a)
            for ei, item in enumerate(graph[a]):
                b, reverse, capacity, cost = item
                if capacity and distance[a] + cost < distance[b]:
                    distance[b], parent[b] = distance[a] + cost, (a, ei)
                    if b not in queued:
                        queue.append(b)
                        queued.add(b)
        if parent[sink] is None:
            break
        b = sink
        while b != source:
            a, ei = parent[b]
            item = graph[a][ei]
            item[2] -= 1
            graph[b][item[1]][2] += 1
            b = a
    return sorted([(gi, pi) for pi, gi, ei in refs if graph[pi][ei][2] == 0])


def parse_sample(sample):
    row = sample['row']
    gold = read(row['gold_path'])
    assert len(gold['tracks']) == row['gold_tracks']
    assert len({key(t) for t in gold['tracks']}) == len(gold['tracks'])
    assert not official.validate(gold), ('invalid_gold', sample['audit_id'])
    try:
        pred = official.parse_json(read(row['prediction_path'])['content'])
    except (ValueError, TypeError, KeyError):
        assert row.get('unparseable'), ('unexpected_parse_failure', sample['audit_id'])
        return gold, None
    assert not row.get('unparseable'), ('stored_parse_flag_mismatch', sample['audit_id'])
    assert official.validate(pred) == row.get('schema_errors', []), ('schema_gate_changed', sample['audit_id'])
    assert len(pred.get('tracks', [])) == row['prediction_tracks']
    return gold, pred


def score_one(sample, gold, pred, nd, td, allows, mode, monetary=True):
    row, aid = sample['row'], sample['audit_id']
    ng, npred = row['gold_tracks'], row['prediction_tracks']
    if pred is None or row.get('schema_errors'):
        return {'audit_id': aid, 'score': official.ratio(0, ng, npred), 'matches': [],
                'original_exempt_predictions': [], 'reinstated_predictions': [],
                'candidate_conflicts': [], 'blocked': [], 'format_gate_failed': True}
    meta = {t['prediction_track_index']: t for t in row['tracks']}
    assert set(meta) == set(range(npred)), ('missing_track_metadata', aid)
    exempt = {pi for pi, t in meta.items() if original_exemption(t, nd, td)}
    adjacency = [[] for _ in range(npred)]
    details, blocked = {}, []
    for pi, pt in enumerate(pred['tracks']):
        mt = meta[pi]
        assert key(pt) == key(mt)
        # Raw shape and all persisted query gates are immutable.
        basic = legacy.structural_violations(pt)
        assert all(v in mt.get('structural_violations', []) for v in basic), ('lost_structural_gate', aid, pi)
        for gi, gt in enumerate(gold['tracks']):
            if key(pt) != key(gt) and (aid, gi, pi) not in allows['track_key']:
                continue
            if key(pt) != key(gt) and 'track_key' not in approved_for_mode(mode):
                continue
            aligned = align(gt, pt, sample['time_enabled'], monetary, allows, aid, gi, pi, mode)
            gate, why = candidate_gate(aligned, mt, nd, pi in exempt)
            selected_alignment, fallback = aligned, False
            if gate != 'matched' and mode != 'baseline_money' and key(gt) == key(pt):
                # Added alternatives must not invalidate an already legal exact alignment.
                original = align(gt, pt, sample['time_enabled'], monetary, allows, aid, gi, pi, 'baseline_money')
                original_gate, _ = candidate_gate(original, mt, nd, pi in exempt)
                if original_gate == 'matched':
                    selected_alignment, fallback = original, True
            if gate == 'matched' or fallback:
                adjacency[pi].append(gi)
                details[(gi, pi)] = {'alignment': selected_alignment, 'kept_baseline_alignment': fallback}
            if gate == 'displaced_anchor_requires_review':
                blocked.append({'gold_track_index': gi, 'prediction_track_index': pi,
                                'reason': gate, 'nodes': why})
    pairs = min_cost_maximum_matching(adjacency, ng, exempt)
    reinstated = sorted({pi for gi, pi in pairs} & exempt)
    denominator = npred - len(exempt) + len(reinstated)
    matches = []
    for gi, pi in pairs:
        detail = details[(gi, pi)]
        used = set()
        if not detail['kept_baseline_alignment']:
            if key(gold['tracks'][gi]) != key(pred['tracks'][pi]):
                used.update(allows['track_key'].get((aid, gi, pi), []))
            for cycle in detail['alignment']['cycles']:
                ci = cycle['cycle_index']
                for gei, pei in cycle['matched']:
                    ge = gold['tracks'][gi]['cycles'][ci]['events'][gei]
                    pe = pred['tracks'][pi]['cycles'][ci]['events'][pei]
                    coord = (aid, gi, pi, ci, gei, pei)
                    if 'value' in approved_for_mode(mode) and not same(money.normalize(ge['value']), money.normalize(pe['value'])):
                        used.update(allows['value'].get(coord, []))
                    if 'scope' in approved_for_mode(mode) and ge['scope_id'] != pe['scope_id']:
                        used.update(allows['scope'].get(coord, []))
        matches.append({'gold_track_index': gi, 'prediction_track_index': pi,
                        'approval_ids': sorted(used), **detail})
    conflicts = [{'gold_track_index': gi, 'prediction_track_indices': [pi for pi, a in enumerate(adjacency) if gi in a]}
                 for gi in range(ng) if sum(gi in a for a in adjacency) > 1]
    return {'audit_id': aid, 'score': official.ratio(len(pairs), ng, denominator), 'matches': matches,
            'original_exempt_predictions': sorted(exempt), 'reinstated_predictions': reinstated,
            'candidate_conflicts': conflicts, 'blocked': blocked, 'format_gate_failed': False}


def empty_allows():
    return {d: {} for d in DIMENSIONS}


def load_approvals(catalog, review_dir, samples, parsed):
    candidates = {c['candidate_id']: c for c in catalog['candidates']}
    assert len(candidates) == len(catalog['candidates']), 'duplicate_candidate_id'
    decisions, provenance = {}, []
    for path in sorted(Path(review_dir).glob('*.json')):
        review = read(path)
        assert review.get('reviewer'), ('missing_reviewer', str(path))
        provenance.append({'path': str(path), 'sha256': sha(path), 'reviewer': review['reviewer']})
        for d in review['decisions']:
            cid = d['candidate_id']
            assert cid in candidates and cid not in decisions, ('unknown_or_duplicate_decision', cid)
            assert d['verdict'] in ('equivalent', 'not_equivalent', 'uncertain')
            assert d.get('reason') and d.get('evidence_quote'), ('missing_review_basis', cid)
            decisions[cid] = d
    supplement_path = Path(review_dir).parent / 'quantity_supplements.json'
    if supplement_path.exists():
        supplement = read(supplement_path)
        assert supplement.get('reviewer')
        provenance.append({'path': str(supplement_path), 'sha256': sha(supplement_path),
                           'reviewer': supplement['reviewer']})
        supplemented = set()
        for item in supplement['decisions']:
            cid = item['candidate_id']
            assert cid in decisions and cid not in supplemented
            assert decisions[cid]['verdict'] == 'equivalent'
            assert decisions[cid].get('quantity_equivalence_verified') is True
            assert decisions[cid].get('quantity_equivalence_reason')
            assert not decisions[cid].get('quantity_alignment')
            decisions[cid]['quantity_alignment'] = item['quantity_alignment']
            supplemented.add(cid)
    lookup = {s['audit_id']: s for s in samples}
    assert len(lookup) == len(samples), 'duplicate_sample_id'
    allows, occurrences = empty_allows(), 0
    for cid, candidate in candidates.items():
        dim = candidate['dimension']
        assert dim in DIMENSIONS and candidate['occurrences']
        seen = set()
        for occurrence in candidate['occurrences']:
            aid = occurrence['audit_id']
            sample = lookup[aid]
            assert candidate['sample_id'] == sample['row']['sample_id']
            gold, pred = parsed[aid]
            assert pred is not None and not sample['row'].get('schema_errors'), ('candidate_on_bad_schema', cid)
            gi, pi = occurrence['gold_track_index'], occurrence['prediction_track_index']
            gt, pt = gold['tracks'][gi], pred['tracks'][pi]
            mt = next(t for t in sample['row']['tracks'] if t['prediction_track_index'] == pi)
            assert not mt.get('structural_violations'), ('candidate_on_bad_structure', cid)
            if dim == 'track_key':
                coord = aid, gi, pi
                assert key_payload(candidate['gold']) == key(gt)
                assert key_payload(candidate['prediction']) == key(pt)
            else:
                ci, gei, pei = (occurrence[k] for k in ('cycle_index', 'gold_event_index', 'prediction_event_index'))
                coord = aid, gi, pi, ci, gei, pei
                ge, pe = gt['cycles'][ci]['events'][gei], pt['cycles'][ci]['events'][pei]
                assert hard_event_equal(ge, pe, sample['time_enabled']), ('protected_mismatch_candidate', cid)
                if dim == 'scope':
                    assert candidate['gold'] == ge['scope_id'] and candidate['prediction'] == pe['scope_id']
                else:
                    assert same(money.normalize(candidate['gold']), money.normalize(ge['value']))
                    assert same(money.normalize(candidate['prediction']), money.normalize(pe['value']))
                    # Structured amount/unit pairs must remain mechanically equal.
                    # Textual spellings/reordering need an explicit, individual
                    # quantity check; semantic approval alone cannot bypass this.
                    if decisions.get(cid, {}).get('verdict') == 'equivalent':
                        assert quantity_approval_valid(ge['value'], pe['value'], decisions[cid]), ('numeric_change_approval', cid)
                        assert ge['value'] is not None and pe['value'] is not None, ('null_value_approval', cid)
            assert coord not in seen, ('duplicate_occurrence', cid, coord)
            seen.add(coord)
            occurrences += 1
            if decisions.get(cid, {}).get('verdict') == 'equivalent':
                allows[dim].setdefault(coord, []).append(cid)
    dependency = dependency_coverage(catalog, decisions)
    coverage = {'candidates': len(candidates), 'reviewed': len(decisions),
                'unreviewed': dependency['unreviewed_required'],
                'verdict_counts': dict(Counter(d['verdict'] for d in decisions.values())),
                'occurrences': occurrences, 'complete': dependency['complete'],
                'dependency_audit': dependency}
    return allows, coverage, provenance


def dependency_coverage(catalog, decisions):
    def pair(c):
        return (c['sample_id'], tuple(c.get('gold_key', c.get('gold', []))),
                tuple(c.get('prediction_key', c.get('prediction', []))))
    keys = {}
    for candidate in catalog['candidates']:
        if candidate['dimension'] == 'track_key':
            keys.setdefault(pair(candidate), []).append(candidate['candidate_id'])
    required, blocked, pending = [], [], []
    for candidate in catalog['candidates']:
        cid = candidate['candidate_id']
        # Small self-contained tests need not repeat keys on same-field records.
        same_key = ('gold_key' not in candidate or
                    candidate['gold_key'] == candidate['prediction_key'])
        if candidate['dimension'] == 'track_key' or same_key:
            required.append(cid)
            continue
        parent_ids = keys.get(pair(candidate), [])
        labels = [decisions.get(k, {}).get('verdict') for k in parent_ids]
        if any(v == 'equivalent' for v in labels):
            required.append(cid)
        elif not parent_ids or all(v in ('not_equivalent', 'uncertain') for v in labels):
            blocked.append({'candidate_id': cid, 'parent_candidate_ids': parent_ids,
                            'parent_verdicts': labels, 'status': 'dependency_blocked',
                            'reason': 'track_key_not_approved' if parent_ids else 'no_track_key_candidate',
                            'semantic_verdict': None})
        else:
            pending.append({'candidate_id': cid, 'parent_candidate_ids': parent_ids,
                            'status': 'dependency_pending'})
    unreviewed = sorted(set(required) - set(decisions))
    return {'required_candidates': len(required), 'reviewed_required': len(set(required) & set(decisions)),
            'unreviewed_required': unreviewed, 'dependency_blocked_count': len(blocked),
            'dependency_pending_count': len(pending), 'dependency_blocked': blocked,
            'dependency_pending': pending, 'complete': not unreviewed and not pending,
            'policy': 'Dependency blocking is not a semantic decision; unresolved keys remain pending.'}


def protected_quantities(value):
    """Canonical structured numeric leaves, invariant to object key order only.

    Numeric-bearing free text must preserve its literal quantities too. This is
    deliberately conservative: semantic review cannot add/drop facts or convert
    non-money units. Pure representation changes remain possible after review.
    """
    import re
    value = money.normalize(value)
    out = []

    def walk(v, path):
        if isinstance(v, dict):
            if 'number' in v or 'unit' in v:
                out.append((path, 'quantity', official.freeze(v.get('number')), official.freeze(v.get('unit'))))
            for k in sorted(v):
                if k not in ('number', 'unit'):
                    walk(v[k], path + (k,))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                walk(item, path + (i,))
        elif isinstance(v, str):
            tokens = re.findall(r'\d+(?:\.\d+)?', v)
            if tokens:
                out.append((path, 'text_digits', tuple(tokens)))
        elif type(v) in (int, float):
            out.append((path, 'number_literal', official.freeze(v)))
    walk(value, ())
    return out


def structured_quantities(value):
    """Quantity multiset with object-key paths; list permutation is not a gate.

    This does not make arrays equal in the scorer. It only checks that an
    explicitly approved semantic pair did not change/drop structured quantities.
    """
    out = []

    def walk(v, path):
        if isinstance(v, dict):
            if 'number' in v or 'unit' in v:
                out.append((path, 'quantity', official.freeze(v.get('number')), official.freeze(v.get('unit'))))
            for k in sorted(v):
                if k not in ('number', 'unit'):
                    walk(v[k], path + (k,))
        elif isinstance(v, list):
            for item in v:
                walk(item, path + ('[]',))
        elif type(v) in (int, float):
            out.append((path, 'number_literal', official.freeze(v)))
    walk(money.normalize(value), ())
    return sorted(out, key=repr)


def quantity_approval_valid(gold, prediction, decision):
    if protected_quantities(gold) == protected_quantities(prediction):
        return True
    explicitly_checked = (decision.get('quantity_equivalence_verified') is True
                          and bool(decision.get('quantity_equivalence_reason', '').strip()))
    if not explicitly_checked:
        return False
    if structured_quantities(gold) == structured_quantities(prediction):
        return True
    return explicit_quantity_alignment_valid(gold, prediction, decision.get('quantity_alignment', []))


def explicit_quantity_alignment_valid(gold, prediction, alignment):
    """Validate individually supplied numeric object-to-text correspondences.

    Not a number-bag matcher: every structured quantity must be named by exact
    path once, and text ends must contain the declared number+unit surface.
    Values are compared with the currency normalizer. The reviewer
    still must certify complete semantic equivalence (including all other text).
    """
    if not alignment:
        return False

    def structured_paths(value, path=()):
        paths = set()
        if isinstance(value, dict):
            if 'number' in value or 'unit' in value:
                paths.add(path)
            for k, item in value.items():
                if k not in ('number', 'unit'):
                    paths.update(structured_paths(item, path + (k,)))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                paths.update(structured_paths(item, path + (i,)))
        return paths

    def endpoint(value, spec):
        path = tuple(spec['path'])
        raw = value
        for segment in path:
            raw = raw[segment]
        declared = {'number': spec['number'], 'unit': spec['unit']}
        canonical = money.normalize(declared)
        if isinstance(raw, dict):
            assert 'number' in raw and 'unit' in raw
            assert same(canonical, money.normalize({'number': raw['number'], 'unit': raw['unit']}))
            return path, canonical, True
        assert isinstance(raw, str)
        surface = spec['surface']
        assert isinstance(surface, str) and surface and surface in raw
        literal = ''.join(surface.split()) == ''.join((str(spec['number']) + str(spec['unit'])).split())
        if not literal:
            # These exceptional surfaces are opt-in on a specific already
            # reviewed candidate, never an automatic global equivalence rule.
            assert canonical.get('number') == '0' and spec.get('surface_equivalence_reason')
            form = spec.get('zero_surface_kind')
            assert ((form == 'chinese_zero_amount' and surface == '零' + str(spec['unit'])) or
                    (form == 'no_disbursement' and surface == '未放款' and raw == '未放款'))
        return (path, surface), canonical, False

    # Numerical literals outside number/unit objects cannot disappear by moving
    # some other amount into a text field.
    literals = lambda value: [q for q in structured_quantities(value) if q[1] == 'number_literal']
    if literals(gold) != literals(prediction):
        return False
    structured_seen = [set(), set()]
    all_seen = [set(), set()]
    try:
        for pair in alignment:
            ends = [endpoint(gold, pair['gold']), endpoint(prediction, pair['prediction'])]
            if not same(ends[0][1], ends[1][1]):
                return False
            for side, (identity, canonical, structured) in enumerate(ends):
                if identity in all_seen[side]:
                    return False
                all_seen[side].add(identity)
                if structured:
                    structured_seen[side].add(identity)
    except (AssertionError, KeyError, TypeError, IndexError, ValueError):
        return False
    return structured_seen == [structured_paths(gold), structured_paths(prediction)]


def verify_frozen(manifest):
    for path, expected in manifest['frozen_files'].items():
        assert sha(path) == expected, ('frozen_source_changed', path)
