"""Ordered event alignment and one-to-one trajectory matching."""
from collections import deque
from functools import lru_cache
from . import schema, structure, timing, money

MODES = ("baseline_money", "value_only", "label_only", "joint")
DIMENSIONS = {"value", "scope", "track_key"}

def same(a, b):
    return schema.freeze(a) == schema.freeze(b)


def key(track):
    return track['entity_id'], track['field_id']


def approved_for_mode(mode):
    return {'baseline_money': set(), 'value_only': {'value'},
            'label_only': {'scope', 'track_key'}, 'joint': DIMENSIONS}[mode]


def hard_event_equal(g, p, time_enabled):
    if any(g.get(k) != p.get(k) for k in ('action', 'speaker_id', 'result')):
        return False
    if not time_enabled:
        return True
    try:
        return timing.time_match(p.get('start_ms'), p.get('end_ms'),
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
    norm = money.normalize if monetary else schema.normalize
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


def score_one(sample, gold, pred, nd, td, allows, mode, monetary=True):
    row, aid = sample['row'], sample['audit_id']
    ng, npred = row['gold_tracks'], row['prediction_tracks']
    if pred is None or row.get('schema_errors'):
        return {'audit_id': aid, 'score': schema.ratio(0, ng, npred), 'matches': [],
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
        basic = structure.structural_violations(pt)
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
    return {'audit_id': aid, 'score': schema.ratio(len(pairs), ng, denominator), 'matches': matches,
            'original_exempt_predictions': sorted(exempt), 'reinstated_predictions': reinstated,
            'candidate_conflicts': conflicts, 'blocked': blocked, 'format_gate_failed': False}


def empty_allows():
    return {d: {} for d in DIMENSIONS}
