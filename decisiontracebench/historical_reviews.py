"""Readers for archived candidate catalogs; not used by the public CLI pipeline."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from . import schema, money
from .matching import DIMENSIONS, key, same, hard_event_equal, empty_allows
from .quantities import quantity_approval_valid

ROOT = Path(__file__).resolve().parent.parent

def read(path):
    p=Path(path)
    return json.loads((p if p.is_absolute() else ROOT/p).read_text(encoding='utf-8-sig'))


def sha(path):
    p=Path(path)
    return hashlib.sha256((p if p.is_absolute() else ROOT/p).read_bytes()).hexdigest()


def key_payload(value):
    if isinstance(value, dict):
        return value['entity_id'], value['field_id']
    assert isinstance(value, (list, tuple)) and len(value) == 2
    return tuple(value)


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
