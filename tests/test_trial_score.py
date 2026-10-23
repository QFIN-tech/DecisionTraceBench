"""Regression tests for trajectory matching and adjudication constraints."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import trial_score as s


def event(eid='E1', value=None, action='start', speaker='S1', scope='scope', source=None, **extra):
    return dict(event_id=eid, action=action, speaker_id=speaker, scope_id=scope,
                source_event_id=source, value={'status': '原文'} if value is None else value,
                start_ms=0, end_ms=1000, **extra)


def track(entity='entity', field='field', evs=None):
    return {'entity_id': entity, 'field_id': field,
            'cycles': [{'cycle_id': 'C1', 'events': evs or [event()]}]}


def fixture(gold=None, pred=None, nd=None, td=None, time_enabled=True):
    gold, pred = gold or [track()], pred or [track()]
    rows = []
    for i, p in enumerate(pred):
        g = next((g for g in gold if s.key(g) == s.key(p)), None)
        al = s.align(g, p, time_enabled, True, s.empty_allows(), 'a', 0, i, 'baseline_money') if g else {'covered': False, 'cycles': []}
        unmatched = [dict(cycle_index=c['cycle_index'], event_index=e) for c in al['cycles'] for e in c['unmatched_prediction']]
        if g is None:
            unmatched = [dict(cycle_index=ci, event_index=ei) for ci, c in enumerate(p['cycles']) for ei, e in enumerate(c['events'])]
        rows.append({'review_id': f'a/T{i}', 'prediction_track_index': i, 'entity_id': p['entity_id'],
                     'field_id': p['field_id'], 'gold_present': g is not None,
                     'structural_violations': s.legacy.structural_violations(p), 'unresolved_nodes': unmatched})
    sample = {'audit_id': 'a', 'source_group': 'test', 'time_enabled': time_enabled,
              'row': {'gold_tracks': len(gold), 'prediction_tracks': len(pred), 'schema_errors': s.official.validate({'tracks': pred}),
                      'tracks': rows, 'sample_id': 'fsb_test'}}
    return sample, {'tracks': gold}, {'tracks': pred}, nd or {}, td or {}


def score(f, mode='baseline_money', allows=None):
    return s.score_one(*f, allows or s.empty_allows(), mode)


class TrialTests(unittest.TestCase):
    def test_exact_baseline(self):
        self.assertEqual(score(fixture())['score']['f1'], 1)

    def test_money_equivalence(self):
        f = fixture([track(evs=[event(value={'number': '5', 'unit': '万元'})])],
                    [track(evs=[event(value={'number': '50000', 'unit': '元'})])])
        self.assertEqual(score(f)['score']['f1'], 1)

    def test_free_text_not_equal_by_default(self):
        f = fixture(pred=[track(evs=[event(value={'status': '同义原文'})])])
        self.assertEqual(score(f)['score']['tp'], 0)

    def test_occurrence_value_approval_only(self):
        f = fixture(pred=[track(evs=[event(value={'status': '同义原文'})])])
        a = s.empty_allows(); a['value'][('a', 0, 0, 0, 0, 0)] = ['value1']
        self.assertEqual(score(f, 'value_only', a)['score']['tp'], 1)
        self.assertEqual(score(f, 'label_only', a)['score']['tp'], 0)
        self.assertEqual(score(f, 'value_only', a)['matches'][0]['approval_ids'], ['value1'])

    def test_scope_and_joint(self):
        f = fixture(pred=[track(evs=[event(value={'status': '同义'}, scope='alias')])])
        a = s.empty_allows(); coord = ('a', 0, 0, 0, 0, 0)
        a['value'][coord] = ['v']; a['scope'][coord] = ['sc']
        self.assertEqual(score(f, 'value_only', a)['score']['tp'], 0)
        self.assertEqual(score(f, 'label_only', a)['score']['tp'], 0)
        self.assertEqual(score(f, 'joint', a)['score']['tp'], 1)

    def test_action_is_not_relaxed(self):
        f = fixture(pred=[track(evs=[event(action='hold')])])
        a = s.empty_allows(); a['value'][('a', 0, 0, 0, 0, 0)] = ['v']
        self.assertEqual(score(f, 'joint', a)['score']['tp'], 0)

    def test_speaker_is_not_relaxed(self):
        f = fixture(pred=[track(evs=[event(speaker='S2')])])
        self.assertEqual(score(f, 'joint')['score']['tp'], 0)

    def test_result_is_not_relaxed(self):
        f = fixture([track(evs=[event(action='settle', result='accepted')])],
                    [track(evs=[event(action='settle', result='rejected')])])
        self.assertEqual(score(f, 'joint')['score']['tp'], 0)

    def test_time80_is_preserved(self):
        p = event(); p['end_ms'] = 1250
        self.assertEqual(score(fixture(pred=[track(evs=[p])]))['score']['tp'], 1)
        p['end_ms'] = 1251
        self.assertEqual(score(fixture(pred=[track(evs=[p])]))['score']['tp'], 0)

    def test_oracle_time_disabled(self):
        p = event(); p.pop('start_ms'); p.pop('end_ms')
        self.assertEqual(score(fixture(pred=[track(evs=[p])], time_enabled=False))['score']['tp'], 1)

    def test_source_gate_is_preserved(self):
        p = event(source='broken')
        self.assertEqual(score(fixture(pred=[track(evs=[p])]), 'joint')['score']['tp'], 0)

    def test_cycle_count_preserved(self):
        p = track(); p['cycles'].append({'cycle_id': 'C2', 'events': [event('E2')]})
        self.assertEqual(score(fixture(pred=[p]), 'joint')['score']['tp'], 0)

    def test_one_prediction_cannot_cover_two_gold(self):
        g = [track(), track(field='other')]; p = [track(entity='alias')]
        f = fixture(g, p); a = s.empty_allows()
        a['track_key'][('a', 0, 0)] = ['k0']; a['track_key'][('a', 1, 0)] = ['k1']
        self.assertEqual(score(f, 'label_only', a)['score']['tp'], 1)

    def test_two_predictions_cannot_cover_one_gold(self):
        f = fixture(pred=[track(), track(evs=[event('E2')])])
        result = score(f)
        self.assertEqual(result['score']['tp'], 1)
        self.assertEqual(result['score']['pred'], 2)
        self.assertEqual(len(result['candidate_conflicts']), 1)

    def test_exempt_new_track_reenters_denominator_when_matched(self):
        td = {'a/T0': {'label': 'exempt_new_track'}}
        f = fixture(pred=[track(entity='alias')], td=td)
        self.assertEqual(score(f)['score']['pred'], 0)
        a = s.empty_allows(); a['track_key'][('a', 0, 0)] = ['k']
        result = score(f, 'label_only', a)
        self.assertEqual(result['score']['tp'], 1)
        self.assertEqual(result['score']['pred'], 1)
        self.assertEqual(result['reinstated_predictions'], [0])

    def test_prefer_already_counted_prediction(self):
        pairs = s.min_cost_maximum_matching([[0], [0]], 1, {0})
        self.assertEqual(pairs, [(0, 1)])

    def test_matching_uses_augmenting_paths(self):
        self.assertEqual(len(s.min_cost_maximum_matching([[0, 1], [0]], 2, set())), 2)

    def test_known_extra_error_still_blocks(self):
        p = track(evs=[event(), event('E2', action='advance', source='E1')])
        nd = {('a/T0', 0, 1): {'label': 'invalid_extra'}}
        self.assertEqual(score(fixture(pred=[p], nd=nd), 'joint')['score']['tp'], 0)

    def test_reviewed_uncertain_extra_remains_exempt(self):
        p = track(evs=[event(), event('E2', action='advance', source='E1')])
        nd = {('a/T0', 0, 1): {'label': 'evidence_uncertain'}}
        self.assertEqual(score(fixture(pred=[p], nd=nd))['score']['tp'], 1)

    def test_unreviewed_extra_not_exempt(self):
        p = track(evs=[event(), event('E2', action='advance', source='E1')])
        self.assertEqual(score(fixture(pred=[p]))['score']['tp'], 0)

    def test_displaced_anchor_uses_legal_baseline_not_automatic_exemption(self):
        p = track(evs=[event(value={'status': '同义'}), event('E2', source='E1')])
        nd = {('a/T0', 0, 0): {'label': 'exempt_extra'}}
        f = fixture(pred=[p], nd=nd)
        a = s.empty_allows(); a['value'][('a', 0, 0, 0, 0, 0)] = ['v']
        result = score(f, 'value_only', a)
        self.assertEqual(result['score']['tp'], 1)
        self.assertTrue(result['matches'][0]['kept_baseline_alignment'])
        self.assertEqual(result['blocked'][0]['reason'], 'displaced_anchor_requires_review')

    def test_numeric_guard_and_object_order(self):
        self.assertEqual(s.protected_quantities({'amount': {'number': '5', 'unit': '万'}, 'status': 'a'}),
                         s.protected_quantities({'status': 'b', 'amount': {'unit': '元', 'number': '50000'}}))
        self.assertNotEqual(s.protected_quantities({'number': '5', 'unit': '个月'}),
                            s.protected_quantities({'number': '6', 'unit': '个月'}))

    def test_equivalence_uncertainty_not_approved(self):
        f = fixture(pred=[track(evs=[event(value={'status': '同义原文'})])])
        sample, gold, pred, _, _ = f
        candidate = {'candidate_id': 'c', 'dimension': 'value', 'sample_id': 'fsb_test',
                     'gold': {'status': '原文'}, 'prediction': {'status': '同义原文'},
                     'occurrences': [{'audit_id': 'a', 'gold_track_index': 0, 'prediction_track_index': 0,
                                      'cycle_index': 0, 'gold_event_index': 0, 'prediction_event_index': 0}]}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'review.json').write_text(json.dumps({'reviewer': 'test', 'decisions': [
                {'candidate_id': 'c', 'verdict': 'uncertain', 'reason': '不确定', 'evidence_quote': '原文'}]}), encoding='utf-8')
            allows, coverage, _ = s.load_approvals({'candidates': [candidate]}, folder, [sample], {'a': (gold, pred)})
        self.assertTrue(coverage['complete'])
        self.assertFalse(allows['value'])

    def test_text_quantity_spelling_needs_explicit_check(self):
        g, p = {'term': '6个月'}, {'term': '六个月'}
        self.assertFalse(s.quantity_approval_valid(g, p, {'verdict': 'equivalent'}))
        d = {'quantity_equivalence_verified': True, 'quantity_equivalence_reason': '六个月与6个月相同'}
        self.assertTrue(s.quantity_approval_valid(g, p, d))
        self.assertFalse(s.quantity_approval_valid({'number': '6', 'unit': '个月'},
                                                  {'number': '7', 'unit': '个月'}, d))

    def test_list_order_only_allowed_with_explicit_quantity_check(self):
        g, p = {'items': ['9月报表', '材料']}, {'items': ['材料', '9月报表']}
        self.assertFalse(s.quantity_approval_valid(g, p, {}))
        self.assertTrue(s.quantity_approval_valid(g, p, {
            'quantity_equivalence_verified': True, 'quantity_equivalence_reason': '同一9月报表，仅列表次序变化'}))

    def test_dependencies_pending_blocked_and_approved(self):
        pair = {'sample_id': 's', 'gold_key': ['g', 'f'], 'prediction_key': ['p', 'f']}
        catalog = {'candidates': [dict(pair, candidate_id='k', dimension='track_key'),
                                  dict(pair, candidate_id='v', dimension='value'),
                                  dict(pair, candidate_id='s', dimension='scope')]}
        pending = s.dependency_coverage(catalog, {})
        self.assertFalse(pending['complete'])
        self.assertEqual(pending['dependency_pending_count'], 2)
        for verdict in ('uncertain', 'not_equivalent'):
            blocked = s.dependency_coverage(catalog, {'k': {'verdict': verdict}})
            self.assertTrue(blocked['complete'])
            self.assertEqual(blocked['dependency_blocked_count'], 2)
        approved = s.dependency_coverage(catalog, {'k': {'verdict': 'equivalent'}})
        self.assertFalse(approved['complete'])
        self.assertEqual(approved['unreviewed_required'], ['s', 'v'])

    def test_same_key_dependencies_need_review_even_without_key_candidate(self):
        catalog = {'candidates': [{'candidate_id': 'v', 'sample_id': 's', 'dimension': 'value',
                                  'gold_key': ['x', 'f'], 'prediction_key': ['x', 'f']}]}
        result = s.dependency_coverage(catalog, {})
        self.assertFalse(result['complete'])
        self.assertEqual(result['unreviewed_required'], ['v'])

    def test_individually_aligned_structured_amount_to_text(self):
        g, p = {'number': '0', 'unit': '万'}, {'status': '实际已放款0万元'}
        d = {'quantity_equivalence_verified': True, 'quantity_equivalence_reason': '同一实际放款金额均为0',
             'quantity_alignment': [{'gold': {'path': [], 'number': '0', 'unit': '万'},
                                     'prediction': {'path': ['status'], 'number': '0', 'unit': '万元', 'surface': '0万元'}}]}
        self.assertTrue(s.quantity_approval_valid(g, p, d))
        self.assertFalse(s.quantity_approval_valid({'number': '1', 'unit': '万'}, p, d))
        self.assertFalse(s.quantity_approval_valid(g, {'status': '实际已放款1万元'}, d))

    def test_quantity_alignment_cannot_omit_second_amount(self):
        g = {'amount': {'number': '0', 'unit': '万'}, 'fee': {'number': '1', 'unit': '万'}}
        p = {'status': '实际已放款0万元'}
        d = {'quantity_equivalence_verified': True, 'quantity_equivalence_reason': '测试',
             'quantity_alignment': [{'gold': {'path': ['amount'], 'number': '0', 'unit': '万'},
                                     'prediction': {'path': ['status'], 'number': '0', 'unit': '万元', 'surface': '0万元'}}]}
        self.assertFalse(s.quantity_approval_valid(g, p, d))

    def test_zero_surface_certificate_is_specific_and_cannot_hide_positive_amount(self):
        d = {'quantity_equivalence_verified': True, 'quantity_equivalence_reason': '项目累计未放款为零',
             'quantity_alignment': [{'gold': {'path': [], 'number': '0', 'unit': '万'},
                                     'prediction': {'path': ['status'], 'number': '0', 'unit': '万', 'surface': '未放款',
                                                    'zero_surface_kind': 'no_disbursement', 'surface_equivalence_reason': '仅此累计未放款状态'}}]}
        self.assertTrue(s.quantity_approval_valid({'number': '0', 'unit': '万'}, {'status': '未放款'}, d))
        self.assertFalse(s.quantity_approval_valid({'number': '1', 'unit': '万'}, {'status': '未放款'}, d))
        self.assertFalse(s.quantity_approval_valid({'number': '0', 'unit': '万'}, {'status': '今天未放款'}, d))


if __name__ == '__main__':
    unittest.main()
