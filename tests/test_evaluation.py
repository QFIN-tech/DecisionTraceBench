import copy
import json
from pathlib import Path
import unittest

from decisiontracebench import evaluation as cli
from test_matching import event, track

ROOT = Path(__file__).resolve().parents[1]


def simple_case():
    g = {'tracks': [track()]}
    return {'inference_status': 'completed', 'time_enabled': True,
            'gold': g, 'prediction': copy.deepcopy(g)}


def node_review(ei, label='invalid_extra', pi=0):
    return dict(prediction_track_index=pi, cycle_index=0, event_index=ei,
                label=label, reason='Constructed test.', rule_quote='Test rule.', evidence_quote='Test evidence.')


def value_review(c, label='equivalent'):
    return dict(candidate_id='v1', dimension='value', gold_track_index=0,
                prediction_track_index=0, cycle_index=0, gold_event_index=0,
                prediction_event_index=0, gold=c['gold']['tracks'][0]['cycles'][0]['events'][0]['value'],
                prediction=c['prediction']['tracks'][0]['cycles'][0]['events'][0]['value'],
                verdict=label, reason='Fixture equivalence.', evidence_quote='Test evidence.')


class AdapterTests(unittest.TestCase):
    def test_worked_example_and_input_immutability(self):
        case = json.loads((ROOT/'assets/examples/case.json').read_text(encoding='utf-8'))
        before = copy.deepcopy(case)
        result = cli.evaluate(case)
        expected = json.loads((ROOT/'assets/examples/expected.json').read_text(encoding='utf-8'))
        for k, v in expected.items(): self.assertEqual(result[k], v, k)
        self.assertEqual(case, before)

    def test_exact(self):
        self.assertEqual(cli.evaluate(simple_case())['dcist'], 1)

    def test_money_equivalence(self):
        c = simple_case()
        c['gold']['tracks'][0]['cycles'][0]['events'][0]['value'] = {'number':'5','unit':'万'}
        c['prediction']['tracks'][0]['cycles'][0]['events'][0]['value'] = {'unit':'元','number':'50000'}
        self.assertEqual(cli.evaluate(c)['dcist'], 1)

    def test_wrong_amount_cannot_be_mapped(self):
        c = simple_case()
        c['gold']['tracks'][0]['cycles'][0]['events'][0]['value'] = {'number':'5','unit':'万'}
        c['prediction']['tracks'][0]['cycles'][0]['events'][0]['value'] = {'number':'6','unit':'万'}
        c['semantic_decisions'] = [value_review(c)]
        with self.assertRaisesRegex(ValueError, 'quantity'): cli.evaluate(c)

    def test_unreviewed_is_not_uncertain(self):
        c = simple_case()
        c['prediction']['tracks'][0]['cycles'][0]['events'].append(event('E2',action='advance',source='E1'))
        self.assertEqual(cli.evaluate(c)['status'], 'review_required')
        self.assertIsNone(cli.evaluate(c)['dcist'])
        c['node_decisions'] = [node_review(1, 'evidence_uncertain')]
        self.assertEqual(cli.evaluate(c)['dcist'], 1)
        self.assertEqual(cli.evaluate(c)['extra_node_ratio'], .5)

    def test_known_wrong_extra_blocks(self):
        c = simple_case()
        c['prediction']['tracks'][0]['cycles'][0]['events'].append(event('E2',action='advance',source='E1'))
        c['node_decisions'] = [node_review(1)]
        self.assertEqual(cli.evaluate(c)['dcist'], 0)

    def test_missing_anchor_cannot_be_exempted(self):
        c = simple_case()
        c['gold']['tracks'][0]['cycles'][0]['events'].append(event('E2',action='advance',source='E1'))
        self.assertEqual(cli.evaluate(c)['dcist'], 0)

    def test_extra_track_needs_every_node_reviewed(self):
        c = simple_case()
        c['prediction']['tracks'].append(track(field='new',evs=[event('E2')]))
        c['track_decisions'] = [dict(prediction_track_index=1,label='exempt_new_track',reason='Test.')]
        with self.assertRaisesRegex(ValueError,'every node'): cli.evaluate(c)
        c['node_decisions'] = [node_review(0,'exempt_extra',pi=1)]
        self.assertEqual(cli.evaluate(c)['dcist'],1)
        c['node_decisions'][0]['label']='invalid_extra'
        with self.assertRaisesRegex(ValueError,'every node'): cli.evaluate(c)

    def test_source_link_is_not_repaired(self):
        c = simple_case();c['prediction']['tracks'][0]['cycles'][0]['events'][0]['source_event_id']='bad'
        self.assertEqual(cli.evaluate(c)['status'],'completed_format_error')
        self.assertEqual(cli.evaluate(c)['dcist'],0)

    def test_time_threshold(self):
        c = simple_case();c['prediction']['tracks'][0]['cycles'][0]['events'][0]['end_ms']=1250
        self.assertEqual(cli.evaluate(c)['dcist'],1)
        c['prediction']['tracks'][0]['cycles'][0]['events'][0]['end_ms']=1251
        c['node_decisions']=[node_review(0,'replacement_error')]
        self.assertEqual(cli.evaluate(c)['dcist'],0)

    def test_no_time_only_when_explicit(self):
        c = simple_case();c['time_enabled']=False
        c['prediction']['tracks'][0]['cycles'][0]['events'][0].pop('start_ms')
        c['prediction']['tracks'][0]['cycles'][0]['events'][0].pop('end_ms')
        self.assertEqual(cli.evaluate(c)['dcist'],1)

    def test_bad_gold_interval_fails(self):
        c = simple_case();c['gold']['tracks'][0]['cycles'][0]['events'][0]['end_ms']=0
        with self.assertRaises(ValueError): cli.evaluate(c)

    def test_completed_malformed_answer_is_zero(self):
        c = simple_case();c['prediction']='not json'
        self.assertEqual(cli.evaluate(c)['dcist'],0)

    def test_failure_is_not_model_zero(self):
        for state in ['failed','truncated','pending']:
            c = simple_case();c['inference_status']=state
            with self.assertRaisesRegex(ValueError,'not a scored zero'): cli.evaluate(c)

    def test_mapping_must_bind_actual_prediction(self):
        c = simple_case();c['semantic_decisions']=[value_review(c)]
        c['semantic_decisions'][0]['prediction']={'status':'not the actual output'}
        with self.assertRaisesRegex(ValueError,'does not bind'): cli.evaluate(c)

    def test_scope_and_field_mapping(self):
        c = simple_case();pt=c['prediction']['tracks'][0]
        pt['entity_id']='alias';pt['cycles'][0]['events'][0]['scope_id']='scope_alias'
        c['semantic_decisions']=[dict(candidate_id='key',dimension='track_key',gold_track_index=0,
            prediction_track_index=0,gold=['entity','field'],prediction=['alias','field'],
            verdict='equivalent',reason='Test.',evidence_quote='Test.'),
            dict(candidate_id='scope',dimension='scope',gold_track_index=0,prediction_track_index=0,
            cycle_index=0,gold_event_index=0,prediction_event_index=0,gold='scope',prediction='scope_alias',
            verdict='equivalent',reason='Test.',evidence_quote='Test.')]
        self.assertEqual(cli.evaluate(c)['dcist'],1)

    def test_cross_field_event_mapping_needs_key_approval(self):
        c=simple_case();c['prediction']['tracks'][0]['entity_id']='alias'
        c['semantic_decisions']=[value_review(c)]
        with self.assertRaisesRegex(ValueError,'track_key'):cli.evaluate(c)

    def test_mapping_does_not_relax_speaker(self):
        c=simple_case();c['prediction']['tracks'][0]['cycles'][0]['events'][0]['speaker_id']='S2'
        c['semantic_decisions']=[value_review(c)]
        with self.assertRaisesRegex(ValueError,'protected'):cli.evaluate(c)

    def test_negative_index_rejected(self):
        c=simple_case();d=value_review(c);d['gold_event_index']=-1;c['semantic_decisions']=[d]
        with self.assertRaisesRegex(ValueError,'index'):cli.evaluate(c)

    def test_derived_state_not_in_main_score(self):
        c=simple_case();c['prediction']['tracks'][0]['derived_final_state']='different'
        self.assertEqual(cli.evaluate(c)['dcist'],1)

    def test_uncertain_semantics_not_approved(self):
        c=simple_case();c['prediction']['tracks'][0]['cycles'][0]['events'][0]['value']={'status':'alias'}
        c['semantic_decisions']=[value_review(c,'uncertain')]
        c['node_decisions']=[node_review(0,'replacement_error')]
        self.assertEqual(cli.evaluate(c)['dcist'],0)


if __name__ == '__main__':unittest.main()
