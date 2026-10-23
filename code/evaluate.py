"""Single-case trajectory scoring with semantic and extra-event decisions."""
import argparse
import copy
import json
from pathlib import Path
import sys

import trial_score as core

NODE_LABELS={'exempt_extra','invalid_extra','replacement_error','evidence_uncertain'}
TRACK_LABELS={'exempt_new_track','wrong_field_or_invalid_track','evidence_uncertain'}

def require(condition,message):
    if not condition:raise ValueError(message)

def at(items,index):
    require(type(index) is int and 0<=index<len(items),'Invalid zero-based index')
    return items[index]

def event_at(document,ti,ci,ei):
    return at(at(at(document['tracks'],ti)['cycles'],ci)['events'],ei)

def positions(track):
    return {(ci,ei) for ci,c in enumerate(track['cycles']) for ei,_ in enumerate(c['events'])}

def base_sample(gold,pred,time_enabled):
    rows=[]
    for pi,pt in enumerate(pred['tracks']):
        candidates=[(gi,g) for gi,g in enumerate(gold['tracks']) if core.key(g)==core.key(pt)]
        aligned=core.align(candidates[0][1],pt,time_enabled,True,core.empty_allows(),'case',candidates[0][0],pi,'baseline_money') if candidates else {'cycles':[]}
        matched={(c['cycle_index'],ei) for c in aligned['cycles'] for _,ei in c['matched']}
        rows.append({'review_id':f'case/T{pi}','prediction_track_index':pi,
            'entity_id':pt['entity_id'],'field_id':pt['field_id'],'gold_present':bool(candidates),
            'structural_violations':core.legacy.structural_violations(pt),
            'unresolved_nodes':[dict(cycle_index=ci,event_index=ei) for ci,ei in sorted(positions(pt)-matched)]})
    return {'audit_id':'case','time_enabled':time_enabled,'row':{'sample_id':'case',
        'gold_tracks':len(gold['tracks']),'prediction_tracks':len(pred['tracks']),
        'schema_errors':[],'tracks':rows}}

def approved_mappings(case,gold,pred,time_enabled):
    allows=core.empty_allows();seen=set();rows=case.get('semantic_decisions',[])
    for item in rows:
        cid=item['candidate_id'];dim=item['dimension'];label=item['verdict']
        require(cid not in seen and isinstance(cid,str),'Duplicate/invalid candidate_id');seen.add(cid)
        require(dim in core.DIMENSIONS,'Unsupported semantic dimension')
        require(label in {'equivalent','not_equivalent','uncertain'},'Invalid semantic verdict')
        require(bool(item.get('reason')) and bool(item.get('evidence_quote')),'Semantic review needs reasons and evidence')
        gi=item['gold_track_index'];pi=item['prediction_track_index']
        gt=at(gold['tracks'],gi);pt=at(pred['tracks'],pi)
        if dim=='track_key':
            gv=list(core.key(gt));pv=list(core.key(pt));coord=('case',gi,pi)
        else:
            ci=item['cycle_index'];gei=item['gold_event_index'];pei=item['prediction_event_index']
            ge=event_at(gold,gi,ci,gei);pe=event_at(pred,pi,ci,pei)
            gv=ge['value'] if dim=='value' else ge['scope_id']
            pv=pe['value'] if dim=='value' else pe['scope_id']
            coord=('case',gi,pi,ci,gei,pei)
            require(core.hard_event_equal(ge,pe,time_enabled),'Semantic candidate violates a protected event/time gate')
        require(core.same(gv,item['gold']) and core.same(pv,item['prediction']),'Candidate does not bind to the supplied gold/prediction')
        if label=='equivalent':
            if dim=='value':
                require(gv is not None and pv is not None,'Null values cannot be filled by semantic review')
                require(core.quantity_approval_valid(gv,pv,item),'Semantic review changed a protected quantity')
            allows[dim].setdefault(coord,[]).append(cid)
    for dim in ('value','scope'):
        for coord in allows[dim]:
            _,gi,pi,*_=coord
            require(core.key(gold['tracks'][gi])==core.key(pred['tracks'][pi]) or ('case',gi,pi) in allows['track_key'],
                    'Cross-field mapping requires approved track_key correspondence')
    return allows

def extra_reviews(case,sample,pred):
    nd={};td={}
    for item in case.get('node_decisions',[]):
        pi=item['prediction_track_index'];ci=item['cycle_index'];ei=item['event_index']
        event_at(pred,pi,ci,ei);coord=(f'case/T{pi}',ci,ei)
        require(coord not in nd,'Duplicate node review')
        require(item.get('label') in NODE_LABELS,'Invalid node label')
        require(all(item.get(k) for k in ('reason','rule_quote','evidence_quote')),'Node review needs reason, rule and evidence')
        nd[coord]={**item,'review_id':coord[0]}
    for item in case.get('track_decisions',[]):
        pi=item['prediction_track_index'];pt=at(pred['tracks'],pi);rid=f'case/T{pi}'
        require(rid not in td,'Duplicate track review')
        require(item.get('label') in TRACK_LABELS and item.get('reason'),'Invalid track review')
        meta=sample['row']['tracks'][pi]
        require(not meta['gold_present'],'Whole-track exemption is restricted to gold-absent fields')
        if item['label'] in {'exempt_new_track','evidence_uncertain'}:
            require(not meta['structural_violations'],'Cannot exempt a structurally invalid track')
            labels=[nd.get((rid,ci,ei),{}).get('label') for ci,ei in positions(pt)]
            allowed={'exempt_extra'} if item['label']=='exempt_new_track' else {'exempt_extra','evidence_uncertain'}
            require(all(label in allowed for label in labels),'Exempting a whole track requires every node reviewed and no known wrong node')
        td[rid]={**item,'review_id':rid}
    return nd,td

def evaluate(case):
    require(__debug__, 'Run Python without -O so scoring assertions remain enabled')
    require(case.get('inference_status')=='completed','Incomplete, failed or truncated generation is not a scored zero')
    require(type(case.get('time_enabled')) is bool,'Explicit time_enabled is required')
    gold=case['gold'];raw=case['prediction'];timed=case['time_enabled']
    require(not core.official.validate(gold),'Invalid gold input')
    require(len({core.key(t) for t in gold['tracks']})==len(gold['tracks']),'Gold fields must be unique')
    for t in gold['tracks']:
        require(not core.legacy.structural_violations(t),'Invalid gold cycle structure')
        if timed:
            for c in t['cycles']:
                for e in c['events']:core.legacy.time_match(e.get('start_ms'),e.get('end_ms'),e.get('start_ms'),e.get('end_ms'))
    try:
        pred=core.official.parse_json(raw) if isinstance(raw,str) else raw
        errors=core.official.validate(pred)
    except (TypeError,ValueError):
        pred=None;errors=['unparseable_completed_answer']
    if errors:
        count=len(pred['tracks']) if isinstance(pred,dict) and isinstance(pred.get('tracks'),list) else 0
        score=core.official.ratio(0,len(gold['tracks']),count)
        return {'status':'completed_format_error','schema_errors':errors,'raw_score':score,
                'dcist':score['f1'],'extra_node_ratio':None,'extra_node_count':None,'node_count':None}
    sample=base_sample(gold,pred,timed)
    allows=approved_mappings(case,gold,pred,timed)
    nd,td=extra_reviews(case,sample,pred)
    raw_result=core.score_one(sample,gold,pred,{}, {},core.empty_allows(),'baseline_money')
    result=core.score_one(sample,gold,pred,nd,td,allows,'joint')
    matched={m['prediction_track_index']:m for m in result['matches']}
    missing=[];mandatory=set()
    for pi,pt in enumerate(pred['tracks']):
        if sample['row']['tracks'][pi]['structural_violations']:continue
        if pi in matched:
            aligned=matched[pi]['alignment']
        else:
            candidates=[]
            for gi,gt in enumerate(gold['tracks']):
                if core.key(gt)==core.key(pt) or ('case',gi,pi) in allows['track_key']:
                    a=core.align(gt,pt,timed,True,allows,'case',gi,pi,'joint')
                    candidates.append((sum(len(c['matched']) for c in a['cycles']),-gi,a))
            aligned=max(candidates,key=lambda x:x[:2])[2] if candidates else {'cycles':[]}
        anchors={(c['cycle_index'],ei) for c in aligned['cycles'] for _,ei in c['matched']}
        mandatory.update((pi,ci,ei) for ci,ei in anchors)
        for ci,ei in sorted(positions(pt)-anchors):
            if (f'case/T{pi}',ci,ei) not in nd:missing.append({'prediction_track_index':pi,'cycle_index':ci,'event_index':ei})
        if not sample['row']['tracks'][pi]['gold_present'] and pi not in matched and f'case/T{pi}' not in td:
            missing.append({'prediction_track_index':pi,'track_decision_required':True})
    exempt_nodes={(item['prediction_track_index'],item['cycle_index'],item['event_index']) for item in nd.values()
                  if item['label'] in {'exempt_extra','evidence_uncertain'}}-mandatory
    total=sum(len(positions(t)) for t in pred['tracks'])
    complete=not missing
    return {'status':'complete' if complete else 'review_required','raw_score':raw_result['score'],
        'reviewed_score':result['score'] if complete else None,'dcist':result['score']['f1'] if complete else None,
        'extra_node_count':len(exempt_nodes) if complete else None,'node_count':total,
        'extra_node_ratio':len(exempt_nodes)/total if complete and total else (0.0 if complete else None),
        'missing_reviews':missing,'matches':result['matches'],'blocked':result['blocked'],
        'format_gate_failed':result['format_gate_failed']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:result=evaluate(json.loads(args.case.read_text(encoding='utf-8-sig')))
    except (ValueError,KeyError,TypeError,AssertionError) as e:
        print(json.dumps({'status':'input_or_review_error','error':str(e)},ensure_ascii=False),file=sys.stderr)
        return 2
    rendered=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        require(args.output.resolve()!=args.case.resolve(),'Do not overwrite the case input')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(rendered,encoding='utf-8')
    else:print(rendered)
    return 0 if result['status'] in ('complete','completed_format_error') else 2

if __name__=='__main__':sys.exit(main())
