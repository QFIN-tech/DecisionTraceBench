"""JSON validation and typed-value comparison for trajectory scoring."""
import json, math, re
def parse_json(text):
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text,count=1)
        text=re.sub(r'\s*```$','',text,count=1)
    return json.loads(text)


def ratio(tp,ng,np):
    p=tp/np if np else (1.0 if ng==0 else 0.0)
    r=tp/ng if ng else 1.0
    return {'tp':tp,'gold':ng,'pred':np,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0}


def normalize(value):
    if isinstance(value,dict):
        return {k:('万' if k=='unit' and v=='万元' else normalize(v)) for k,v in value.items()}
    if isinstance(value,list):return [normalize(v) for v in value]
    return value


def freeze(value):
    # Preserve JSON types: bool false must not match integer 0.
    if isinstance(value,dict):return ('object',tuple((k,freeze(v)) for k,v in sorted(value.items())))
    if isinstance(value,list):return ('array',tuple(freeze(v) for v in value))
    return (type(value).__name__,value)


def json_value_ok(value):
    if value is None or type(value) in (str,bool,int):return True
    if type(value) is float:return math.isfinite(value)
    if isinstance(value,list):return all(json_value_ok(x) for x in value)
    if isinstance(value,dict):return all(isinstance(k,str) and json_value_ok(v) for k,v in value.items())
    return False


def validate(data):
    if not isinstance(data,dict) or not isinstance(data.get('tracks'),list):return ['tracks_not_list']
    errors=[];seen=set()
    for t in data['tracks']:
        if not isinstance(t,dict) or not all(isinstance(t.get(k),str) for k in ('entity_id','field_id')):
            errors.append('invalid_track');continue
        if not isinstance(t.get('cycles'),list) or not t['cycles']:
            errors.append('cycles_missing');continue
        for c in t['cycles']:
            if not isinstance(c,dict) or not isinstance(c.get('events'),list) or not c['events']:
                errors.append('events_missing');continue
            previous=None
            for e in c['events']:
                if not isinstance(e,dict):errors.append('event_not_object');continue
                eid=e.get('event_id')
                if not isinstance(eid,str) or eid in seen:errors.append('invalid_or_duplicate_event_id')
                seen.add(str(eid))
                if 'source_event_id' not in e or e['source_event_id']!=previous:errors.append('non_previous_source_event_id')
                previous=eid
                if e.get('action') not in ('start','advance','hold','settle'):errors.append('invalid_action')
                if not re.fullmatch(r'S[1-9][0-9]*',str(e.get('speaker_id',''))):errors.append('invalid_speaker_id')
                if not isinstance(e.get('scope_id'),str):errors.append('scope_missing')
                if 'value' not in e or not json_value_ok(e['value']):errors.append('invalid_value')
                if e.get('action')=='settle' and e.get('result') not in ('accepted','rejected','withdrawn','revoked'):
                    errors.append('settle_result_missing')
    return errors
