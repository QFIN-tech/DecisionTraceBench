"""Structural constraints for trajectory cycles and settlement events."""
import fullvalue_score as official
from time_scoring import time_match
def structural_violations(t):
    failures = []
    for ci, c in enumerate(t['cycles']):
        for ei, e in enumerate(c['events']):
            if e['action'] == 'settle' and ei != len(c['events'])-1:
                failures.append(dict(cycle=ci,event=ei,rule='settle_must_end_cycle'))
            if e['action'] != 'settle' and e.get('result') is not None:
                failures.append(dict(cycle=ci,event=ei,rule='non_settle_result'))
    return failures
