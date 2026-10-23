"""Temporal matching: intersection / predicted duration >= 80%."""


def valid_interval(start_ms, end_ms):
    return type(start_ms) is int and type(end_ms) is int and 0 <= start_ms < end_ms


def time_match(pred_start_ms, pred_end_ms, gold_start_ms, gold_end_ms):
    if not valid_interval(gold_start_ms, gold_end_ms):
        raise ValueError('Invalid GT interval; repair/confirm GT rather than score the model against it.')
    if not valid_interval(pred_start_ms, pred_end_ms):
        return {'pass': False, 'coverage': None, 'intersection_ms': 0, 'reason': 'invalid_prediction_interval'}
    intersection = max(0, min(pred_end_ms, gold_end_ms) - max(pred_start_ms, gold_start_ms))
    duration = pred_end_ms - pred_start_ms
    return {'pass': 5 * intersection >= 4 * duration, 'coverage': intersection / duration,
            'intersection_ms': intersection, 'prediction_duration_ms': duration}


def aligned_track_time_match(gold_track, pred_track):
    """Invoke only for structurally aligned candidates; never search other GT events."""
    if len(gold_track['cycles']) != len(pred_track['cycles']):
        return False, []
    diagnostics = []
    for gc, pc in zip(gold_track['cycles'], pred_track['cycles']):
        if len(gc['events']) != len(pc['events']):
            return False, []
        for ge, pe in zip(gc['events'], pc['events']):
            result = time_match(pe.get('start_ms'), pe.get('end_ms'), ge.get('start_ms'), ge.get('end_ms'))
            diagnostics.append(result)
    return bool(diagnostics) and all(x['pass'] for x in diagnostics), diagnostics
