"""Protected quantity checks for occurrence-specific semantic approvals."""
from . import schema, money

def same(a, b):
    return schema.freeze(a) == schema.freeze(b)

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
                out.append((path, 'quantity', schema.freeze(v.get('number')), schema.freeze(v.get('unit'))))
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
            out.append((path, 'number_literal', schema.freeze(v)))
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
                out.append((path, 'quantity', schema.freeze(v.get('number')), schema.freeze(v.get('unit'))))
            for k in sorted(v):
                if k not in ('number', 'unit'):
                    walk(v[k], path + (k,))
        elif isinstance(v, list):
            for item in v:
                walk(item, path + ('[]',))
        elif type(v) in (int, float):
            out.append((path, 'number_literal', schema.freeze(v)))
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
