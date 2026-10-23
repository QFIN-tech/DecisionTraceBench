"""Exact amount equivalence for DCIST; no currency exchange or time conversion.

Only structured {number: <decimal string>, unit: <known money unit>} values
are converted. Other keys, JSON types, text, array order and quantities are
preserved. The units 万元 and 万 denote the same CNY amount scale.
"""
from decimal import Decimal, localcontext
import re

VERSION='money_equivalence_v1'
# Scale is a power of ten in the same currency, never an FX rate.
MONEY_UNITS={}
for currency,base,aliases in [
    ('CNY','元',{'元':0,'人民币元':0,'元人民币':0,'万':4,'万元':4,'人民币万元':4,'万元人民币':4,
                '千元':3,'百万元':6,'亿':8,'亿元':8,'人民币亿元':8,'亿元人民币':8}),
    ('USD','美元',{'美元':0,'万美元':4,'百万美元':6,'亿美元':8}),
    ('HKD','港元',{'港元':0,'万港元':4,'百万港元':6,'亿港元':8}),
    ('EUR','欧元',{'欧元':0,'万欧元':4,'百万欧元':6,'亿欧元':8}),
    ('GBP','英镑',{'英镑':0,'万英镑':4,'百万英镑':6,'亿英镑':8}),
]:
    for unit,power in aliases.items():MONEY_UNITS[unit]=(currency,base,power)

DECIMAL=re.compile(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\Z')

def canonical_amount(number,unit):
    """Return exact decimal string + base currency unit, or no conversion."""
    if not isinstance(number,str) or not isinstance(unit,str) or unit not in MONEY_UNITS:return None
    value=number.strip()
    if len(value)>1000 or not DECIMAL.fullmatch(value):return None
    value=value.replace(',','')
    _,base,power=MONEY_UNITS[unit]
    with localcontext() as ctx:
        ctx.prec=max(28,len(value)+power+8)
        amount=Decimal(value)*(Decimal(10)**power)
        rendered=format(amount,'f')
    if '.' in rendered:rendered=rendered.rstrip('0').rstrip('.')
    if Decimal(rendered)==0:rendered='0'
    return rendered,base

def normalize(value):
    """Normalize structured same-currency amounts while preserving other values."""
    if isinstance(value,dict):
        out={k:('万' if k=='unit' and v=='万元' else normalize(v)) for k,v in value.items()}
        converted=canonical_amount(value.get('number'),value.get('unit'))
        if converted is not None:out['number'],out['unit']=converted
        return out
    if isinstance(value,list):return [normalize(v) for v in value]
    return value
