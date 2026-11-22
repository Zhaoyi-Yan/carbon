"""Build from the exact published v5 file and keep both the v5 and new search proposals."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def build(output, projection):
    baseline = HERE.parents[1] / 'yan-v5.py'
    data = baseline.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '0a413a11f80f6872a5c8f0d313061d7bf11121ae16fbdad97247ff3cb15242e5'
    source = data.decode()
    def change(old,new):
        nonlocal source
        assert source.count(old) == 1, old[:80]
        source = source.replace(old,new)
    change('def _act(obs,', (HERE/'initiative.py').read_text()+'\n\ndef _act(obs,')
    change('    workers, _, bases, cash = parse(players)\n    act = {}', '''    if _SEARCH_OK and configuration is not None:
        original = (lw, lb)
        guided, _ = _v6_initiative(obs, est, configuration, lw, lb)
        _ST['v5_history']['alternatives'] = [(guided, lb), original]
        lw = lw + .50 * (guided - lw)
    workers, _, bases, cash = parse(players)
    act = {}''')
    if projection:
        extra = (HERE/'cash_projection.py').read_text()
        weights=json.loads(projection.read_text())['weights']
        change('def _v6_initiative(',extra+'\n_V6_INCOME = np.array('+repr(weights)+')\n\ndef _v6_initiative(')
        change('                    value += _V5_VALUE_MIX * (_v5_value(future, me) - value)',
               '''                    territorial = _v5_value(future, me)
                    forecast = _v6_projection(future, me)
                    value += _V5_VALUE_MIX * (.52*territorial + .48*forecast - value)''')
    compile(source,str(output),'exec')
    output.write_text(source)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--projection',type=Path)
    a=p.parse_args()
    build(a.output,a.projection)
