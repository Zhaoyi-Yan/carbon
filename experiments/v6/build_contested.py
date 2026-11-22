"""Combine the learned action heads, calibrated rival replies and common-sample downside scoring."""
import argparse
from pathlib import Path
import re
import tempfile
from build_guard import build as add_guard
from learn_export import export
from opponent import source_for

HERE = Path(__file__).resolve().parent

def build(output):
    base=HERE.parents[1]/'yan-v5.py'
    source=source_for(base.read_bytes())
    with tempfile.TemporaryDirectory() as temp:
        fitted=Path(temp)/'learned.py'
        export(base,HERE/'learned-heads.npz',fitted)
        weights=re.search(r'^_WEIGHTS = "[^"]*"$',fitted.read_text(),re.M)[0]
    source,n=re.subn(r'^_WEIGHTS = "[^"]*"$',lambda _:weights,source,flags=re.M)
    assert n==1
    source=add_guard(source)
    def change(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,old
        source=source.replace(old,new)
    change('_S_MMAX = 2 ', '_S_MMAX = 4 ')
    change('(0.27 if replies == 2 else 0)','(0.42 if replies == 2 else 0)')
    change('    alive = list(range(len(cands)))', '''    cash_gap = players[me][0] - players[op][0]
    # With common replies, assess the downside of deviating from the policy's own move.
    # A shared swing in both candidates' absolute cash estimates cancels out.
    caution = float(np.clip(.30 + cash_gap / 5500., .12, .52))
    def paired_value(candidate, count):
        if candidate == 0:
            return 0.
        gains = [val[candidate,m] - val[0,m] for m in range(count)]
        mean_gain = sum(gains)/count
        return mean_gain + caution * (min(gains) - mean_gain)

    alive = list(range(len(cands)))''')
    change('sum(val[c, m] for m in range(M)) / M for c in alive','paired_value(c, M) for c in alive')
    change('sum(val[c, m] for m in range(rolls)) / rolls for c in alive','paired_value(c, rolls) for c in alive',2)
    compile(source,str(output),'exec')
    output.write_text(source)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    build(p.parse_args().output)
