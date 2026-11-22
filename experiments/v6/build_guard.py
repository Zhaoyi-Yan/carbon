"""Add the last-worker policy to a frozen self-play candidate."""
import argparse
from pathlib import Path

def build(source):
    marker='def _act(obs,'
    hook="    st['tr'].set_actions(players, act)"
    assert source.count(marker)==1 and source.count(hook)==1
    code=Path(__file__).with_name('last_worker.py').read_text()
    return source.replace(marker,code+'\n\n'+marker).replace(hook,
        "    if configuration is not None:\n        act = _v6_last_worker(obs, configuration, act, lw)\n"+hook)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args();source=build(a.baseline.read_text());compile(source,str(a.output),'exec');a.output.write_text(source)
