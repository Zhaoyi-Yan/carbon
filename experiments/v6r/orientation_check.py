"""Check legal public reflections against the embedded reference transition."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,limit_math_threads,load_strategy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('replays',type=Path)
    a=p.parse_args();configure_engine();limit_math_threads();m=load_strategy('check_orientation',a.candidate)
    checked=0
    for path in sorted(a.replays.glob('*.gz'))[::37]:
        g=json.loads(gzip.decompress(path.read_bytes()))
        for t in (0,19,70,144,267):
            if t+1>=len(g['steps']):continue
            obs=dict(g['steps'][t][0]['observation'],carbon=g['steps'][t][0]['observation']['full_carbon'],player=0,step=t)
            action=[s['action'] or {} for s in g['steps'][t+1]]
            lhs=m._turn(m._r6_reflect_board(obs),g['configuration'],[m._r6_reflect_actions(v) for v in action])
            rhs=m._turn(obs,g['configuration'],action)
            for before,reflected in zip(rhs['players'],lhs['players']):
                assert abs(before[0]-reflected[0])<.0001
                assert {k:m._R6_REFLECTION[v] for k,v in before[1].items()}==reflected[1]
                for slot in (2,3):
                    assert before[slot].keys()==reflected[slot].keys()
                    for uid,v in before[slot].items():
                        other=reflected[slot][uid]
                        assert m._R6_REFLECTION[v[0]]==other[0] and abs(v[1]-other[1])<1.e-6 and v[2:]==other[2:]
            assert max(abs(x-rhs['carbon'][q]) for x,q in zip(lhs['carbon'],m._R6_REFLECTION))<.002
            checked+=1
    print('paired real-board transitions',checked)

if __name__=='__main__':main()
