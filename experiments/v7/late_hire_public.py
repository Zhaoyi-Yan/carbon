"""Validate the frozen late hiring condition against unrelated official v6 replay actions."""
import argparse
import json
from statistics import mean
from pathlib import Path
from branch_replay import ROOT,OUT,HERE
from benchmark import configure_engine,limit_math_threads,load_strategy
from starvation_replay import read_replay
from late_hire import _v7_hire_features
from late_hire_check import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('late_hire_public',HERE/'checkpoints/soundings.py')
    _v7_hire_features.__globals__.update({k:getattr(m,k) for k in ('LAST_TURN','_V7_D','np')})
    rows=[]
    for r in json.loads((a.archive/'index.json').read_text()):
        g=read_replay(a.archive/r['file']);f=g['steps'];side=r['seat'];names=['rival','rival'];names[side]='yan-QQ-v6'
        h=[([z['action'] or {} for z in frame],[q[0] for q in frame[0]['observation']['players']]) for frame in f[1:]]
        init=dict(config=g['configuration'],seats=[dict(observation=z['observation'],status=z['status'],reward=z['reward'],action=z['action'] or {}) for z in f[0]])
        case=dict(actions_and_post_cash=h,seed=g['configuration']['randomSeed'],players=[{'name':k} for k in names],cash=[q[0] for q in f[-1][0]['observation']['players']])
        for rec in run(case,m,init):rows.append(dict(file=r['file'],**rec))
    chosen=[r['net_of_removing'] for r in rows if r['skip']]
    summary=dict(candidates=len(rows),screened=len(chosen),positive=sum(x>0 for x in chosen),negative=sum(x<0 for x in chosen),mean=round(mean(chosen),2) if chosen else None)
    print(summary);(OUT/'last-hire-public.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
