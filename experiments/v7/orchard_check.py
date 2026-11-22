"""Audit an immediate tree-guard intervention when a real rival planter takes a tree."""
import argparse
from collections import Counter
import json
from pathlib import Path
from branch_replay import ROOT,OUT,configure_engine
from benchmark import load_strategy,limit_math_threads
from starvation_replay import read_replay
from orchard_guard import _v7_cover_tree


def analyze(m,path,side):
    g=read_replay(path);fs=g['steps'];records=[];take=Counter()
    for t in range(len(fs)-1):
        o=dict(fs[t][0]['observation'],player=side);players=o['players'];us,them=players[side],players[1-side]
        after=fs[t+1][0]['observation']['players'];occupied={p for p,_,role in them[2].values() if role=='PLANTER'}
        own_by_pos={q:uid for uid,(q,age) in us[3].items() if q in occupied}
        if not own_by_pos:continue
        taken={q for q,age in after[1-side][3].values() if q in own_by_pos}
        take['occupied_turns']+=1;take['recorded_captures']+=len(taken)
        act=fs[t+1][side]['action'] or {};new=_v7_cover_tree(o,g['configuration'],act)
        if new==act:continue
        wid=next(k for k in new if new[k]!=act.get(k));p,load,_=us[2][wid]
        q=int(m._V7_L[p,m.WIDX[new[wid]]]);rate=0.; held=0
        for i in range(t+1,min(t+21,len(fs))):
            v=fs[i][0]['observation'];ids=[tid for tid,(z,_) in v['players'][1-side][3].items() if z==q]
            if not ids:break
            rate+=v['trees'][ids[0]][1];held+=1
        records.append(dict(turn=t,tree=q,captured=q in taken,enemy_income_next20=round(rate,2),enemy_ownership_turns=held,collector_cargo=load))
    return dict(file=path.name,side=side,counts=dict(take),records=records)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('cache',type=Path);a=ap.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('guard_reference',ROOT/'experiments/v7/checkpoints/jetty.py')
    _v7_cover_tree.__globals__.update({k:getattr(m,k) for k in ('_V7_D','_V7_L','LAST_TURN','WIDX','WACT')})
    data=json.loads((a.cache/'index.json').read_text());files=[(a.cache/r['file'],r['seat']) for r in data]
    files.append((a.cache.parent/'starvation-4063.html.gz',1))
    records=[analyze(m,f,s) for f,s in files]
    samples=[v for r in records for v in r['records']]
    stats=dict(games=len(records),observed_captures=sum(r['counts'].get('recorded_captures',0) for r in records),responses=len(samples),
               caught=sum(v['captured'] for v in samples),actual_opponent_tree_income=sum(v['enemy_income_next20'] for v in samples))
    (OUT/'orchard-guard.json').write_text(json.dumps(dict(summary=stats,games=records),indent=2)+'\n');print(stats)
if __name__=='__main__':main()
