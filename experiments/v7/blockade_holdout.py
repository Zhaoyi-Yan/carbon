"""Confirm the frozen guard on two other Master7sE bankruptcy maps, with reactive continuations."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'

from blockade_check import policy,prepare,run,FOES,HORIZON,ROOT,OUT
from branch_replay import digest,fields
from benchmark import configure_engine,limit_math_threads,new_game
from starvation_replay import distance

IDS=('f5e69abd6f15278bf9ae01a40601c330','efa15ce0a9b4d18398cbe73ad5ca6dc6')


def portable(g):
    f=g['steps'];cfg=g['configuration'];new_game(cfg['randomSeed'],len(f))
    from zerosum_env.envs.carbon import idgen
    initial=dict(seats=[dict(observation=p['observation'],status=p['status'],reward=p['reward'],action=p['action'] or {}) for p in f[0]],
                 id_counters=dict(idgen._id_dict))
    return dict(configuration=cfg,initial=initial,actions=[[s['action'] or {} for s in fr] for fr in f[1:]],
                expected=[digest(fields([s['observation'] for s in fr],[s['reward'] for s in fr])) for fr in f])


def select(g,side):
    starts=[]
    for t,fr in enumerate(g['steps'][:-1]):
        if t<18 or t>g['configuration']['episodeSteps']-HORIZON-1 or starts and t<starts[-1]+40:continue
        ob=fr[0]['observation'];own,foe=ob['players'][side],ob['players'][1-side];homes=foe[1].values()
        near=lambda p:min(distance(p,q) for q in homes)
        candidate=[w for w in own[2].values() if w[2]=='COLLECTOR' and w[1]<=13.5 and near(w[0])<=3]
        shipment=[w[1] for w in foe[2].values() if w[2]=='COLLECTOR' and w[1]>=49.5 and 2<=near(w[0])<=8]
        if own[0]>=90 and foe[0]<45 and len(own[2])>=4 and candidate and sum(shipment)>=90 and sum(ob['trees'][k][1] for k in foe[3])<10:
            starts.append(t)
        if len(starts)==2:break
    return starts


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',required=True,type=Path);a=ap.parse_args()
    limit_math_threads();configure_engine();m=policy()
    index={v['id']:v for v in json.loads((a.data/'index.json').read_text())}
    rows=[];selections=[]
    for uid in IDS:
        row=index[uid]
        g=json.load(gzip.open(a.data/row['file'],'rt'));side=next(i for i,k in enumerate(row['order']) if 'Master7sE' in row['strategies'][k]['name'])
        cfg=g['configuration'];starts=select(g,side)
        selections.append(dict(match=uid,attacker_seat=side,seed=cfg['randomSeed'],starts=starts,source_file=row['file']))
        print('held-out',selections[-1],flush=True)
        if not starts:continue
        data=portable(g);snaps=prepare(m,data,side,starts)
        for t in starts:
            for proxy in FOES:
                ctrl=run(m,snaps[t],cfg,side,t,proxy,False);alt=run(m,snaps[t],cfg,side,t,proxy,True)
                diff={k:round(alt[k]-ctrl[k],2) for k in ['margin','own_bank','opponent_bank','opponent_below_hire']}
                rows.append(dict(match=uid,seed=cfg['randomSeed'],attacker_seat=side,start=t,proxy=proxy,control=ctrl,guard=alt,delta=diff))
                print(t,proxy,len(alt['opportunities']),diff,flush=True)
    import blockade
    report=dict(guard_sha256=hashlib.sha256(Path(blockade.__file__).read_bytes()).hexdigest(),
                starts_rule='First two blockade-eligible observations in chronological order, at least 40 turns apart. Eligibility uses only current public worker positions, loads, trees and cash; it does not inspect future movement or yields.',
                conditional_sample='Two different known official v5 bankruptcy losses to Master7sE. Their subsequent event sequence is never supplied to either decision-maker. Outcomes on this selected map population do not estimate a league win rate.',
                horizon=HORIZON,seed=22063,selections=selections,cases=rows)
    (OUT/'blockade-holdout.json').write_text(json.dumps(report,indent=1)+'\n')

if __name__=='__main__':main()
