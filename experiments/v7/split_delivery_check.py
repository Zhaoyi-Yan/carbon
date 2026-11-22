"""Test long convoy returns in the actual Master position under recorded and reactive depot guards."""
import argparse,copy,gzip,json,os
from collections import Counter
from pathlib import Path
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import numpy as np
from branch_replay import ROOT,OUT,board_digest,step
from branch_breakout import PACKED
from benchmark import load_strategy,configure_engine,observe,validate_actions
from close_replay import archive_state,restore,referee_events
from corridor_adversary import blockade
from split_delivery import build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);a=p.parse_args()
    configure_engine();ds=json.loads(gzip.decompress(PACKED.read_bytes()));cases=[]
    m=load_strategy('split_gate',a.parent,build(a.parent.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')));m._torch.set_num_threads(1)
    for name,starts in (('loss',(90,132,164,193)),('win',(48,90,164))):
        g=ds[name];st,env=restore(g['initial'],g['configuration']);cfg={k:v for k,v in g['configuration'].items() if k!='randomSeed'}
        hist={};prev=None;cmd={};tr=m.BeliefTracker(m._Bel,1);snaps={}
        for t in range(max(starts)+1):
            if board_digest(st)!=g['expected'][t]:raise ValueError(t)
            o=observe(st,1,t);pl=o['players']
            if t in starts:snaps[t]=(archive_state(st),copy.deepcopy(tr),dict(hist),prev,cmd)
            e,k=tr.step(pl,t,o['carbon']);m._sense_tree_carbon(tr.bel,o,cfg,prev,e,k);m._v7_watch_cash(o,prev,cmd,hist)
            if t==max(starts):break
            cmd=g['actions'][t][1];tr.set_actions(pl,cmd);prev=pl;step(st,env,g['actions'][t])
        for t0 in starts:
            for raid in (False,True):
                for treatment in (False,True):
                    snap,tr,hh,prev,cmd=copy.deepcopy(snaps[t0]);st,env=restore(snap,g['configuration']);m._ST={};m._FC_HISTORY={}
                    marks=[];trace=[];flows=[Counter(),Counter()];first=None
                    for t in range(t0,t0+35):
                        o=observe(st,1,t);pl=o['players'];e,k=tr.step(pl,t,o['carbon'])
                        e,k=m._sense_tree_carbon(tr.bel,o,cfg,prev,e,k);x=m._quant(m.encode(pl,t,1,e,k,prev))
                        m._v5_root(o,e,x,prev)
                        drought=m._v7_watch_cash(o,prev,cmd,hh)
                        acts=[{uid:act for uid,act in g['actions'][t][s].items() if uid in pl[s][1] or uid in pl[s][2]} for s in (0,1)]
                        if raid:acts[0]=blockade(m,observe(st,0,t),acts[0])
                        ours=acts[1];acts[1]=m._v7_short_deposit(o,cfg,ours,drought);acts[1],short=m._v7_corridor(o,cfg,acts[1],drought,m._ST)
                        plans=None
                        if treatment and short is None and not m._ST.get('corridor_track') and not m._ST.get('v7_next_depot'):
                            acts[1],plans=m._v7_split_delivery(o,cfg,acts[1],drought,m._ST['v5_history']['opponent'][0],prev)
                        trace.append(dict(t=t,proposed=plans,short=short,actions=acts))
                        for s in (0,1):validate_actions(acts[s],pl[s])
                        tr.set_actions(pl,acts[1]);prev,cmd=pl,acts[1];events=[]
                        with referee_events(events):step(st,env,acts)
                        for tag,s,v,*_ in events:
                            if tag in ('bank','base_collision'):
                                flows[s]['gross']+=v
                                if s==1 and v>0 and first is None:first=t+1
                            if tag=='tree_yield':flows[s]['trees']+=v
                        if t+1-t0 in (10,20,35):
                            post=st[0].observation.players
                            marks.append(dict(step=t+1,cash=[z[0] for z in post],deposits=[round(v['gross'],2) for v in flows],trees=[len(z[3]) for z in post]))
                    cases.append(dict(game=name,start=t0,raid=raid,alternative=treatment,first=first,marks=marks,changes=sum(r['proposed'] is not None for r in trace),actions=trace))
                    if treatment:print(name,t0,raid,cases[-2]['marks'][-1],marks[-1],flush=True)
    (OUT/'split-delivery.json.gz').write_bytes(gzip.compress(json.dumps(cases,separators=(',',':')).encode(),mtime=0))
    (OUT/'split-delivery.json').write_text(json.dumps([{k:v for k,v in d.items() if k!='actions'} for d in cases],indent=2)+'\n')
if __name__=='__main__':main()
