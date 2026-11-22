"""At four late states per map, compare terminal rollouts with the actual remaining game."""
import argparse
import json
from pathlib import Path
from collections import Counter
from branch_replay import ROOT,OUT,HERE,step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe
from close_replay import restore,archive_state
import settle_search


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();base=HERE/'checkpoints/pilotage.py'
    m=load_strategy('terminal_search_verify',base,settle_search.build(base.read_text()))
    m._torch.set_num_threads(1)
    results=[]
    for f in a.files:
        for g in map(json.loads,f.open()):
            if g['turns']!=299:continue
            s=next(i for i,p in enumerate(g['players']) if p['name']!='yan-v6')
            st,e=new_game(g['seed'],300);cfg=dict(e.configuration);cfg.pop('randomSeed',None)
            tr=m.BeliefTracker(m._Bel,s);m._ST={};prev=None;cases=[];frames={};h=g['actions_and_post_cash']
            for t,(actions,x) in enumerate(h):
                o=observe(st,s,t);pl=o['players'];stock,known=tr.step(pl,t,o['carbon'])
                if t>=282:
                    xx=m._quant(m.encode(pl,t,s,stock,known,prev));w,b=m._v5_root(o,stock,xx,prev)
                    frames[t]=pl
                    if t in (292,294,296,298):
                        proposal=m._v7_settle_end(o,cfg,stock,actions[s],w,b,0.,True)
                        if proposal is not None and proposal!=actions[s]:
                            cases.append((t,archive_state(st),proposal))
                tr.set_actions(pl,actions[s]);prev=pl;step(st,e,actions)
                assert x==[p[0] for p in st[0].observation.players]
            frames[299]=st[0].observation.players
            for start,snap,proposal in cases:
                st,e=restore(snap,dict(e.configuration))
                mapping=[{u:u for u in z[2]} for z in frames[start]]
                for t in range(start,299):
                    old=st[0].observation.players;real=frames[t];actual_next=frames[t+1]
                    aa=[]
                    for team in (0,1):
                        played=h[t][0][team]
                        act={u:v for u,v in played.items() if u in old[team][1]}
                        act.update({mapping[team][u]:v for u,v in played.items() if u in mapping[team] and mapping[team][u] in old[team][2]})
                        aa.append(act)
                    if t==start:aa[s]=proposal
                    step(st,e,aa);new=st[0].observation.players
                    for team in (0,1):
                        mapping[team]={u:v for u,v in mapping[team].items() if u in actual_next[team][2] and v in new[team][2]}
                        newborn={(p,k):u for u,(p,_,k) in new[team][2].items() if u not in old[team][2]}
                        for u,(p,_,kind) in actual_next[team][2].items():
                            if u not in real[team][2] and (p,kind) in newborn:mapping[team][u]=newborn[p,kind]
                cash=[q[0] for q in st[0].observation.players];before=g['cash'][s]-g['cash'][1-s];after=cash[s]-cash[1-s]
                results.append(dict(file=f.name,seed=g['seed'],seat=s,step=start,original=round(before,2),after=round(after,2),delta=round(after-before,2)))
        print(f,len(results),flush=True)
    output={}
    for f in a.files:
        r=[v for v in results if v['file']==f.name];d=[v['delta'] for v in r]
        output[f.name]=dict(changes=len(r),improved=sum(x>.01 for x in d),worse=sum(x<-.01 for x in d),mean=round(sum(d)/max(1,len(d)),2),
                            wins_gained=sum(v['original']<=0<v['after'] for v in r),wins_lost=sum(v['original']>0>=v['after'] for v in r))
    (OUT/'settle-search-check.json').write_text(json.dumps(dict(summary=output,rows=results),indent=2)+'\n');print(output)
if __name__=='__main__':main()
