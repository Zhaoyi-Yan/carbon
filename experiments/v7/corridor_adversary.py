"""Try a live depot blockade against the early three/four-step courier intervention."""
import json
from collections import Counter
import numpy as np
from branch_replay import ROOT,OUT,step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe,validate_actions
from close_replay import restore,archive_state,referee_events
from starvation_replay import read_replay
from depot_corridor import _v7_corridor


def blockade(model,obs,recorded):
    me=obs['player'];raiders=obs['players'][me][2];defenders=obs['players'][1-me]
    c=[(u,p,carry) for u,(p,carry,k) in raiders.items() if k=='COLLECTOR']
    loads=[(p,v) for p,v,k in defenders[2].values() if k=='COLLECTOR' and v>=25]
    if not c or not loads:return recorded
    homes=list(defenders[1].values());out=dict(recorded)
    goal={}
    for p,v in loads:
        near=min(model._V7_D[p,h] for h in homes)
        for tile in range(225):
            to_depot=min(model._V7_D[tile,h] for h in homes)
            detour=model._V7_D[p,tile]+to_depot-near
            if detour>2 or to_depot==0:continue
            goal[tile]=max(goal.get(tile,0.),v*(.8**detour)*(.92**model._V7_D[p,tile]))
    if not goal:return out
    destinations=sorted({int(z) for _,p,_ in c for z in model._V7_L[p]})
    score=np.full((len(c),len(destinations)),-1e6)
    for i,(u,p,carry) in enumerate(c):
        feasible={int(z):j for j,z in enumerate(model._V7_L[p])}
        for j,z in enumerate(destinations):
            if z not in feasible:continue
            hit=max((v for q,v in loads if model._V7_D[q,z]<=1 and carry<=v),default=0.)
            cover=max((v/(1+model._V7_D[z,q]) for q,v in goal.items()),default=0.)
            score[i,j]=.25*hit+cover+.01*int(model.WIDX.get(recorded.get(u),0)==feasible[z])-.02*carry
    possible=[{z:(score[i,j],) for j,z in enumerate(destinations) if score[i,j]>-1e5} for i in range(len(c))]
    for i,z in enumerate(model.assign_moves(possible)):
        uid,p,_=c[i];a=list(model._V7_L[p]).index(z)
        if a:out[uid]=model.WACT[a]
        else:out.pop(uid,None)
    return out


def main():
    limit_math_threads();configure_engine()
    from zerosum_env.envs.carbon import idgen
    m=load_strategy('corridor_blockade',ROOT/'experiments/v7/checkpoints/soundings.py')
    _v7_corridor.__globals__.update({k:getattr(m,k) for k in ('_V7_L','_V7_D','WIDX','WACT','LAST_TURN')})
    g=read_replay(ROOT.parent/'v7-data/starvation-4063.html.gz');frames=g['steps'];cfg=g['configuration']
    new_game(cfg['randomSeed'],300);init=dict(seats=[dict(observation=b['observation'],status=b['status'],reward=b['reward'],action=b['action'] or {}) for b in frames[0]],id_counters=dict(idgen._id_dict))
    st,env=restore(init,cfg);hist={};pr={};prev=None;capture={}
    for t in range(90):
        o=observe(st,1,t)
        if t in (48,49,89):capture[t]=(archive_state(st),dict(hist),pr,prev)
        m._v7_watch_cash(o,prev,pr,hist);aa=[b['action'] or {} for b in frames[t+1]];pr,prev=aa[1],o['players'];step(st,env,aa)
    rows=[]
    for t0,(snap,oldhist,pr0,prev0) in capture.items():
        for variant in ('recorded_ours','two_turn','four_turn'):
            st,env=restore(snap,cfg);hist=dict(oldhist);m._ST={};pr,prev=pr0,prev0;ledger=Counter();changes=0;marks=[];trace=[]
            for t in range(t0,t0+12):
                obs=[observe(st,i,t) for i in (0,1)];aa=[{k:v for k,v in (f['action'] or {}).items() if k in obs[i]['players'][i][2] or k in obs[i]['players'][i][1]} for i,f in enumerate(frames[t+1])]
                raid=blockade(m,obs[0],aa[0]);changes+=sum(raid.get(k)!=aa[0].get(k) for k in raid.keys()|aa[0].keys());aa[0]=raid
                dt=m._v7_watch_cash(obs[1],prev,pr,hist)
                route=None
                if variant!='recorded_ours':aa[1]=m._v7_short_deposit(obs[1],cfg,aa[1],dt)
                if variant=='four_turn':aa[1],route=_v7_corridor(obs[1],cfg,aa[1],dt,m._ST)
                pr,prev=aa[1],obs[1]['players']
                trace.append(dict(t=t,workers={u:[p,round(v,2),aa[1].get(u,'STAY')] for u,(p,v,k) in prev[1][2].items() if k=='COLLECTOR'},path=route,opponent=aa[0],enemy_positions=list(prev[0][2].values()),cash=prev[1][0],drought=dt,pending=m._ST.get('v7_next_depot')))
                for i,a in enumerate(aa):validate_actions(a,obs[i]['players'][i])
                events=[]
                with referee_events(events):step(st,env,aa)
                for e in events:
                    if e[1]==1:ledger[e[0]]+=e[2]
                if t-t0+1 in (4,6,12):marks.append(dict(t=t+1,cash=st[0].observation.players[1][0],gross=round(ledger['bank']+ledger['base_collision'],2)))
            rows.append(dict(t=t0,variant=variant,enemy_worker_decisions_changed=changes,marks=marks,trace=trace));print(t0,variant,marks)
    (OUT/'corridor-adversary.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
