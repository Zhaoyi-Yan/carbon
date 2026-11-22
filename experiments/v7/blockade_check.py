"""Short paired active branches on two public QQ-v6 versus Master7sE maps, without future actions."""
from collections import Counter
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'

from branch_breakout import MATCHES,load_replay,choose
from branch_replay import ROOT,OUT,HERE,step,board_digest
from branch_run import fixed_release
from benchmark import configure_engine,limit_math_threads,observe,validate_actions
from close_replay import archive_state,restore,referee_events
from surrogate_variation import make
import blockade

# Chosen before any counterfactual results, including a clean QQ win-state negative control.
CASES={'loss':[(0,48),(0,90),(0,164),(0,225)],'win':[(1,38),(1,164)]}
HORIZON=26
FOES=('artemis','apollo-varied')


def policy():
    import torch
    torch.set_num_threads(1)
    m=fixed_release()
    from surrogate import _engine
    _engine()
    torch.set_num_threads(1)
    assert torch.get_num_threads()==1
    exec(inspect.getsource(blockade._v7_blockade),m.__dict__)
    def keep(obs,cfg,est,action,*args):
        m.home_trial_est=est
        return action
    m._search=keep
    m._reply_choice=lambda obs,cfg,est,act,*rest:act
    m.home_trial_est=None
    return m


def prepare(m,data,side,starts):
    config=data['configuration'];cfg={k:v for k,v in config.items() if k!='randomSeed'}
    state,env=restore(data['initial'],config);m._ST={};m._FC_HISTORY={}
    foes={p:make(p,config['randomSeed'],cfg) for p in FOES}
    snaps={}
    for t in range(max(starts)+1):
        if board_digest(state)!=data['expected'][t]:raise ValueError(('source mismatch',t))
        if t in starts:
            snaps[t]=[archive_state(state),copy.deepcopy((m._ST,m._FC_HISTORY,foes))]
        if t==max(starts):break
        views=[observe(state,i,t) for i in (0,1)]
        prefix=data['actions'][t]
        for p in foes.values():p.observe(views[1-side],prefix[1-side])
        choose(m,views[side],cfg,side,t)
        m._ST['tr'].set_actions(views[side]['players'],prefix[side])
        step(state,env,prefix)
    return snaps


def run(m,snap,cfg,side,t0,proxy,active):
    from zerosum_env.envs.carbon import idgen
    st,memory=snap;m._ST,m._FC_HISTORY,foes=copy.deepcopy(memory)
    state,env=restore(st,cfg);other=1-side;opp=foes[proxy]
    rules={k:v for k,v in cfg.items() if k!='randomSeed'}
    initial=[p[0] for p in state[0].observation.players]
    events=[Counter(),Counter()]
    changes=[];pressure_turns=0;ms=[]
    for t in range(t0,t0+HORIZON):
        us=observe(state,side,t);them=observe(state,other,t)
        idstate=dict(idgen._id_dict)
        other_moves=opp.act(them)
        base=choose(m,us,rules,side,t)
        start=time.perf_counter();alt=m._v7_blockade(us,rules,m.home_trial_est,base);ms.append((time.perf_counter()-start)*1000)
        if alt!=base:
            changed=[u for u in sorted(base.keys()|alt.keys()) if base.get(u)!=alt.get(u)]
            assert len(changed)==1
            u=changed[0];pl=us['players'];pos,load,role=pl[side][2][u]
            end=m._V5_REACH[pos,m.WIDX.get(alt.get(u),0)]
            assert role=='COLLECTOR' and end not in pl[other][1].values()
            assert all(end!=m._V5_REACH[pos2,m.WIDX.get(alt.get(u2),0)] for u2,(pos2,_,_) in pl[side][2].items() if u2!=u)
            changes.append([t,u,base.get(u),alt.get(u),round(pl[side][0],1),round(pl[other][0],1)])
        mine=alt if active else base
        m._ST['tr'].set_actions(us['players'],mine)
        commands=[None,None];commands[side]=mine;commands[other]=other_moves
        idgen._id_dict.clear();idgen._id_dict.update(idstate)
        for a,p in zip(commands,us['players']):validate_actions(a,p)
        ev=[]
        with referee_events(ev):step(state,env,commands)
        for typ,s,v,*info in ev:
            if typ in ('bank','base_collision','harvest','tree_yield','loot'):
                events[s][typ]+=v
            if typ=='death':events[s]['deaths']+=1
        pressure_turns+=int(state[0].observation.players[other][0]<30)
        if any(s.status=='DONE' for s in state):break
    end=state[0].observation.players
    return dict(turns=t+1-t0,opponent_below_hire=pressure_turns,starting_cash=initial,ending_cash=[p[0] for p in end],
                ending_cargo=[round(sum(w[1] for w in p[2].values()),2) for p in end],events=[{k:round(v,3) for k,v in c.items()} for c in events],
                margin=round(end[side][0]-end[other][0],2),own_bank=round(events[side]['bank']+events[side]['base_collision'],2),
                opponent_bank=round(events[other]['bank']+events[other]['base_collision'],2),opportunities=changes,extra_ms_total=round(sum(ms),3))


def main():
    limit_math_threads();configure_engine()
    candidates=blockade.build((ROOT/'yan-v6.py').read_text())
    compile(candidates,'<v6-blockade>','exec')
    cases=[];m=policy()
    for name,selection in CASES.items():
        # No post-branch commands or future official boards are passed into run().
        data=load_replay(Path('/__use_packed_map_only__'),name);cfg=data['configuration']
        for side in sorted(set(s for s,_ in selection)):
            starts=[s for p,s in selection if p==side]
            snaps=prepare(m,data,side,starts)
            for t in starts:
                for rival in FOES:
                    ctl=run(m,snaps[t],cfg,side,t,rival,False);alt=run(m,snaps[t],cfg,side,t,rival,True)
                    diff={k:round(alt[k]-ctl[k],2) for k in ['margin','own_bank','opponent_bank','opponent_below_hire']}
                    row=dict(match=MATCHES[name][0],official_result_seed=MATCHES[name][1],referee_seed=cfg['randomSeed'],
                             attacker_seat=side,start=t,proxy=rival,control=ctl,guard=alt,delta=diff)
                    cases.append(row);print(name,t,side,rival,len(alt['opportunities']),diff,flush=True)
    report=dict(candidate_sha256=hashlib.sha256(candidates.encode()).hexdigest(),
                seed=22063,source_cases=CASES,horizon=HORIZON,foes=FOES,
                methodology='Both actors react every turn. Real actions are used solely for the strict pre-fork history. Both arms use the frozen v6 root, without lookahead search or reply reranking, for the attacker. The victim is a learned reactive approximation; neither actor is represented as real Master7sE. Common seeds by turn and seat.',
                cases=cases)
    (OUT/'blockade.json').write_text(json.dumps(report,indent=1)+'\n')

if __name__=='__main__':main()
