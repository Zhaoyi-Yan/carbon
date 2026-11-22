"""Measure sustained liquidity after handing the real Master7sE position to two reactive agents."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
for variable in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np
from branch_breakout import PACKED
from branch_replay import ROOT,OUT,HERE,board_digest,step
from benchmark import configure_engine,limit_math_threads,load_strategy,observe,validate_actions
from close_replay import restore,archive_state,referee_events
from corridor_adversary import blockade


def policy(p):
    src=p.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    i=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));j=src.index('    stages =',i)
    src=src[:i]+'    most,D2,M=4,min(2,D2),1\n'+src[j:]
    src=src.replace('for tick in range(1,15):','for tick in range(1,4):')
    m=load_strategy('pressure_'+p.stem,p,src);m._time=SimpleNamespace(perf_counter=lambda:0.);m._S_MMAX=1
    return m


def set_exec(m,obs,action):
    m._ST['tr'].set_actions(obs['players'],action)
    if 'liquidity_action' in m._ST:m._ST['liquidity_action']=dict(action)


def game(m,rival,geometry,replay,start,snapshot,mem,rival_mem,raiding):
    from zerosum_env.envs.carbon import idgen
    cfg={k:v for k,v in replay['configuration'].items() if k!='randomSeed'}
    state,env=restore(snapshot,replay['configuration'])
    m._ST,m._FC_HISTORY=copy.deepcopy(mem);rival._ST=copy.deepcopy(rival_mem)
    amounts=[Counter(),Counter()];marks=[];longest=below=0;first=None;trail=[]
    for t in range(start,299):
        obs=[observe(state,s,t) for s in (0,1)]
        saved=dict(idgen._id_dict);commands=[]
        for s,mod in enumerate((rival,m)):
            mod._ST['rng']=np.random.RandomState(73613+1000003*t+s)
            a=mod.agent(obs[s],cfg)
            if s==0 and raiding and t-start<45:
                targets=obs[0]['players'][1]
                if targets[0]<200 and sum(x[1] for x in targets[2].values())>=65:
                    a=blockade(geometry,obs[0],a)
            validate_actions(a,obs[s]['players'][s]);set_exec(mod,obs[s],a);commands.append(a)
            idgen._id_dict.clear();idgen._id_dict.update(saved)
        ev=[]
        with referee_events(ev):step(state,env,commands)
        pl=state[0].observation.players
        for kind,s,amount,*_ in ev:
            if kind in ('bank','base_collision'):
                amounts[s]['bank']+=amount
                if s==1 and amount>0 and first is None:first=t+1
            if kind=='tree_yield':amounts[s]['tree_yield']+=amount
        below=below+1 if pl[1][0]<30 else 0;longest=max(longest,below)
        trail.append(commands)
        ended=t==298 or any(x.status!='ACTIVE' for x in state)
        if t+1-start in (6,12,25,50,100) or ended:
            marks.append(dict(step=t+1,cash=[p[0] for p in pl],workers=[len(p[2]) for p in pl],trees=[len(p[3]) for p in pl],
                delivered=[round(a['bank'],2) for a in amounts],tree_income=[round(a['tree_yield'],2) for a in amounts]))
        if ended:break
    return dict(start=start,opponent='45-turn-depot-pressure' if raiding else 'public-master-root',first_deposit=first,
                longest_cash_below_recruitment=longest,marks=marks),trail


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();limit_math_threads();configure_engine()
    os.environ['CARBON_DEBUG']='1'
    replay=json.loads(gzip.decompress(PACKED.read_bytes()))['loss'];starts=(48,164)
    cfg={k:v for k,v in replay['configuration'].items() if k!='randomSeed'}
    m=policy(a.candidate);parent=policy(ROOT/'yan-v6.py')
    enemyfile=ROOT/'opponents/claude-AlphaCarbon-Master3s2.py';enemy=load_strategy('master_fork',enemyfile,enemyfile.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)'))
    enemy._search=lambda o,c,e,act,*args:act
    enemy._torch.set_num_threads(1)
    state,env=restore(replay['initial'],replay['configuration']);cases=[];logs=[]
    for mod in (m,parent):
        mod._ps,mod._pc=mod._search,mod._reply_choice;mod._search=mod._reply_choice=lambda o,c,e,act,*args:act
    saved={};ownmods={'candidate':m,'v6':parent}
    for t in range(max(starts)+1):
        if board_digest(state)!=replay['expected'][t]:raise ValueError(t)
        if t in starts:saved[t]=(archive_state(state),copy.deepcopy(enemy._ST),{n:copy.deepcopy((mod._ST,mod._FC_HISTORY)) for n,mod in ownmods.items()})
        if t==max(starts):break
        for s,mod in ((0,enemy),(1,m),(1,parent)):
            o=observe(state,s,t);mod.agent(o,cfg);set_exec(mod,o,replay['actions'][t][s])
            mod._ST.pop('corridor_track',None);mod._ST.pop('v7_next_depot',None)
        step(state,env,replay['actions'][t])
    for mod in (m,parent):mod._search,mod._reply_choice=mod._ps,mod._pc
    for t in starts:
        snap,em,own=saved[t]
        for raiding in (False,True):
            for label,mod in ownmods.items():
                case,actions=game(mod,enemy,m,replay,t,snap,own[label],em,raiding)
                cases.append(dict(policy=label,**case));logs.append(dict(policy=label,start=t,raid=raiding,actions=actions))
                print(label,t,raiding,case['first_deposit'],case['longest_cash_below_recruitment'],case['marks'][-1],flush=True)
    data=dict(candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),opponent_sha256=hashlib.sha256(enemyfile.read_bytes()).hexdigest(),
        design='Historical Master7sE map and exact positions. Both own policies select live moves with the same four joint candidates, two search plies, one response and three extra tail plies. Rival is the responsive public Master3s2 root, optionally prioritizing depot blockade for up to 45 turns when we are poor and hold cargo. One shared per-step random stream, all continuations run to termination.',cases=cases)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    a.output.with_suffix('.json.gz').write_bytes(gzip.compress(json.dumps(logs,separators=(',',':')).encode(),mtime=0))
if __name__=='__main__':main()
