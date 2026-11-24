"""Replay public v8 positions and compare a one-step plant with a continued tree task."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/decision_audit'))
from paired_forks import (archive_state, board_digest, configure_engine, decision,
                          digest, history, model, observe, replay_prefix, reset, restore,
                          step, validate_actions, write)
from instrument import build
from starvation_replay import cash_flows, read_replay
from scripts.selfplay import FROZEN
from plan import TreeTask, arrivals
from select_roots import choose

REVISIONS = {
    'yan-v7': FROZEN['yan-v7'][:2],
    'yan-v8': ('v11-yan-v8-elite-refit', 'a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396'),
}

ARMS=('original','plant_once','continue_task')
PROXIES=(('yan-v7',2279),('yan-v8',7237))


def update_memory(mod, obs, natural, action):
    if natural==action:
        return
    st=mod._ST
    st['tr'].set_actions(obs['players'],action)
    st['liquidity_action']=dict(action)
    plan=st.get('v7_next_depot')
    if plan and natural.get(plan[1])!=action.get(plan[1]):
        st.pop('v7_next_depot')
    if 'corridor_track' in st:
        st['corridor_track']=[x for x in st['corridor_track'] if natural.get(x[1])==action.get(x[1])]


def metrics(ledger,seat):
    def team(c):
        return dict(base=c['base'],net_tree=c['tree']+c['collision_tree']-c['plant_cost']-c['seize_cost'],
                    hire=c['hire_cost'],plants=c['plants'],seizes=c['seizes'],deaths=c['deaths'])
    own,opp=[team(ledger[i]) for i in (seat,1-seat)]
    return dict(own=own,opponent=opp,net={k:round(own[k]-opp[k],3) for k in own})


def simulate(mods,snapshot,cfg,visible,case,arm,salt,first,memory):
    state,env=restore(snapshot,cfg)
    reset(mods,memory)
    side,turn,cell,worker=[case[k] for k in ('seat','turn','cell','unit')]
    task=TreeTask(worker,cell,turn,case['cost'],seize_cost=cfg['seizeCost']) if arm=='continue_task' else None
    ledger=[Counter(),Counter()]
    before=state[0].observation
    owners={uid:i for i,p in enumerate(before.players) for uid in p[3]}
    bank_hist={}
    target_id=None
    target_yield=[0.,0.]
    target_cost=0.
    tracked_turns=[0,0]
    amendments=[]
    trace=hashlib.sha256()
    for t in range(turn,299):
        before=state[0].observation
        actions=first if t==turn else [decision(m,state,i,t,visible,salt) for i,m in enumerate(mods)]
        if t==turn and arm!='original':
            obs=observe(state,side,t)
            natural=actions[side]
            target={u:a for u,a in natural.items() if u!=worker}
            blocked=cell in arrivals(obs['players'][side],target,worker)
            if blocked:
                raise ValueError(f'Forced root tile conflicts with teammate at {case["match"]}/{turn}')
            actions=list(actions)
            actions[side]=target
            update_memory(mods[side],obs,natural,target)
        if t>turn and task is not None:
            obs=observe(state,side,t)
            natural=actions[side]
            actions[side]=task.follow(obs,natural)
            if natural!=actions[side]:
                amendments.append([t,natural.get(worker,'STAY'),actions[side].get(worker,'STAY')])
            update_memory(mods[side],obs,natural,actions[side])
        for i,a in enumerate(actions):
            validate_actions(a,before.players[i])
        step(state,env,actions)
        after=state[0].observation
        if task is not None:
            task.settle(after,side)
        delta,_,_,changes=cash_flows([{'observation':before},{}],
              [{'observation':after,'action':actions[0]},{'action':actions[1]}],cfg,owners)
        for c,d in zip(ledger,delta):
            c.update(d)
        if t==turn:
            for _,owner,kind,uid,p,_ in changes:
                if owner==side and kind=='plants' and p==cell:
                    target_id=uid
                    target_cost=case['cost']
        if target_id:
            for i,p in enumerate(after.players):
                if target_id in p[3]:
                    target_yield[i]+=after.trees[target_id][1]
                    tracked_turns[i]+=1
        trace.update(bytes.fromhex(digest([actions,board_digest(state)])))
        done=any(x.status!='ACTIVE' for x in state)
        end=t+1
        if end in (turn+1,turn+10,turn+18,turn+35,299) or done:
            bank_hist[end]=dict(cash_margin=round(after.players[side][0]-after.players[1-side][0],2),
                                target_net=round(target_yield[side]-target_yield[1-side]-target_cost,3),
                                ledger=metrics(ledger,side))
        if done:
            break
    assert end==299 or all(s.status=='DONE' for s in state)
    scores=[p.reward for p in state]
    return dict(first=bank_hist[turn+1],end=end,scores=scores,
                margin=round(scores[side]-scores[1-side],2),marks=bank_hist,
                target_id=target_id,target_cost=target_cost,
                target_yield=[round(v,3) for v in target_yield],target_turns=tracked_turns,
                amendments=amendments,exit=task.stop if task is not None else None,
                trajectory=trace.hexdigest())


def evaluate(spec):
    data,out,case,rival,salt,aa=spec
    configure_engine()
    side,turn=case['seat'],case['turn']
    path=data/(case['match']+'-1.html.gz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['replay_sha256']
    game=read_replay(path)
    cfg=game['configuration']
    files=[out/(rival+'.py')]*2
    files[side]=out/'yan-v8.py'
    mods=[model(f,f'tree_plain_{i}',True) for i,f in enumerate(files)]
    state,mem,visible=replay_prefix(game,mods,turn)
    start=archive_state(state)
    reset(mods,mem)
    ours=mods[side]
    normal=[decision(m,state,i,turn,visible,salt) for i,m in enumerate(mods)]
    memories=history(mods)
    traced=model(out/'traced.py','tree_traced_v8',True)
    state,_=restore(start,cfg)
    reset([traced],[mem[side]])
    traced._AUDIT=dict(proposed=[],stages=[])
    assert decision(traced,state,side,turn,visible,salt)==normal[side]
    log=traced._AUDIT
    unit=case['unit']
    result=dict(case=case,proxy=rival,salt=salt,board=digest(start),audit=log,
                first_actions=normal,original_plants=unit not in normal[side])
    arms={}
    replay_arms=('original','original') if aa else ARMS
    for name in replay_arms:
        row=simulate(mods,start,cfg,visible,case,name,salt,normal,memories)
        if aa and name in arms:
            assert row==arms[name], 'A/A diverged'
        arms[name]=row
    if not aa:
        assert arms['plant_once']['first']==arms['continue_task']['first']
        assert arms['plant_once']['target_id']==arms['continue_task']['target_id']
    result['arms']=arms
    return result


def report(rows):
    report={}
    for x,y in zip(ARMS,ARMS[1:]):
        names=y+'-minus-'+x
        paired=[]
        for r in rows:
            before,after=[r['arms'][a] for a in (x,y)]
            paired.append(dict(seed=r['case']['seed'],seat=r['case']['seat'],turn=r['case']['turn'],
                               category=r['case']['category'],proxy=r['proxy'],edited=len(after['amendments']),
                               delta=round(after['margin']-before['margin'],2),
                               cash18=round(after['marks'][r['case']['turn']+18]['cash_margin']-
                                            before['marks'][r['case']['turn']+18]['cash_margin'],2)))
        report[names]=dict(mean=round(mean(r['delta'] for r in paired),2),
                           higher=sum(r['delta']>0 for r in paired),lower=sum(r['delta']<0 for r in paired),
                           equal=sum(r['delta']==0 for r in paired),comparisons=paired)
    return report


def main(data,out):
    out.mkdir(parents=True,exist_ok=False)
    for name in ('yan-v7','yan-v8'):
        spec=REVISIONS[name]
        src=subprocess.check_output(['git','show',spec[0]+':submission.py'],cwd=ROOT)
        assert hashlib.sha256(src).hexdigest()==spec[1]
        (out/(name+'.py')).write_bytes(src)
    source=(out/'yan-v8.py').read_text()
    (out/'traced.py').write_text(build(source,expected_sha256=REVISIONS['yan-v8'][1]))
    cases=choose(data)
    jobs=[(data,out,c,rival,salt,False) for c in cases for rival,salt in PROXIES]
    write(out/'protocol.json',dict(roots=cases,proxy_streams=PROXIES,arms=ARMS,
        budget='width 4, depth at most 2, one opponent reply, three tail steps',
        frozen=REVISIONS))
    aa=(*jobs[-1][:-1],True)
    with ProcessPoolExecutor(3) as pool:
        result=pool.submit(evaluate,aa).result()
        write(out/'aa.json', {k:v for k,v in result.items() if k!='audit'})
        rows=[]
        with gzip.open(out/'branches.jsonl.gz','xt') as f:
            for result in pool.map(evaluate,jobs):
                rows.append(result)
                f.write(json.dumps(result,default=lambda x:x.item())+'\n')
    assert len(rows)==16
    write(out/'results.json',report(rows))


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    os.environ['CARBON_DEBUG']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='1'
    main(args.data.resolve(),args.out.resolve())
