"""Paired 35-turn blockade rescue tests on exact historical states with one CPU thread."""
from collections import Counter
import copy
import gzip
import json
import os

from branch_replay import ROOT, OUT, HERE, extract, board_digest, step, write
from branch_run import fixed_release, sha
from benchmark import configure_engine, limit_math_threads, load_strategy, observe, validate_actions
from close_replay import archive_state, restore, referee_events

STARTS=(40,60,90,164)
FROZEN= "31d69d74bcfe28ea08d509bad919165678808a9c1203bdcb93a43fc23e3154d2"
PACKED=OUT/"branch-breakout-replays.json.gz"
SEED=22063
MATCHES={
    "loss":("4063a34afedb6d4767a9118e1dee9e5d",157458776),
    "win":("cf276b32e4fd38f43497e60f271a3854",1941966395),
}


def load():
    import breakout
    import lanes
    import torch
    torch.set_num_threads(1)
    assert sha(HERE/"breakout.py")==FROZEN
    base=fixed_release()
    public=ROOT/"opponents/claude-AlphaCarbon-Master3s2.py"
    enemy=load_strategy("breakout_public_master",public,public.read_text().replace("_torch.set_num_threads(2)","_torch.set_num_threads(1)"))
    base._reply_choice=lambda obs,cfg,est,action,*rest:action
    for m in (base,enemy):
        def capture(obs,cfg,est,action,worker_logits,*args,policy=m):
            policy.root_logits=worker_logits
            return action
        m._search=capture
    breakout._V7_D,breakout._V7_L=lanes._V7_D,lanes._V7_L
    breakout.WIDX,breakout.WACT,breakout._probs,breakout.LAST_TURN=base.WIDX,base.WACT,base._probs,298
    base._torch.set_num_threads(1)
    return base,enemy,breakout


def snapshot(base,enemy,hist):
    return copy.deepcopy((base._ST,base._FC_HISTORY,enemy._ST,hist))


def restore_policy(saved,base,enemy):
    base._ST,base._FC_HISTORY,enemy._ST,hist=copy.deepcopy(saved)
    return hist


def choose(m,obs,cfg,seat,t):
    import numpy as np
    m._ST['rng']=np.random.RandomState(SEED+1000003*(t+1)+seat)
    return m.agent(obs,cfg)


def load_replay(path, name):
    if path.exists():
        replay=extract(path)
        replay.update(match_id=MATCHES[name][0], game=1, seats=["claude-AlphaCarbon-Master7sE","yan-QQ-v6"])
    else:
        replay=json.loads(gzip.decompress(PACKED.read_bytes()))[name]
    return replay


def prepare(path,name,base,enemy,route):
    d=load_replay(path,name)
    config=d['configuration'];cfg={k:v for k,v in config.items() if k!='randomSeed'}
    state,env=restore(d['initial'],config)
    base._ST={};enemy._ST={};base._FC_HISTORY={};hist={}
    saves={}; public=[];previous=None;move={}
    for t,expected in enumerate(d['expected']):
        if board_digest(state)!=expected:raise ValueError((path,t))
        obs=[observe(state,s,t) for s in (0,1)];public.append(obs)
        if t in STARTS:
            saves[t]=(archive_state(state),snapshot(base,enemy,hist),previous,move)
        if t==len(d['actions']):break
        acts=d['actions'][t]
        if t<max(STARTS):
            route._v7_watch_cash(obs[1],previous,move,hist)
            for s,m in enumerate((enemy,base)):
                choose(m,obs[s],cfg,s,t)
                m._ST['tr'].set_actions(obs[s]['players'],acts[s])
        previous,move=obs[1]['players'],acts[1]
        step(state,env,acts)
    return d,saves,public,cfg


def play(start,checkpoint,data,public,cfg,base,enemy,route,source,apply):
    from zerosum_env.envs.carbon import idgen
    snap,mem,previous,prior=checkpoint
    hist=restore_policy(mem,base,enemy)
    state,env=restore(snap,data['configuration'])
    matched=set(state[0].observation.players[0][2]);matched_missing=0
    totals=[Counter(),Counter()];marks={}; opportunities=[]
    for t in range(start,start+35):
        obs=[observe(state,s,t) for s in (0,1)]
        workers=obs[0]['players'][0][2]
        ids=dict(idgen._id_dict)
        if source=='recorded':
            action,ours=data['actions'][t]
        else:
            if source=='live-master3s2':
                action=choose(enemy,obs[0],cfg,0,t)
            else:
                commands=data['actions'][t][0]
                allowed=matched|obs[0]['players'][0][1].keys()
                action={k:v for k,v in commands.items() if k in allowed}
                matched_missing+=len(workers.keys()-matched)
            idgen._id_dict.clear();idgen._id_dict.update(ids)
            ours=choose(base,obs[1],cfg,1,t)
            drought=route._v7_watch_cash(obs[1],previous,prior,hist)
            alt=route._v7_cash_route(obs[1],cfg,ours,base.root_logits,base._ST['v5_history']['opponent'][0],drought)
            if alt!=ours:
                changed=[k for k in sorted(alt.keys()|ours.keys()) if alt.get(k)!=ours.get(k)]
                opportunities.append([t,changed,round(obs[1]['players'][1][0],2),drought])
                if apply:ours=alt
            base._ST['tr'].set_actions(obs[1]['players'],ours)
            previous,prior=obs[1]['players'],ours
        idgen._id_dict.clear();idgen._id_dict.update(ids)
        for a,p in zip((action,ours),obs[0]['players']): validate_actions(a,p)
        old=set(workers)
        ev=[]
        with referee_events(ev):step(state,env,[action,ours])
        posts=state[0].observation.players
        if source=='recorded' and board_digest(state)!=data['expected'][t+1]:raise ValueError(('restore',t))
        if source=='open-loop':
            new=posts[0][2];actual=public[t+1][0]['players'][0][2]
            was=public[t][0]['players'][0][2]
            matched.intersection_update(new.keys()&actual.keys())
            matched.update(k for k in (new.keys()-old)&(actual.keys()-was) if new[k][::2]==actual[k][::2])
        for kind,s,amount,*rest in ev:
            if kind in ('bank','base_collision'):totals[s]['deposits']+=amount
            if kind=='bank':totals[s]['smelts']+=1
            if kind=='death':totals[s]['deaths']+=1
        if t-start+1 in (20,35):
            marks[t-start+1]=[dict(cash=p[0],workers=len(p[2]),trees=len(p[3]),cargo=round(sum(x[1] for x in p[2].values()),2),
                                    deposits=round(totals[i]['deposits'],2),smelts=totals[i]['smelts'],
                                    minimum_recruit_reserve=p[0]>=cfg['recCollectorCost'],
                                    deaths=totals[i]['deaths']) for i,p in enumerate(posts)]
    return dict(marks=marks,attempts=opportunities,open_loop_unmatched_worker_turns=matched_missing)


def main():
    os.environ['CARBON_DEBUG']='1'
    limit_math_threads();configure_engine()
    base,enemy,route=load();cases=[]
    for name,filename in [('loss','starvation-4063.html.gz'),('win','starvation-candidate.html.gz')]:
        path=ROOT.parent/'v7-data'/filename
        data,saves,public,cfg=prepare(path,name,base,enemy,route)
        label=dict(map=name,match_id=MATCHES[name][0],listed_seed=MATCHES[name][1],game=1,config_seed=data['configuration']['randomSeed'],source_sha256=data["source_gzip_sha256"],scores=data['final_rewards'],
                   verified_original_frames=len(data['expected']),positions=[])
        for t in STARTS:
            block=dict(start=t,initial=public[t][1]['players'][1][0],methods={})
            for source in ('recorded','open-loop','live-master3s2'):
                args=(t,saves[t],data,public,cfg,base,enemy,route,source)
                control=play(*args,False)
                block['methods'][source]={'control':control}
                if source!='recorded':
                    # No alternate action anywhere on this path proves coincident deterministic trajectories.
                    treated=play(*args,True) if control['attempts'] else control
                    block['methods'][source].update(treated=treated,identical_without_activation=not bool(control['attempts']))
            label['positions'].append(block)
            print(name,t,[(s,len(v['control']['attempts'])) for s,v in block['methods'].items()],flush=True)
        cases.append(label)
    write(OUT/'branch-breakout.json',dict(function_sha256=FROZEN,public_master3s2_sha256=sha(ROOT/'opponents/claude-AlphaCarbon-Master3s2.py'),
          release_sha256=sha(ROOT/'yan-v6.py'),rng_seed=SEED,qq_seat=1,smelt_cost=0,starts=STARTS,
          policy_budget='Frozen v6 root logits, collision reservation, endgame and normal root action; zero forward search or reply reranking on either side. Paired random streams keyed by turn and seat.',
          open_loop='Original Master7sE commands on continuing identities. New workers must be born in both runs at same time, square and role. Unmatched workers idle. Open loop cannot respond to changes.',
          cases=cases))


if __name__=='__main__':main()
