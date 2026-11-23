"""Prospective tree threats in ten CarbonEdge games, with paired continuations to terminal."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys,copy,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from statistics import mean
from grove import _grove_threats,build
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/v7'))
from phase_switch import preserved_decision,model,budget,archive_state,restore,board_digest,observe,step,new_game,configure_engine,validate_actions
from branch_replay import fields
from starvation_replay import read_replay,distance,cash_flows

DIST=[[distance(a,b) for b in range(225)] for a in range(225)]
SALTS=(317,921)


def select(data):
    picked=[]
    for g in json.loads((data/'metrics.json').read_text()):
        if g['group']!='v7_top5' or g['opp'] not in ('CarbonEdge_Apollo','CarbonEdge_Prometheus','Artemis'):
            continue
        name=f"{g['id']}-{g['game']}.html.gz"
        f=read_replay(data/name)['steps'];side=g['seat'];used=set()
        for low,high in ((35,110),(110,190),(190,275)):
            pool=[]
            for t in range(low,high):
                o=dict(f[t][0]['observation'],player=side)
                for tid,q,age,rate,z in _grove_threats(o,DIST):
                    if tid not in used:
                        key=hashlib.sha256(f'grove/v1/{name}/{t}/{tid}'.encode()).digest()
                        pool.append((key,t,tid,q))
            _,t,tid,q=min(pool);used.add(tid)
            taken=any(tid in frame[0]['observation']['players'][1-side][3] for frame in f[t+1:t+7])
            picked.append(dict(file=name,side=side,t=t,tid=tid,q=q,taken=taken,opponent=g['opp']))
    assert len(picked)==30 and 0<sum(x['taken'] for x in picked)<30
    return picked


def play(args):
    data,out,cases=args
    configure_engine();e=read_replay(data/cases[0]['file']);frames=e['steps'];cfg=e['configuration']
    visible={k:v for k,v in cfg.items() if k!='randomSeed'}
    side=cases[0]['side']
    modules=[model(ROOT/'yan-v7.py','grove_0',True),model(ROOT/'yan-v7.py','grove_1',True)]
    guard=model(out/'yan-v8-grove.py','grove_policy',True)
    from zerosum_env.envs.carbon import idgen
    new_game(cfg['randomSeed'],300)
    initial=dict(seats=[dict(observation=x['observation'],status=x['status'],reward=x['reward'],action=x['action'] or {}) for x in frames[0]],id_counters=idgen._id_dict.copy())
    state,env=restore(initial,cfg)
    for m in modules:
        m._ST={};m._FC_HISTORY={};budget(m,False)
    snaps={}
    starts={c['t'] for c in cases}
    for t in range(max(starts)+1):
        f=frames[t]
        assert fields([p.observation for p in state],[p.reward for p in state])==fields([p['observation'] for p in f],[p['reward'] for p in f])
        if t in starts:
            snaps[t]=(archive_state(state),copy.deepcopy([(m._ST,m._FC_HISTORY) for m in modules]),board_digest(state))
        if t==max(starts):break
        acts=[p['action'] or {} for p in frames[t+1]]
        for i,m in enumerate(modules):
            ob=observe(state,i,t)
            preserved_decision(m,ob,visible,317)
            m._ST['tr'].set_actions(ob['players'],acts[i]);m._ST['liquidity_action']=dict(acts[i])
        step(state,env,acts)
    rows=[]
    for case in cases:
        start,mem,digest=snaps[case['t']]
        for salt in SALTS:
            pair=[]
            for special in (False,True):
                state,env=restore(start,cfg)
                assert board_digest(state)==digest
                mods=list(modules)
                if special:mods[side]=guard
                for m,history in zip(mods,mem):
                    m._ST,m._FC_HISTORY=copy.deepcopy(history);budget(m,True)
                totals=[Counter(),Counter()];attempts=0;close_attempts=0;taken=False
                owners={uid:s for s,pl in enumerate(state[0].observation.players) for uid in pl[3]}
                for t in range(case['t'],299):
                    before=state[0].observation
                    moves=[]
                    for i,m in enumerate(mods):
                        ob=observe(state,i,t)
                        a=preserved_decision(m,ob,visible,salt)
                        validate_actions(a,ob['players'][i]);moves.append(a)
                    enemy=before.players[1-side]
                    target=case['tid']
                    if target in before.players[side][3] and enemy[0]>=cfg['seizeCost']:
                        probe=any(p==case['q'] and k=='PLANTER' and not moves[1-side].get(u) for u,(p,_,k) in enemy[2].items())
                        attempts+=probe
                        if t<case['t']+6:close_attempts+=probe
                    step(state,env,moves)
                    after=state[0].observation
                    if t<case['t']+6:taken|=target in after.players[1-side][3]
                    f,*_=cash_flows([{'observation':before},{}],[{'observation':after,'action':moves[0]},{'action':moves[1]}],cfg,owners)
                    for s,flow in enumerate(f):totals[s].update(flow)
                    if any(p.status!='ACTIVE' for p in state):break
                def net(s):
                    f=totals[s]
                    return f['tree']+f['collision_tree']-f['plant_cost']-f['seize_cost']
                pair.append(dict(turn=t+1,margin=round(state[side].reward-state[1-side].reward,2),own_net=round(net(side),3),tree_gap=round(net(side)-net(1-side),3),target_attempts=attempts,attempts6=close_attempts,taken6=taken))
            rows.append(dict(**case,salt=salt,start_hash=digest,baseline=pair[0],guard=pair[1]))
    return rows


def main():
    data,out=map(lambda x:Path(x).resolve(),sys.argv[1:])
    selected=select(data)
    (out/'grove-manifest.json').write_text(json.dumps(selected,indent=2))
    (out/'yan-v8-grove.py').write_text(build((ROOT/'yan-v7.py').read_text()))
    groups=[[c for c in selected if c['file']==key] for key in sorted({x['file'] for x in selected})]
    with ProcessPoolExecutor(3) as pool, (out/'grove-results.jsonl').open('x') as f:
        for group in pool.map(play,[(data,out,g) for g in groups]):
            for row in group:f.write(json.dumps(row)+'\n')
            f.flush()

if __name__=='__main__':main()
