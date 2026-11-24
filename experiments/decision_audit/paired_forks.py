"""Deploy each finalist through the v7 execution chain and measure paired terminal returns."""
import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from types import FunctionType

from run import (ROOT, V7_SHA256, archive_state, board_digest, budget, configure_engine,
                 history, model, new_game, observe, preserved_decision, reset, restore, step,
                 validate_actions)
from branch_replay import digest, fields
from rank import _v8_rank_features
from starvation_replay import LAND, cash_flows, distance, read_replay
from stratify import classify, select

ROOT_SALT = 317
STREAMS = (733, 1889, 4099)
BASE = ROOT/'yan-v7.py'


def traced_source():
    src = BASE.read_bytes()
    assert hashlib.sha256(src).hexdigest() == V7_SHA256
    text = src.decode()
    anchor = '    if _S_REC is not None:'
    assert text.count(anchor) == 1
    hook = '''    if _fork_record is not None:
        _fork_record.update(selected=best, actions={i:dict(cands[i]) for i in alive},
                            scores={i:float(mean[i]) for i in alive})
        if _fork_force is not None:
            assert _fork_force in alive
            best = _fork_force
'''
    return text.replace(anchor, hook+anchor) + '\n_fork_record = _fork_force = None\n'


def decision(mod, state, seat, turn, cfg, salt):
    obs = observe(state, seat, turn)
    a = preserved_decision(mod, obs, cfg, salt)
    validate_actions(a, obs['players'][seat])
    return a


def warm(mod, state, seat, turn, cfg, action):
    proposed = decision(mod, state, seat, turn, cfg, ROOT_SALT)
    st = mod._ST
    st['tr'].set_actions(state[0].observation.players, action)
    st['liquidity_action'] = dict(action)
    pending = st.get('v7_next_depot')
    if pending and proposed.get(pending[1]) != action.get(pending[1]):
        st.pop('v7_next_depot')
    if 'corridor_track' in st:
        st['corridor_track'] = [x for x in st['corridor_track'] if proposed.get(x[1]) == action.get(x[1])]


def prefix(data, case, mods, turn):
    state, mem, visible = replay_prefix(data,mods,turn)
    flags, context = classify(state[0].observation.players, case['seat'], data['configuration'])
    assert flags[case['category']]
    return archive_state(state), mem, visible, context


def replay_prefix(data, mods, turn):
    from zerosum_env.envs.carbon import idgen
    frames = data['steps']
    cfg = data['configuration']
    visible = {k:v for k,v in cfg.items() if k != 'randomSeed'}
    new_game(cfg['randomSeed'], 300)
    initial = dict(seats=[{**{k:p[k] for k in ('observation','status','reward')},
                           'action':p['action'] or {}} for p in frames[0]], id_counters=dict(idgen._id_dict))
    state, env = restore(initial, cfg)
    for m in mods:
        m._ST, m._FC_HISTORY = {}, {}
        budget(m, False)
    for t in range(turn+1):
        assert fields([p.observation for p in state],[p.reward for p in state]) == fields(
            [p['observation'] for p in frames[t]], [p['reward'] for p in frames[t]])
        if t == turn:
            break
        actions = [p['action'] or {} for p in frames[t+1]]
        for side, mod in enumerate(mods):
            warm(mod,state,side,t,visible,actions[side])
        step(state,env,actions)
    return state, history(mods), visible


def summaries(obs, a, b):
    side = obs['player']
    own, foe = obs['players'][side], obs['players'][1-side]
    details = []
    for uid in sorted(a.keys() | b.keys()):
        if a.get(uid) == b.get(uid):
            continue
        info = dict(uid=uid, commands=[x.get(uid,'STAY') for x in (a,b)])
        if uid in own[2]:
            pos, cargo, kind = own[2][uid]
            tiles = [LAND[pos][k] for k in info['commands']]
            lo = [p for p,v,k in foe[2].values() if k=='COLLECTOR' and v<=cargo]
            info.update(kind=kind,position=pos,cargo=cargo,tiles=tiles,
                        depot_distance=[min(distance(q,h) for h in own[1].values()) for q in tiles],
                        lighter_enemy_distance=[min((distance(q,h) for h in lo), default=None) for q in tiles],
                        friendly_arrivals=[sum(LAND[p][act.get(w,'STAY')]==q for w,(p,_,_) in own[2].items() if w!=uid)
                                           for act,q in zip((a,b),tiles)])
        else:
            info.update(kind='BASE',position=own[1][uid])
        details.append(info)
    return details


def prepare(data_path, trace_path, case, aa=False):
    configure_engine()
    mods = [model(BASE, 'fork_side0', True), model(BASE, 'fork_side1', True)]
    me = case['seat']
    agent = model(trace_path, 'fork_instrument', True)
    path = data_path/case['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['source_sha256']
    replay = read_replay(path)
    cfg = replay['configuration']
    for turn in case['turns']:
        start, mem, visible, context = prefix(replay, case, mods, turn)
        assigned = mods.copy()
        assigned[me] = agent
        state, _ = restore(start,cfg)
        reset(assigned,mem)
        obs = observe(state,me,turn)
        agent._fork_record, agent._fork_force = {}, None
        natural = decision(agent,state,me,turn,visible,ROOT_SALT)
        record, agent._fork_record = agent._fork_record, None
        if len(record.get('scores',{})) < 2 or record['selected'] not in record['scores']:
            continue
        first = record['selected']
        second = max((i for i in record['scores'] if i!=first), key=record['scores'].get)
        break
    else:
        raise ValueError(f"No two finalists in {case['file']} for {case['category']}")
    raw = [record['actions'][i] for i in (first,second)]
    assert raw[0] != raw[1]
    arms = []
    for index in (first,second):
        state, _ = restore(start,cfg)
        reset(assigned,mem)
        agent._fork_record, agent._fork_force = {}, index
        acts = [decision(m,state,i,turn,visible,ROOT_SALT) for i,m in enumerate(assigned)]
        assert agent._fork_record == record
        agent._fork_force = agent._fork_record = None
        arms.append((acts,history(assigned)))
    assert natural == arms[0][0][me] and arms[0][0][1-me] == arms[1][0][1-me]
    feature = FunctionType(_v8_rank_features.__code__, agent.__dict__)
    root = dict(**{k:v for k,v in case.items() if k!='turns'},turn=turn, attempts=case['turns'].index(turn)+1,
                board=digest(start), context=context, raw=raw, indices=[first,second],
                scores=[record['scores'][i] for i in (first,second)], executed=[a[0][me] for a in arms],
                enemy_action=arms[0][0][1-me], features=[feature(obs,me,x) for x in raw],
                differences=summaries(obs,arms[0][0][me],arms[1][0][me]))
    if aa:
        state, _ = restore(start,cfg)
        reset(mods,mem)
        acts = [decision(m,state,i,turn,visible,ROOT_SALT) for i,m in enumerate(mods)]
        assert acts == arms[0][0]
        arms.append((acts,history(mods)))
    return mods, start, cfg, visible, root, arms


def play(mods, start, cfg, visible, root, arm, salt):
    actions, memory = arm
    state, env = restore(start,cfg)
    reset(mods,memory)
    me, turn = root['seat'], root['turn']
    totals = [Counter(),Counter()]
    owners = {u:i for i,p in enumerate(state[0].observation.players) for u in p[3]}
    trajectory = hashlib.sha256()
    marks = {}
    for t in range(turn,299):
        if t != turn:
            actions = [decision(m,state,i,t,visible,salt) for i,m in enumerate(mods)]
        before = state[0].observation
        step(state,env,actions)
        after = state[0].observation
        trajectory.update(bytes.fromhex(digest([actions, board_digest(state)])))
        flows,*_ = cash_flows([{'observation':before},{}],
                              [{'observation':after,'action':actions[0]},{'action':actions[1]}],cfg,owners)
        for bucket,flow in zip(totals,flows):
            bucket.update(flow)
        if t+1 in (turn+1, turn+10,turn+35,299):
            diff = totals[me].copy()
            diff.subtract(totals[1-me])
            marks[t+1] = dict(margin=round(after.players[me][0]-after.players[1-me][0],2),
                              depot=round(diff['base'],2),
                              tree=round(diff['tree']+diff['collision_tree']-diff['plant_cost']-diff['seize_cost'],2))
        if any(p.status!='ACTIVE' for p in state):
            break
    assert (t+1 == 299 and all(p.status in ('ACTIVE','DONE') for p in state)
            or all(p.status=='DONE' for p in state)), 'Non-terminal continuation'
    scores = [p.reward for p in state]
    return dict(end=t+1, statuses=[p.status for p in state], scores=scores, margin=round(scores[me]-scores[1-me],2), marks=marks,
                trajectory=trajectory.hexdigest(), terminal=board_digest(state))


def run_case(args):
    data, traced, case, aa = args
    mods,start,cfg,visible,root,arms = prepare(data,traced,case,aa)
    if aa:
        x,y = [play(mods,start,cfg,visible,root,arms[i],STREAMS[0]) for i in (2,0)]
        assert x == y, 'Frozen and forced-v7 A/A trajectories diverged'
        return dict(root=root['board'],turn=root['turn'],seed=root['seed'],seat=root['seat'],stream=STREAMS[0],**x)
    root['streams'] = []
    for salt in STREAMS:
        a,b = [play(mods,start,cfg,visible,root,arm,salt) for arm in arms]
        root['streams'].append(dict(salt=salt,baseline=a,alternative=b,delta=round(b['margin']-a['margin'],2)))
    return root


def write(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    traced = out/'instrumented.py'
    traced.write_text(traced_source())
    cases = select(args.data)
    write(out/'protocol.json',dict(tag='v10-yan-v7-selfplay-prior',sha256=V7_SHA256,root_salt=ROOT_SALT,
                                  streams=STREAMS,proxy='yan-v7',cases=cases))
    jobs = [(args.data.resolve(),traced,c,False) for c in cases]
    with ProcessPoolExecutor(3) as pool:
        # Validate an independent uninstrumented v7 trajectory before releasing the batch.
        write(out/'aa.json',pool.submit(run_case,(*jobs[-1][:3],True)).result())
        with gzip.open(out/'roots.jsonl.gz','xt') as f:
            for root in pool.map(run_case,jobs):
                f.write(json.dumps(root)+'\n')
    with gzip.open(out/'roots.jsonl.gz','rt') as f:
        results = list(map(json.loads,f))
    effects = [[s['delta'] for s in r['streams']] for r in results]
    assert len(effects) == 12 and all(len(d)==3 for d in effects)
    write(out/'summary.json',dict(roots=12,branches=72,higher=sum(min(d)>0 for d in effects),
         lower=sum(max(d)<0 for d in effects),mixed=sum(min(d)<0<max(d) for d in effects),
         average=round(mean(mean(d) for d in effects),2),deltas=effects))


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    os.environ['CARBON_DEBUG']='1'
    main()
