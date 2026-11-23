"""Audit v7 decisions from historical local games with paired closed-loop continuations."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
from instrument import build, V7_SHA256

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/v7'))
from phase_switch import model, budget, preserved_decision
from branch_replay import archive_state, restore, board_digest, step
from benchmark import configure_engine, new_game, observe, validate_actions
from arena import engine_fingerprint
from starvation_replay import cash_flows

SALT = 317
TURNS = (90, 185, 245)


def schedule():
    archive = ROOT/'analysis'
    groups = (
        (archive/'v7/13-confirmation.jsonl', 'selfplay-prior', ROOT/'yan-v6.py', V7_SHA256),
        (archive/'v8/trip-games.jsonl.gz', 'v7-control', ROOT/'yan-v7.py', V7_SHA256),
    )
    tasks = []
    for path, own, opponent, digest in groups:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path,'rt') as records:
            rows = [row for row in map(json.loads,records)
                    if row['status']=='completed' and row['turns']==299 and
                    any(p['name']==own and p['sha256']==digest for p in row['players'])]
        seeds = sorted({row['seed'] for row in rows},
                       key=lambda n: hashlib.sha256(f'decision-audit-pilot/{n}'.encode()).digest())
        if len(seeds)<2:
            raise ValueError(f'Missing completed games from {path}')
        for seed in seeds[:2]:
            eligible = [row for row in rows if row['seed']==seed]
            index = len(tasks) % len(eligible)
            row = eligible[index]
            seat = next(i for i,p in enumerate(row['players']) if p['name']==own)
            assert hashlib.sha256(opponent.read_bytes()).hexdigest()==row['players'][1-seat]['sha256']
            tasks.append((row, seat, opponent))
    return tasks


def history(mods):
    return copy.deepcopy([(m._ST,m._FC_HISTORY) for m in mods])


def reset(mods, snapshot):
    for m, (st, fc) in zip(mods,copy.deepcopy(snapshot)):
        m._ST, m._FC_HISTORY = st, fc
        budget(m,True)


def choose(m,state,side,t,cfg):
    obs = observe(state,side,t)
    actions = preserved_decision(m,obs,cfg,SALT)
    validate_actions(actions,obs['players'][side])
    return actions


def commit_action(m,obs,action):
    m._ST['tr'].set_actions(obs['players'],action)
    m._ST['liquidity_action'] = dict(action)


def continuation(state, env, mods, cfg, own, turn, first):
    totals = [Counter(),Counter()]
    owners = {u:i for i,p in enumerate(state[0].observation.players) for u in p[3]}
    marks = {}
    for t in range(turn,299):
        before = state[0].observation
        acts = first if t==turn else [choose(m,state,i,t,cfg) for i,m in enumerate(mods)]
        step(state,env,acts)
        after = state[0].observation
        flows,*_ = cash_flows([{'observation':before},{}],[{'observation':after,'action':acts[0]},{'action':acts[1]}],cfg,owners)
        for team,f in enumerate(flows):
            totals[team].update(f)
        done = any(x.status!='ACTIVE' for x in state)
        if t+1 in (turn+10,turn+35,299) or done:
            marks[t+1] = {
                'margin':round(after.players[own][0]-after.players[1-own][0],2),
                'bank_gap':round(totals[own]['base']-totals[1-own]['base'],3),
                'tree_gap':round(sum((totals[own][k]-totals[1-own][k])*w for k,w in
                                     [('tree',1),('collision_tree',1),('plant_cost',-1),('seize_cost',-1)]),3),
            }
        if done:
            break
    return dict(steps=t+1,margin=round(state[own].reward-state[1-own].reward,2),
                scores=[p.reward for p in state],marks=marks)


def signature(action):
    return tuple(sorted(action.items()))


def branch_case(snapshot, raw_history, mods, cfg, visible, own, turn, trace_index):
    board, info = snapshot
    state,env = restore(board,cfg)
    reset(mods,raw_history)
    ours = mods[own]
    ours._AUDIT = dict(proposed=[],stages=[])
    selected = [choose(m,state,i,turn,visible) for i,m in enumerate(mods)]
    trace, ours._AUDIT = ours._AUDIT,None
    executed = selected[own]
    stages = dict(trace['stages'])
    if stages['executed']!=executed:
        raise ValueError('Action trace did not match executed move')
    after_decision = history(mods)
    admitted = trace.get('admitted',[])
    finalists = sorted(trace.get('final_scores',{}),key=trace['final_scores'].get,reverse=True)
    choices = defaultdict(list)
    sources = defaultdict(set)
    for name,act in trace['proposed']+trace.get('probes',[]):
        sources[signature(act)].add(name)

    def include(label, action):
        choices[signature(action)].append(label)

    include('executed',executed)
    include('search',stages.get('search',executed))
    for candidate in finalists[:2]:
        include('scored-finalist',admitted[candidate])
    # Choose a legal probe outside the admitted pool, independent of its outcome.
    known = set(map(signature,admitted)) | choices.keys()
    extra = sorted({signature(a) for _,a in trace.get('probes',[]) if signature(a) not in known
                    and all(a.get(u)==move for u,move in trace['protected'].items())})
    if extra:
        include('outside-pool',dict(extra[trace_index % len(extra)]))

    results=[]
    for key,labels in choices.items():
        state,env = restore(board,cfg)
        first = selected[:]
        act = dict(key)
        first[own] = act
        validate_actions(act,state[0].observation.players[own])
        reset(mods,after_decision)
        commit_action(ours,observe(state,own,turn),act)
        result = continuation(state,env,mods,visible,own,turn,first)
        results.append(dict(labels=labels,sources=sorted(sources[key]),action=act,**result))
    # With tracing off, check that instrumentation observed the same actual move.
    state,_ = restore(board,cfg)
    reset(mods,raw_history)
    assert choose(ours,state,own,turn,visible)==executed
    return dict(turn=turn, snapshot=info, trace=trace, replies=selected[1-own], results=results)


def audit_match(args):
    row, own, opponent, trace_code = args
    configure_engine()
    from zerosum_env.envs.carbon import carbon
    assert engine_fingerprint(carbon)==row['engine_sha256']
    paths = [opponent,opponent]
    paths[own] = trace_code
    mods = [model(path,f'root_audit_{i}',True) for i,path in enumerate(paths)]
    state,env = new_game(row['seed'],300)
    start=state[0].observation
    layout=[env.configuration.size,start.full_carbon,[sorted(p[1].values()) for p in start.players]]
    assert hashlib.sha256(json.dumps(layout,separators=(',',':')).encode()).hexdigest()==row['layout_sha256']
    cfg = dict(env.configuration)
    visible = {k:v for k,v in cfg.items() if k!='randomSeed'}
    for m in mods:
        m._ST,m._FC_HISTORY = {},{}
        budget(m,False)
    snapshots = {}
    for t,(actual,expected) in enumerate(row['actions_and_post_cash']):
        if t in TURNS:
            snapshots[t] = ((archive_state(state),board_digest(state)),history(mods))
        for i,m in enumerate(mods):
            choose(m,state,i,t,visible)
            commit_action(m,observe(state,i,t),actual[i])
        step(state,env,actual)
        assert [p[0] for p in state[0].observation.players]==expected
        if t==max(TURNS):
            break
    roots = [branch_case(snapshots[t][0],snapshots[t][1],mods,cfg,visible,own,t,i) for i,t in enumerate(TURNS)]
    return dict(seed=row['seed'],layout=row['layout_sha256'],own=own,opponent=opponent.stem,roots=roots)


def summarize(games):
    stages = Counter()
    sources = defaultdict(Counter)
    differences = defaultdict(list)
    for game in games:
        for root in game['roots']:
            tr = root['trace']
            prior = None
            for name,act in tr['stages']:
                stages[name+':visited']+=1
                if prior is not None and prior!=act:
                    stages[name+':changed']+=1
                prior=act
            pool = list(map(signature,tr.get('admitted',[])))
            groups = defaultdict(set)
            for name,a in tr['proposed']:
                groups[signature(a)].add(name)
            for key, names in groups.items():
                for name in names:
                    sources[name]['unique_generated']+=1
                    if key in pool:
                        sources[name]['admitted']+=1
                        if pool.index(key) in tr.get('scored_depth',{}):
                            sources[name]['scored']+=1
                    else:
                        sources[name]['not_admitted']+=1
            results=root['results']
            base=next(r for r in results if 'executed' in r['labels'])
            for label in ('search','outside-pool'):
                row=next((r for r in results if label in r['labels']),None)
                if row:
                    differences[label].append(round(row['margin']-base['margin'],2))
    return dict(stages=stages,sources=sources, comparisons={k:dict(count=len(v),mean=round(mean(v),2),
                higher=sum(x>0 for x in v),equal=sum(x==0 for x in v),values=v) for k,v in differences.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    os.environ['CARBON_DEBUG']='1'
    a.out.mkdir(parents=True,exist_ok=False)
    script=a.out/'instrumented.py'
    source=build((ROOT/'yan-v7.py').read_text())
    compile(source,str(script),'exec')
    script.write_text(source)
    tasks=schedule()
    with ProcessPoolExecutor(a.jobs) as pool, (a.out/'games.jsonl').open('x') as f:
        results=[]
        for game in pool.map(audit_match,[(*args,script.resolve()) for args in (tasks[:1] if a.smoke else tasks)]):
            results.append(game)
            f.write(json.dumps(game)+'\n')
            f.flush()
    report=dict(v7=V7_SHA256,salt=SALT,turns=TURNS,games=len(results),**summarize(results))
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    main()
