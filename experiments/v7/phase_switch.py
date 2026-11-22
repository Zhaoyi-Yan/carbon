"""Paired second halves from eight historical boards, with independently reacting frozen seats."""
import os
import sys

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')
if __name__ == '__main__' and (any(os.environ.get(k) != '1' for k in THREAD_KEYS) or os.environ.get('PYTHONHASHSEED') != '0'):
    os.environ.update(dict.fromkeys(THREAD_KEYS, '1'), PYTHONHASHSEED='0')
    os.execv(sys.executable, [sys.executable, *sys.argv])
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from branch_replay import ROOT, HERE, OUT, board_digest, digest, step
from close_replay import archive_state, restore, referee_events
from preempt_tree_probe import model, chooser, budget
from benchmark import configure_engine, new_game, observe, validate_actions
from arena import engine_fingerprint

BRANCH = 150
ARMS = ('soundings', 'tidewatch', 'estuary')
SOURCES = {
    'development': 'v6r-qual-05-estuary.jsonl',
    'holdout': 'v6r-confirm-05-estuary.jsonl',
}
HASHES = {
    'soundings': '95d5d869385f83f401aa511b151b8dbdfe692bec02893d1d063465b46dd7247b',
    'tidewatch': '101586bc07558cbcd6753c48e7973e85602776d1e5b4976300084782cef834d4',
    'estuary': 'e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0',
}
CASH_RULES = {'bank', 'base_collision', 'hire_collector', 'hire_planter', 'plant', 'seize', 'tree_yield', 'rounding', 'tree_collision'}


def single_thread():
    import torch
    from threadpoolctl import threadpool_info
    assert all(os.environ.get(k) == '1' for k in THREAD_KEYS)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pools = [{k: p[k] for k in ('internal_api', 'prefix', 'num_threads')} for p in threadpool_info()]
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1 and all(p['num_threads'] == 1 for p in pools), pools
    return dict(environment={k: os.environ[k] for k in THREAD_KEYS}, torch_intraop=torch.get_num_threads(),
                torch_interop=torch.get_num_interop_threads(), pools=pools, workers=1)


def selected():
    choices = []
    for split, filename in SOURCES.items():
        rows = [json.loads(line) for line in (ROOT / 'analysis' / filename).open()]
        seeds = sorted({r['seed'] for r in rows}, key=lambda n: hashlib.sha256(f'v7-phase-150/{n}'.encode()).digest())
        for i, seed in enumerate(seeds[:4]):
            side = i % 2
            row, = [r for r in rows if r['seed'] == seed and r['players'][side]['name'] == 'estuary']
            assert row['turns'] == 299 and row['status'] == 'completed'
            choices.append((split, side, row))
    assert len({r['layout_sha256'] for _, _, r in choices}) == len(choices) == 8
    return choices


def checkpoint(name, tag):
    path = HERE / 'checkpoints' / f'{name}.py' if name != 'estuary' else ROOT / 'experiments/v6r/checkpoints/estuary.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[name]
    return model(path, 'switch_' + tag, True)


def preserved_decision(policy, obs, visible, salt):
    from zerosum_env.envs.carbon import idgen
    ids = idgen._id_dict.copy()
    try:
        return chooser(policy, obs, visible, salt)
    finally:
        idgen._id_dict.clear()
        idgen._id_dict.update(ids)


def prepare(row, side, modules, salt):
    from zerosum_env.envs.carbon import carbon
    assert engine_fingerprint(carbon) == row['engine_sha256']
    state, env = new_game(row['seed'], 300)
    cfg = dict(env.configuration)
    visible = {k:v for k,v in cfg.items() if k != 'randomSeed'}
    roles = [side, side, side, 1-side]
    for p in modules:
        p._ST, p._FC_HISTORY = {}, {}
        budget(p, False)
    for t, (actions, expected_cash) in enumerate(row['actions_and_post_cash']):
        if t == BRANCH:
            snap = archive_state(state)
            mid_hash = board_digest(state)
            memory = copy.deepcopy([(p._ST, p._FC_HISTORY) for p in modules])
        if t < BRANCH:
            for p, s in zip(modules, roles):
                obs = observe(state, s, t)
                proposed = preserved_decision(p, obs, visible, salt)
                actual = actions[s]
                p._ST['tr'].set_actions(obs['players'], actual)
                if 'liquidity_action' in p._ST:
                    p._ST['liquidity_action'] = dict(actual)
                pending = p._ST.get('v7_next_depot')
                if pending and proposed.get(pending[1]) != actual.get(pending[1]):
                    p._ST.pop('v7_next_depot')
                if 'corridor_track' in p._ST:
                    p._ST['corridor_track'] = [item for item in p._ST['corridor_track']
                                              if proposed.get(item[1]) == actual.get(item[1])]
        step(state, env, actions)
        assert [s[0] for s in state[0].observation.players] == expected_cash
    assert [s.reward for s in state] == row['scores']
    return cfg, visible, snap, mid_hash, memory


def run_arm(own, foe, side, mem, snap, cfg, visible, salt):
    from zerosum_env.envs.carbon import idgen
    policies = [own, foe] if side == 0 else [foe, own]
    for p, pair in zip((own, foe), mem):
        p._ST, p._FC_HISTORY = copy.deepcopy(pair)
        budget(p, True)
    counts = [p._S_STATS['searched'] for p in policies]
    state, env = restore(snap, cfg)
    initial = [pl[0] for pl in state[0].observation.players]
    audit = [Counter(), Counter()]
    marks, trace = {}, []
    for t in range(BRANCH, 299):
        actions = []
        for s, p in enumerate(policies):
            ob = observe(state, s, t)
            a = preserved_decision(p, ob, visible, salt)
            validate_actions(a, ob['players'][s])
            actions.append(a)
        events = []
        with referee_events(events):
            step(state, env, actions)
        trace.append(actions)
        for kind, s, amount, *_ in events:
            if kind in CASH_RULES:
                audit[s][kind] += amount
        end = t + 1
        done = any(s.status != 'ACTIVE' for s in state)
        if end % 25 == 0 or done or end == 299:
            ps = state[0].observation.players
            cash = [p[0] for p in ps]
            assert all(abs(cash[i]-initial[i]-sum(audit[i].values())) < .0001 for i in (0,1))
            marks[str(end)] = dict(cash=cash, our_margin=round(cash[side]-cash[1-side], 2),
                cash_flow=[{k:round(v, 4) for k,v in a.items()} for a in audit],
                workers=[len(p[2]) for p in ps], trees=[len(p[3]) for p in ps])
        if done: break
    scores = [s.reward for s in state]
    assert end == 299 or all(s.status == 'DONE' for s in state)
    return dict(end=end, scores=scores, won=(1 if scores[side] > scores[1-side] else (0 if scores[side] == scores[1-side] else -1)),
                score_margin=round(scores[side]-scores[1-side], 2), milestones=marks, actions_sha256=digest(trace),
                terminal_hash=board_digest(state), id_counters=dict(idgen._id_dict),
                completed_searches_by_seat=[p._S_STATS['searched']-n for p,n in zip(policies,counts)]), trace


def output(path, content, compressed=False):
    raw = (json.dumps(content, ensure_ascii=False, separators=(',',':'))+'\n').encode()
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
    tmp.replace(path)


def main():
    runtime = single_thread()
    configure_engine()
    os.environ['CARBON_DEBUG'] = '1'
    schedules = selected()
    modules = [checkpoint(n, n) for n in ARMS] + [checkpoint('estuary', 'opponent')]
    import torch
    from threadpoolctl import threadpool_info
    assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
    runtime['loaded_pools'] = [{k:p[k] for k in ('internal_api','prefix','num_threads')} for p in threadpool_info()]
    assert all(p['num_threads'] == 1 for p in runtime['loaded_pools'])
    results = dict(runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runtime=runtime, sha256=HASHES, branch_step=BRANCH, selection='Four fixed SHA256(v7-phase-150/seed) per source, alternating historical Estuary seat, independent of outcomes.',
            source_sha256={s:hashlib.sha256((ROOT/'analysis'/f).read_bytes()).hexdigest() for s,f in SOURCES.items()},
            schedule=[(s,r['seed'],i,r['layout_sha256']) for s,i,r in schedules],
            budget=dict(search_each_turn=True, width=4, depth=2, opponent_replies=1, tail=3, wall_clock=False),
            opponent='estuary', cases=[])
    traces = {}
    def save():
        output(OUT / 'phase-switch.json', results)
        output(OUT / 'phase-switch-actions.json.gz', traces, True)
    save()
    for split, side, row in schedules:
        salt = int(hashlib.sha256(('phase-common/'+row['layout_sha256']).encode()).hexdigest()[:8],16) % 100000
        cfg, visible, snap, mid, memory = prepare(row, side, modules, salt)
        initial = [p[0] for p in snap['seats'][0]['observation']['players']]
        case = dict(split=split, seed=row['seed'], side=side, salt=salt, initial=initial,
            initial_margin=round(initial[side]-initial[1-side],2), initial_board_sha256=mid,
            initial_id_counters=snap['id_counters'], historical_names=[p['name'] for p in row['players']],
            historical_scores=row['scores'], layout=row['layout_sha256'], arms={})
        results['cases'].append(case)
        key = f"{row['seed']}/{side}"
        print('verified original', key, flush=True)
        traces[key] = {}
        for i,name in enumerate(ARMS):
            branch, actions = run_arm(modules[i],modules[-1],side,(memory[i],memory[-1]),snap,cfg,visible,salt)
            case['arms'][name],traces[key][name] = branch,actions
            print(split,row['seed'],side,name,branch['score_margin'],branch['completed_searches_by_seat'],flush=True)
            save()
    print('verified',len(results['cases']), 'complete triplets', flush=True)


if __name__ == '__main__':
    main()
