"""Evaluate Crosscurrent on the frozen Master7sE branching test with both nested search engines budgeted."""
import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import master_e_branch as rescue

ARTIFACT = '5bcabdb9daa0b8f89b461d118edec68d90c1d459f2ec75ccea80a0b2eb7b091a'
NAME = 'crosscurrent'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one occurrence of {old!r}, found {source.count(old)}')
    return source.replace(old, new)


def patch_engine(text, nested):
    prefix = '    ' if nested else ''
    team = 'reserve' if nested else 'pursuit'
    # Match within one strategy block. Each injected global is private to the experiment harness.
    start = text.index(f'{prefix}    if _S_COST[0] > 0.0:', text.index(f'{prefix}def _search('))
    end = text.index(f'{prefix}    stages =', start)
    text = (text[:start] + f'{prefix}    most,D2,M=4,min(D2,2),1\n'
            + f'{prefix}    _CC_BUDGET.append(("{team}",step,most,D2,M,_S_MMAX,_time.perf_counter(),_BRANCH_SEED))\n' + text[end:])
    changes = [
        ('+ 17 * me) % (2 ** 31)', '+ 17 * me + _BRANCH_SEED) % (2 ** 31)'),
        ('for tick in range(1,15):', 'for tick in range(1,4):'),
        ('_torch.set_num_threads(2)', '_torch.set_num_threads(1)'),
        ('return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT', 'return False'),
        ('_S_MMAX = 2 ', '_S_MMAX = 1 '),
        ('import time as _time', '_time = _CC_CLOCK'),
        ('def _reply_choice(obs,config,est,act,logits,locked,t0):\n',
         f'def _reply_choice(obs,config,est,act,logits,locked,t0):\n{prefix}    if _CC_WARM: return act\n'),
        ('def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):\n',
         f'def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):\n{prefix}    if _CC_WARM: return act0\n'),
    ]
    for old, new in changes:
        text = once(text, old, new)
    return text


def load_dual(path):
    if rescue.shot(path) != ARTIFACT:
        raise ValueError('Crosscurrent file differs from predeclared SHA')
    left, middle, right = path.read_text().partition('def _v7_make_reserve(shared_shadow):')
    if not middle:
        raise ValueError('No nested reserve strategy')
    prelude = ('from types import SimpleNamespace as _CC_NS\n'
               '_CC_CLOCK=_CC_NS(perf_counter=lambda:0.)\n'
               '_CC_BUDGET=[]\n_CC_WARM=True\n_BRANCH_SEED=0\n')
    code = prelude + patch_engine(left, False) + middle + patch_engine(right, True)
    m = rescue.load_strategy('crosscurrent_fixed_budget', path, code)
    assert m._SEARCH_OK and m._RX and m._torch.get_num_threads() == 1
    return m


def check_budget(rows, seed):
    assert rows and all(width == 4 and depth in (1, 2) and replies == 1 and ceiling == 1
                        and clock == 0. and rng == seed
                        for _, _, width, depth, replies, ceiling, clock, rng in rows)


def intervals(rows):
    result = []
    for step, team in rows:
        if result and result[-1][2] == team and result[-1][1] == step - 1:
            result[-1][1] = step
        else:
            result.append([step, step, team])
    return result


def check_both(m, cfg, snap, seed):
    select = m._v7_choose_team
    out = {}
    for which in ('reserve','pursuit'):
        state, _ = rescue.restore(snap[0], cfg)
        m._ST, m._FC_HISTORY = copy.deepcopy(snap[2][NAME])
        m._BRANCH_SEED = seed
        m._ST['rng'] = rescue.np.random.RandomState(seed + 1000003 * 165 + 1)
        m._v7_choose_team = lambda obs, state: which
        m._CC_BUDGET.clear()
        obs = rescue.observe(state, 1, 164)
        action = m.agent(obs, {k:v for k,v in cfg.items() if k != 'randomSeed'})
        rescue.validate_actions(action, obs['players'][1])
        check_budget(m._CC_BUDGET, seed)
        assert m._CC_BUDGET[0][:2] == (which, 164)
        out[which] = {'search_record': m._CC_BUDGET[0], 'action': action}
    m._v7_choose_team = select
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=rescue.OUT/'crosscurrent-master-e-rescue.json')
    parser.add_argument('--progress', type=Path, required=True)
    args = parser.parse_args()
    rescue.limit_math_threads()
    rescue.configure_engine()
    rescue.os.environ['CARBON_DEBUG'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    m = load_dual(args.candidate)
    clone = rescue.Clone()
    parent_path = rescue.OUT/'master-e-rescue.json'
    prior = json.loads(parent_path.read_text())
    assert prior['match'] == rescue.MATCH and prior['frozen_sha256'] == rescue.FROZEN
    assert prior['clone_weights'] == {k:rescue.shot(rescue.HERE/n) for k,n in
        [('worker','master-worker-model.npz'),('recruit','master-model.npz'),('trunk','checkpoints/fairlead.py')]}
    data = json.loads(gzip.decompress((rescue.OUT/'branch-breakout-replays.json.gz').read_bytes()))['loss']
    cfg = data['configuration']
    data['actions'] = data['actions'][:164]
    data['expected'] = data['expected'][:165]
    snaps = rescue.history(data, {NAME:m}, clone)
    del data
    prefix = {t:{k:snap[2][NAME][0].get(k) for k in
              ('liquidity_step','last_delivery','dual_team','dual_switch','corridor_track')} for t,snap in snaps.items()}
    m._CC_WARM = False
    smoke = check_both(m, cfg, snaps[164], rescue.SEEDS[0])
    print('prefix / forced single-turn budget audit',prefix,smoke,flush=True)
    choose = m._v7_choose_team
    route = []
    def traced(obs, state):
        which = choose(obs, state)
        route.append((obs['step'],which))
        return which
    m._v7_choose_team = traced
    rows = []
    for start,end in rescue.START_END:
        for seed in rescue.SEEDS:
            route.clear()
            m._CC_BUDGET.clear()
            row = rescue.branch(cfg,snaps[start],clone,m,NAME,start,end,seed)
            assert len(route) == row['through']-start and len({t for t,_ in route}) == len(route)
            check_budget(m._CC_BUDGET, seed)
            assert all(dict(route)[t] == team for team,t,*_ in m._CC_BUDGET)
            row['route_intervals'] = intervals(route)
            row['choice_counts'] = dict(Counter(team for _,team in route))
            row['search_counts'] = dict(Counter(team for team,*_ in m._CC_BUDGET))
            rows.append(row)
            args.progress.write_text(json.dumps(rows))
            print(start,seed,row['first'].get('own_seat1'),row['final_cash_margin'],row['choice_counts'],flush=True)
    comparison = []
    for row in rows:
        pair = [c for c in prior['cases'] if c['start'] == row['start'] and c['rng_seed'] == row['rng_seed']]
        assert len(pair) == 2
        fact = {'start':row['start'],'seed':row['rng_seed']}
        for peer in [*pair,row]:
            snap = peer['milestones'].get(str(row['start']+25)) or peer['milestones'][row['start']+25]
            a = snap['seats'][1]
            fact[peer['arm']] = {'first_own_deposit':peer['first'].get('own_seat1'), 'own_deposits_25':snap['gross_by_recipient_source'][1][1],
                                 'workers_25':a['workers'], 'cash_25':a['cash'], 'trees_25':a['trees'], 'end_margin':peer['final_cash_margin'],
                                 'same_action_hash_as_crosscurrent':peer['action_sha256'] == row['action_sha256']}
        comparison.append(fact)
    result={'match':rescue.MATCH,'official_result_seed':prior['official_result_seed'],'official_replay_seed':cfg['randomSeed'],
            'seats':['Master7sE behavior clone','Crosscurrent'],'candidate_sha256':ARTIFACT,'baseline_sha256':rescue.shot(parent_path),
            'clone_weights':prior['clone_weights'],'source_prefix_state':prefix,'smoke':smoke,
            'search_record_columns':['team','step','width cap','depth','sampled replies','max rollouts','clock','branch seed'],
            'independent_clone_calibration':'master-e-rescue.json','cases':rows,'comparison':comparison}
    args.out.write_text(json.dumps(result,indent=1)+'\n')
    print('wrote',args.out,flush=True)


if __name__ == '__main__':
    main()
