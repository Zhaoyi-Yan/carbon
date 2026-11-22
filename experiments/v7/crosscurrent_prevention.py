"""Six independent games from the public initial board against a reactive Master7sE behavior clone."""
import argparse
import gzip
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import crosscurrent_rescue as cross
import master_e_branch as rescue

SEEDS = (29033,71069)
ARMS = ('v6','harbormaster','crosscurrent')
PURSUIT = 'crosscurrent_pursuit'


def liquidity_spells(series):
    """Inclusive board observations 30..250, total carried stock, own cash strictly below 30."""
    spans = []
    for turn, cash, cargo in series:
        if not (30 <= turn <= 250 and cash < 30 and cargo >= 90):
            continue
        if spans and spans[-1][1] + 1 == turn:
            spans[-1][1] = turn
        else:
            spans.append([turn, turn])
    return {'intervals':spans, 'longest_turns':max((b-a+1 for a,b in spans),default=0),
            'total_turns':sum(b-a+1 for a,b in spans)}


def run(arm, seed, dual):
    rescue.limit_math_threads()
    rescue.configure_engine()
    os.environ['CARBON_DEBUG'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if arm in (cross.NAME,PURSUIT):
        m = cross.load_dual(dual)
        m._CC_WARM = False
    else:
        path = rescue.ROOT/'yan-v6.py' if arm == 'v6' else rescue.HERE/'checkpoints/harbormaster.py'
        m = rescue.load_ours(arm, path)
    clone = rescue.Clone()
    raw = json.loads(gzip.decompress((rescue.OUT/'branch-breakout-replays.json.gz').read_bytes()))['loss']
    assert raw['match_id'] == rescue.MATCH
    cfg, opening, expected = raw['configuration'], raw['initial'], raw['expected'][0]
    del raw
    state,_ = rescue.restore(opening,cfg)
    assert rescue.board_digest(state) == expected
    # Starting from zero: neither strategy nor the judge receives replay actions or subsequent observations.
    snap = opening,clone.snap(),{arm:(dict(),dict())}
    states = []
    steps = []
    actual_step = rescue.step
    def audit(state,env,actions):
        actual_step(state,env,actions)
        turn = len(steps)+1
        steps.append(turn)
        pl = state[0].observation.players
        states.append([rescue.readings(pl,s) for s in (0,1)])
    rescue.step = audit
    route = []
    if arm in (cross.NAME,PURSUIT):
        choose = m._v7_choose_team
        def traced(obs,state):
            team = 'pursuit' if arm == PURSUIT else choose(obs,state)
            route.append((obs['step'],team))
            return team
        m._v7_choose_team = traced
    case = rescue.branch(cfg,snap,clone,m,arm,0,299,seed)
    through = case['through']
    assert len(states) == through
    if arm in (cross.NAME,PURSUIT):
        cross.check_budget(m._CC_BUDGET, seed)
        assert len(route) == through
        assert all(dict(route)[t] == team for team,t,*_ in m._CC_BUDGET)
        case['route_intervals'] = cross.intervals(route)
        case['route_counts'] = dict(Counter(x for _,x in route))
    seq = [[i,s[1]['cash'],s[1]['cargo']] for i,s in enumerate(states,1)]
    case['liquidity'] = liquidity_spells(seq)
    case['step_164'] = states[163] if through >= 164 else None
    case['self_cash_cargo_30_250'] = [p for p in seq if 30 <= p[0] <= 250]
    final = states[-1]
    assert case['final_cash_margin'] == round(final[1]['cash']-final[0]['cash'],2)
    case['final_seats'] = final
    case['finish'] = 'episode horizon' if through == 299 else 'early referee termination'
    return case


def real_game():
    with gzip.open(rescue.OUT/'starvation-ledger.jsonl.gz','rt') as inp:
        real=[json.loads(line) for line in inp]
    ledger=json.loads((rescue.OUT/'starvation.json').read_text())
    assert ledger['match']==rescue.MATCH and real[-1]['t']==299
    series=[[c['t'],c['QQ']['cash'],c['QQ']['cargo']] for c in real]
    event=next(v for v in ledger['events']['deposits'] if v[1]==v[4]==1)
    return {'match':rescue.MATCH,'source':'observed public match against original Master7sE',
            'first_own_deposit':{'t':event[0],'amount':event[3],'worker':event[2]},
            'liquidity':liquidity_spells(series),'step_164':real[164],
            'end_margin':round(real[-1]['QQ']['cash']-real[-1]['Master']['cash'],2)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--out',type=Path,default=rescue.OUT/'crosscurrent-master-e-prevention.json')
    parser.add_argument('--arm',choices=(*ARMS,PURSUIT))
    parser.add_argument('--ablation-only',action='store_true')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    if args.arm:
        case=run(args.arm,args.seed,args.candidate)
        path=args.work/f'{args.seed}-{args.arm}.json'
        path.write_text(json.dumps(case)+'\n')
        print(args.seed,args.arm,'first',case['first'].get('own_seat1'),'longest',case['liquidity']['longest_turns'],
              'T164',case['step_164'],'end',case['final_cash_margin'],flush=True)
        return
    rows=[]
    for seed in SEEDS:
        for arm in ((PURSUIT,) if args.ablation_only else ARMS):
            # One fresh interpreter per cell, run serially under the driver's inherited nice setting.
            subprocess.run([sys.executable,'-u',__file__,'--candidate',str(args.candidate),'--work',str(args.work),
                            '--arm',arm,'--seed',str(seed)],check=True)
            rows.append(json.loads((args.work/f'{seed}-{arm}.json').read_text()))
    if args.ablation_only:
        parent_file=rescue.OUT/'crosscurrent-master-e-prevention.json'
        original=json.loads(parent_file.read_text())
        comparison=[]
        for fixed in rows:
            auto=next(c for c in original['cases'] if c['arm']==cross.NAME and c['rng_seed']==fixed['rng_seed'])
            first=next((a for a,b,team in auto['route_intervals'] if team=='reserve'),None)
            through=auto['through'] if first is None else first
            def pre(case):
                return ([r for r in case['self_cash_cargo_30_250'] if r[0]<=through],
                        [r for r in case['deposits'] if r[0]<=through],
                        {t:v for t,v in case['milestones'].items() if int(t)<=through})
            assert pre(auto)==pre(fixed) and len(fixed['route_intervals'])==1 and fixed['route_intervals'][0][2]=='pursuit'
            if first is None:
                assert fixed['action_sha256']==auto['action_sha256']
            comparison.append({'seed':fixed['rng_seed'],'first_reserve_decision':first,
                               'financial_prefix_identical_through':through,'auto_margin':auto['final_cash_margin'],
                               'forced_margin':fixed['final_cash_margin'],
                               'switch_margin_difference':round(auto['final_cash_margin']-fixed['final_cash_margin'],2),
                               'auto_longest_drought':auto['liquidity']['longest_turns'],
                               'forced_longest_drought':fixed['liquidity']['longest_turns']})
        result={'candidate_sha256':cross.ARTIFACT,'starting_source':str(parent_file.name),
                'starting_result_sha256':rescue.shot(parent_file),'map':original['source_map_sha256'],
                'seats':original['seats'],'seeds':SEEDS,'cases':rows,'comparison':comparison}
        args.out.write_text(json.dumps(result,indent=1)+'\n')
        print('wrote',args.out,flush=True)
        return
    prior=json.loads((rescue.OUT/'master-e-rescue.json').read_text())
    out={'match':rescue.MATCH,'official_replay_seed':prior['official_replay_seed'],
         'official_result_seed':prior['official_result_seed'],'source_map_sha256':json.loads((rescue.OUT/'starvation.json').read_text())['official_map_sha256'],
         'start':0,'seats':['Master7sE behavior clone','QQ arm'],'seeds':SEEDS,
         'hashes':dict(rescue.FROZEN,crosscurrent=cross.ARTIFACT),'clone_weights':prior['clone_weights'],
         'real_game':real_game(),'cases':rows,
         'limits':'One fixed map and seat; the two predeclared seed pairs share the initial board, not later states. Agent inputs exclude future actions, future snapshots and full_carbon. Cash margins come from the actual referee at the episode horizon or early termination. Proxy calibration is in master-e-rescue.json.'}
    args.out.write_text(json.dumps(out,indent=1)+'\n')
    print('wrote',args.out,flush=True)


if __name__=='__main__':
    main()
