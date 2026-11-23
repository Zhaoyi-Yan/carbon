"""Verify all recorded outcomes, then reconstruct the paired cash ledgers locally."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/v7')]
from arena import engine_fingerprint
from benchmark import configure_engine, new_game
from starvation_replay import cash_flows


def audit(row, contract, checkpoints=(), *, candidate=None):
    from zerosum_env.envs.carbon import carbon
    players = row['players']
    seat = (next(i for i,p in enumerate(players) if p['name'] != 'yan-v7') if candidate is None
            else next(i for i,p in enumerate(players) if p['name']==candidate))
    result = dict(arm=players[seat]['name'], seed=row['seed'], seat=seat, layout=row['layout_sha256'],
                  status=row['status'], turns=row['turns'], players=players)
    if row['engine_sha256'] != engine_fingerprint(carbon):
        raise ValueError('Different referee')
    for p in players:
        assert p['sha256'] == contract['sha256'][p['name']]
    if row['status'] != 'completed':
        return result
    state, env = new_game(row['seed'], row['episode_steps'])
    board = state[0].observation
    layout = [env.configuration.size, board.full_carbon, [sorted(p[1].values()) for p in board.players]]
    assert hashlib.sha256(json.dumps(layout, separators=(',',':')).encode()).hexdigest() == result['layout']
    frames = [[dict(observation=state[0].observation), {}]]
    origin = {}
    ledgers = [[Counter(),Counter()] for _ in range(3)]
    opening = [dict(first=None, by_port=Counter(), deaths=0) for _ in range(2)]
    snapshots = {}
    for t,(actions,expected) in enumerate(row['actions_and_post_cash'], 1):
        for s,moves in zip(state,actions):
            s.action = moves
        carbon.compute_next_board(state,env)
        board = state[0].observation
        assert [p[0] for p in board.players] == expected
        after = [dict(observation=board,action=actions[0]), dict(action=actions[1])]
        flows, deposits, _, _ = cash_flows(frames[0], after, env.configuration, origin)
        if checkpoints and t<=max(checkpoints):
            for i, f in enumerate(flows):
                opening[i]['deaths'] += f['deaths']
            for _, origin_side, _, amount, receiver, port in deposits:
                if origin_side==receiver and amount>0:
                    ledger_entry=opening[receiver]
                    if ledger_entry['first'] is None:
                        ledger_entry['first']=t
                    ledger_entry['by_port'][port]+=amount
            if t in checkpoints:
                snapshots[t]=[dict(cash=expected[i],first=opening[i]['first'],deaths=opening[i]['deaths'],
                                   by_port=dict(opening[i]['by_port'])) for i in (seat,1-seat)]
        frames[0] = after
        ledger = ledgers[0 if t<=190 else 1 if t<=250 else 2]
        for i,f in enumerate(flows):
            ledger[i]['tree_net'] += f['tree'] + f['collision_tree'] - f['plant_cost'] - f['seize_cost']
            ledger[i]['base'] += f['base']
            ledger[i]['hire_cost'] += f['hire_cost']
            ledger[i]['plants'] += f['plants']
            ledger[i]['seizes'] += f['seizes']
            ledger[i]['own_receipts'] += f['base_own']
            ledger[i]['deliveries'] += sum(x[1] == x[4] == i for x in deposits)
    assert [s.reward for s in state] == row['scores']
    result.update(margin=round(row['scores'][seat]-row['scores'][1-seat],2), scores=row['scores'],
                  phases=[[[k,round(a[k],3),round(b[k],3)] for k in a.keys()|b.keys()]
                          for a,b in [(pair[seat],pair[1-seat]) for pair in ledgers]])
    if checkpoints:
        result['opening']=snapshots
    return result


def summarize(records):
    valid = [r for r in records if 'margin' in r]
    if not valid:
        return dict(valid=0,total=len(records))
    by_seat = [[r['margin'] for r in valid if r['seat']==s] for s in range(2)]
    pairs = defaultdict(dict)
    for r in valid:
        assert r['seat'] not in pairs[r['seed']]
        pairs[r['seed']][r['seat']] = r['margin']
    def cash(phases):
        own,other = Counter(),Counter()
        for r in valid:
            for phase in phases:
                for k,a,b in r['phases'][phase]:
                    own[k]+=a
                    other[k]+=b
        delta={k:round((own[k]-other[k])/len(valid),2) for k in sorted(own)}
        sizes=[round(x['own_receipts']/x['deliveries'],2) if x['deliveries'] else None for x in (own,other)]
        return dict(delta=delta,average_delivery=sizes)
    return dict(valid=len(valid),total=len(records),wins_by_seat=[sum(x>0 for x in v) for v in by_seat],
                wins=sum(r['margin']>0 for r in valid), draws=sum(r['margin']==0 for r in valid),
                average_margin=round(mean(r['margin'] for r in valid),2),
                margins=pairs, all_turns=cash(range(3)), final_49=cash([2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    configure_engine()
    protocol=json.loads((args.run/'protocol.json').read_text())
    report=dict(protocol=protocol, stages={})
    layouts={}
    for stage in ('screening','confirmation'):
        path=args.run/(stage+'.jsonl')
        if not path.exists():
            continue
        by_arm=defaultdict(list)
        for line in path.read_text().splitlines():
            r=audit(json.loads(line),protocol)
            assert layouts.setdefault(r['seed'],r['layout']) == r['layout']
            by_arm[r['arm']].append(r)
        arms=('trees','cargo','combined') if stage=='screening' else ('combined',)
        assert set(by_arm)==set(arms)
        for rows in by_arm.values():
            assert sorted((r['seed'],r['seat']) for r in rows)==sorted((n,s) for n in protocol[stage] for s in (0,1))
        report['stages'][stage]={a:dict(summary=summarize(by_arm[a]),games=sorted(by_arm[a],key=lambda r:(r['seed'],r['seat']))) for a in arms}
    assert len(layouts)==len(set(layouts.values()))
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
