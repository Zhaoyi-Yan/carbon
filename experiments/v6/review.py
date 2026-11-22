"""Review the frozen public yan-v5 games. Uses the standard library only."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/v6/online-source.json'
STEPS = (10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 299)
KEYS = ('cash', 'absorption', 'base_receipts', 'cargo', 'workers', 'trees',
        'plants', 'seizes', 'own_tree_income', 'stolen_tree_income', 'tree_turns', 'planter_turns')
MOVES = {'UP': (-1, 0), 'RIGHT': (0, 1), 'DOWN': (1, 0), 'LEFT': (0, -1)}


def destination(cell, action, size):
    dr, dc = MOVES.get(action, (0, 0))
    row, col = divmod(cell, size)
    return (row+dr) % size * size + (col+dc) % size


def base_receipts(prev, frame, size):
    # With zero smelt/move cost, all arriving cargo goes to the base owner,
    # even when a worker loses a collision on the base or belongs to the enemy.
    owners = {p: j for j, player in enumerate(prev) for p in player[1].values()}
    received = [0., 0.]
    for j, player in enumerate(prev):
        actions = frame[j]['action'] or {}
        for wid, (cell, cargo, _) in player[2].items():
            if cargo:
                target = destination(cell, actions.get(wid), size)
                if target in owners:
                    received[owners[target]] += cargo
    return received


def elimination(history, last_players, seat, threshold):
    last = history[-1]
    assert last['cash'] < threshold and last['trees'] == last['workers'] == 0
    def trailing_start(key, cutoff):
        return 1 + max((t for t, p in enumerate(history) if p[key] >= cutoff), default=-1)
    cash_start = trailing_start('cash', threshold)
    tree_start = trailing_start('trees', 1)
    largest = max(range(cash_start, len(history)), key=lambda t: history[t]['cargo'])
    deposits = [t for t in range(1,len(history))
                if history[t]['base_receipts'] - history[t-1]['base_receipts'] >= threshold]
    before = last_players[seat]
    return dict(last_step=len(history)-1, last_cash=last['cash'], final_unaffordable_from=cash_start,
                final_treeless_from=tree_start, last_single_turn_receipts_at_least_hire=deposits[-1] if deposits else None,
                peak_cargo_while_finally_unaffordable_step=largest,
                peak_cargo_while_finally_unaffordable=round(history[largest]['cargo'],2),
                penultimate_workers=[dict(role=w[2], cell=w[0], cargo=round(w[1],2)) for w in before[2].values()])


def review_match(meta, path, strategy):
    key = next(k for k, entry in meta['strategies'].items() if entry['id'] == strategy)
    seat = meta['order'].index(key)
    other = meta['order'][1-seat]
    with gzip.open(path, 'rt') as inp:
        game = json.load(inp)
    frames, config = game['steps'], game['configuration']
    size = config['size']
    assert config['smeltCost'] == config['moveCost'] == 0
    assert len(frames)-1 == meta['steps'] and game['statuses'] == ['DONE', 'DONE']
    indexed = [meta['scores'][k] for k in meta['order']]
    assert [row['reward'] for row in frames[-1]] == indexed == game['rewards']
    scores = [indexed[seat], indexed[1-seat]]
    won = scores[0] > scores[1]
    assert (meta['winner'] == key) == won and scores[0] != scores[1]
    entry = dict(match=meta['id'], game=meta['game'], opponent=meta['strategies'][other]['name'],
                 opponent_id=meta['strategies'][other]['id'],
                 seat=seat, seed=config['randomSeed'], turns=len(frames)-1, won=won, scores=scores)
    before = frames[0][0]['observation']
    digest = json.dumps([before['full_carbon'], [sorted(p[1].values()) for p in before['players']]],
                        separators=(',', ':')).encode()
    entry['layout'] = hashlib.sha256(digest).hexdigest()
    persistent = [Counter(), Counter()]
    origin, previous_owner, snaps = {}, {}, {}
    first_planter, first_tree = [None, None], [None, None]
    history = []
    for t, frame in enumerate(frames):
        obs = frame[0]['observation']
        assert obs['step'] == t
        now = obs['players']
        receipts = base_receipts(prev, frame, size) if t else [0., 0.]
        for j, p in enumerate(now):
            metrics = persistent[j]
            metrics['cash'] = p[0]
            metrics['base_receipts'] += receipts[j]
            metrics['workers'] = len(p[2])
            metrics['cargo'] = sum(w[1] for w in p[2].values())
            metrics['trees'] = len(p[3])
            metrics['tree_turns'] += len(p[3])
            planters = sum(w[2]=='PLANTER' for w in p[2].values())
            metrics['planter_turns'] += planters
            if planters and first_planter[j] is None:
                first_planter[j] = t
            for tree in p[3]:
                if tree not in origin:
                    origin[tree] = j
                    metrics['plants'] += 1
                    if first_tree[j] is None:
                        first_tree[j] = t
                elif previous_owner[tree] != j:
                    metrics['seizes'] += 1
                previous_owner[tree] = j
                income = obs['trees'][tree][1]
                metrics['absorption'] += income
                metrics['own_tree_income' if origin[tree] == j else 'stolen_tree_income'] += income
        me, them = persistent[seat], persistent[1-seat]
        history.append(me.copy())
        if t in STEPS:
            snaps[t] = {k: (me[k], them[k], me[k]-them[k]) for k in KEYS}
        prev = now
    entry['first_planter'] = [first_planter[seat], first_planter[1-seat]]
    entry['first_tree'] = [first_tree[seat], first_tree[1-seat]]
    entry['cash_diffs'] = [round(snaps[t]['cash'][2],2) if t in snaps else None for t in (30,50,75,100,150,250)]
    entry['economy_diffs_at_75'] = [round(snaps[75][k][2],2)
                                    for k in ('absorption','base_receipts','plants','seizes','trees')]
    if len(frames) < config['episodeSteps']:
        entry['elimination'] = elimination(history, frames[-2][0]['observation']['players'], seat,
                                            min(config['recCollectorCost'], config['recPlanterCost']))
    return entry, snaps


def summarize(rows):
    out = {}
    for won in (True, False):
        group = [(row,snaps) for row,snaps in rows if row['won'] == won]
        steps = {}
        for t in STEPS:
            sample = [snaps[t] for _,snaps in group if t in snaps]
            steps[t] = dict(n=len(sample), median_margin={
                k: round(median(s[k][2] for s in sample),2) for k in KEYS},
                behind_in_cash=sum(s['cash'][2] < 0 for s in sample),
                behind_in_tree_income=sum(s['absorption'][2] < 0 for s in sample))
        at75 = [s[75] for _,s in group]
        side_medians = {k: [round(median(s[k][i] for s in at75),2) for i in range(2)] for k in KEYS}
        first = {k: [median(row[k][i] for row,_ in group if row[k][i] is not None) for i in range(2)]
                 for k in ('first_planter','first_tree')}
        out['wins' if won else 'losses'] = dict(n=len(group), seat_0=sum(row['seat']==0 for row,_ in group),
                seat_1=sum(row['seat']==1 for row,_ in group), first=first,
                by_step=steps, medians_at_75_ours_opponent=side_medians)
    return out


def repeated_opponents(rows):
    opponents = {}
    for row, snaps in rows:
        group = opponents.setdefault(row['opponent_id'], dict(name=row['opponent'], wins=[], losses=[]))
        group['wins' if row['won'] else 'losses'].append(snaps[75]['absorption'][2])
    results = {}
    for key, group in opponents.items():
        if group['wins'] and group['losses']:
            results[key] = dict(name=group['name'], **{outcome: dict(n=len(group[outcome]),
                            median_tree_income_margin_at_75=round(median(group[outcome]),2))
                            for outcome in ('wins','losses')})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Frozen index.json and replays directory.')
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/v6-online.json')
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    strategy = source['leaderboard']['id']
    index = {(m['id'],m['game']): m for m in json.loads((args.data/'index.json').read_text())}
    results = []
    for ref in source['matches']:
        meta = index[ref['id'],ref['game']]
        assert ref['file']==meta['file']
        results.append(review_match(meta, args.data/ref['file'], strategy))
    report = dict(source='https://github.com/JUZI1/carbon-league-replays/tree/'+source['revision'],
                  strategy_id=strategy, n=len(results), unique_layouts=len({r['layout'] for r,_ in results}),
                  normal_full_length=sum(r['turns']==299 for r,_ in results),
                  metric_notes={
                      'margin':'yan-v5 minus its opponent, within each game. Each median uses only games alive at the step.',
                      'absorption':'Sum of per-tree ordinary cash income by the current owner, including cell absorption and cargo tax. Does not count rare collision payouts.',
                      'base_receipts':'All cargo delivered to our bases (including collisions and enemy deliveries), before tree taxes, with zero move and smelt costs.',
                      'split':'own_tree_income and stolen_tree_income split absorption by whether the present owner planted the tree.',
                      'seed':'Official replay configuration randomSeed. Layout hash is SHA-256 of compact JSON [initial full_carbon, sorted base cells by seat].',
                      'elimination':'Early official rewards differ from final on-board cash. No workers, no trees and cash below one hire ends the game.',
                      'per_game_cash_diffs_steps':[30,50,75,100,150,250],
                      'per_game_economy_diffs_75':['absorption','base_receipts','plants','seizes','trees']},
                  observations=[
                      '胜局及败局在第30步的现金差中位数分别为 −6.27 和 −10.36，主要分化出现在第40–75步。第75步两组的树吸收收入差中位数约为 −161 和 −738，基地到款差约为 +203 和 +29。',
                      '第75步，败局中双方累计新种数中位分别为 5 和 10，胜局为 6 和 8。败局第一个种植员和首棵树出现的中位时间分别为第7、14步，胜局为第8、15步。',
                      '按同一对手版本对照，在六个既有胜局又有败局的对手中，五个对手的败局在第75步的树收入差中位更低，小肥鱼v5-it720呈相反方向。每组只有1至3局，比赛地图不同。',
                      '两场提前淘汰的现金在第47步或第59步以后一直低于招募价，且在途载货峰值达到571和1179。最后一名工人死亡之前已多轮丧失补员能力。',
                  ],
                  proposed_test='保留原策略候选，在第30–75步按已观察到的敌树收入和树龄为首轮树林补种、夺树或协防提供备选；用新图双席分别记录树收入、基地到款、招募能力和胜负。先把长期现金不足且重载未回的极端局单列，而非把所有败局归为同一种错误。',
                  outcome_groups=summarize(results), same_opponent_win_loss=repeated_opponents(results),
                  matches=[r for r,_ in results])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1)+'\n')


if __name__ == '__main__':
    main()
