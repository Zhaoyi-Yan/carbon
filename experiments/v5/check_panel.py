"""Check mirrored, completed full-length games against the registered opponent and maps."""
import argparse
import json
from pathlib import Path
from statistics import mean


def review(games, seeds, candidate, opponent):
    by_game={}
    for game in games:
        order=[p['sha256'] for p in game['players']]
        if sorted(order)!=sorted((candidate,opponent)):
            raise ValueError('A different strategy binary was used')
        seat=order.index(candidate)
        key=(game['seed'],seat)
        if game['seed'] not in seeds or key in by_game:
            raise ValueError('Unexpected seed or duplicate seat')
        by_game[key]=game
    rows=[]
    engines={g['engine_sha256'] for g in games}
    layouts=set()
    if len(engines)!=1:raise ValueError('Mixed engine revisions')
    for seed in seeds:
        pair=[by_game.get((seed,seat)) for seat in (0,1)]
        if not all(pair):raise ValueError(f'Missing a seat for {seed}')
        if pair[0]['layout_sha256']!=pair[1]['layout_sha256'] or pair[0]['layout_sha256'] in layouts:
            raise ValueError('Unpaired or repeated layout')
        layouts.add(pair[0]['layout_sha256'])
        for seat,g in enumerate(pair):
            completed=(g['status']=='completed' and g['episode_steps']==300 and g['turns']==299
                       and all(not p['failure'] and p['calls']==299 for p in g['players']))
            margin=round(g['cash'][seat]-g['cash'][1-seat],2) if completed else None
            won=completed and margin>0 and g['winner']==seat
            rows.append(dict(seed=seed,seat=seat,completed=completed,won=won,margin=margin,
                             max_act_seconds=g['players'][seat]['max_act_seconds'],
                             slow_steps=g['players'][seat]['slow_steps']))
    margins=[r['margin'] for r in rows if r['completed']]
    return dict(candidate=candidate,opponent=opponent,passed=all(r['won'] for r in rows),
                wins=sum(r['won'] for r in rows),completed=len(margins),games=len(rows),
                mean_margin=round(mean(margins),2) if margins else None,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('record',type=Path);p.add_argument('candidate_sha')
    p.add_argument('--panel',choices=('qualification','confirmation'),default='qualification')
    a=p.parse_args();here=Path(__file__).parent
    design=json.loads((here/'design.json').read_text())
    seeds=design['qualification'] if a.panel=='qualification' else json.loads((here/'confirmation.json').read_text())['seeds']
    games=[json.loads(row) for row in a.record.read_text().splitlines()]
    result=review(games,seeds,a.candidate_sha,design['opponent_sha256'])
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
