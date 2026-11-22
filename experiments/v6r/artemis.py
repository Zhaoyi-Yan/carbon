"""Summarize frozen Artemis, Apollo and yan-v5 public game files by opponent and seat."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('online_review', ROOT/'experiments/v6/review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

REVISION = '1f456cfdb568010b5ff8c276ed74b39d609f115b'
NAMES = {'226891089ba7f54b198c8f8dc771dfc2':'Artemis', 'dd1f117a64c676225f0c6333b1cdd1e7':'Apollo',
         'ec91ff24ea3af3510fe28c948b5d7da3':'yan-v5'}
METRICS = ('first_planter','first_tree','new_trees75','seizes75','tree_income75','base_receipts75',
           'trees75','collector_turns75','planter_turns75','empty_pressure75','cash_margin75','score_margin')


def torus(a, b, size):
    dy, dx = abs(a//size-b//size), abs(a%size-b%size)
    return min(dy,size-dy)+min(dx,size-dx)


def measure(meta, data):
    file_sha = hashlib.sha256((data/meta['file']).read_bytes()).hexdigest()
    with gzip.open(data/meta['file']) as stream:
        g = json.load(stream)
    frames = g['steps']
    assert len(frames)-1 == meta['steps']
    assert g['rewards'] == [meta['scores'][k] for k in meta['order']]
    assert g['statuses'] == ['DONE', 'DONE']
    size = g['configuration']['size']
    assert g['configuration']['smeltCost'] == g['configuration']['moveCost'] == 0
    initial = frames[0][0]['observation']
    layout = json.dumps([initial['full_carbon'], [sorted(p[1].values()) for p in initial['players']]],
                        separators=(',', ':')).encode()
    # Both axes are mirrored by the referee at initial generation; used only for this audit.
    carbon0 = initial['full_carbon']
    assert all(v == carbon0[(size-1-i//size)*size+i%size] == carbon0[i//size*size+size-1-i%size]
               for i,v in enumerate(carbon0))
    sides = [[] for _ in range(2)]
    for side, slot in enumerate(meta['order']):
        entry = meta['strategies'][slot]
        if entry['id'] not in NAMES:
            continue
        opp = meta['strategies'][meta['order'][1-side]]
        score = g['rewards'][side] - g['rewards'][1-side]
        row = dict(name=NAMES[entry['id']],match=meta['id'],game=meta['game'],seat=side,
                   opponent=opp['name'],opponent_id=opp['id'],revision=meta['revision'],turns=len(frames)-1,
                   seed=g['configuration']['randomSeed'],file_sha256=file_sha,layout=hashlib.sha256(layout).hexdigest(),
                   won=score>0,score_margin=round(score,2))
        actions = frames[1][side]['action'] or {}
        first = [actions[k] for k in initial['players'][side][1] if k in actions]
        row['opening'] = '+'.join(sorted(first)) or 'WAIT'
        opening_stats = dict(roster=[], hire_sequence=[], first_hire_step=None, first_hire=None,
                             first_opponent_hire_step=None, first_bank=None, first_bank_turn=None,
                             first_bank_from_own_carriers=None, step1_status=frames[1][side]['status'],
                             overage_before=frames[0][side]['observation']['remainingOverageTime'],
                             overage_after=frames[1][side]['observation']['remainingOverageTime'])
        ids = [p[1] for p in initial['players']]
        for turn, frame in enumerate(frames):
            players = frame[0]['observation']['players']
            if turn <= 12:
                opening_stats['roster'].append([sum(w[2]==role for w in players[side][2].values())
                                                for role in ('COLLECTOR','PLANTER')])
            for j in (side,1-side):
                a = frame[j]['action'] or {}
                roles = sorted(a[b][3:] for b in ids[j] if b in a)
                if not roles:
                    continue
                key = 'first_hire_step' if j==side else 'first_opponent_hire_step'
                if opening_stats[key] is None:
                    opening_stats[key] = turn
                    # Initial hires should appear as employed workers in that same replay frame.
                    if j == side:
                        assert not frames[turn-1][0]['observation']['players'][side][2]
                        assert sorted(w[2] for w in players[side][2].values()) == roles
                        opening_stats['first_hire'] = '+'.join(roles)
                if j == side and turn <= 12:
                    opening_stats['hire_sequence'].append([turn,'+'.join(roles)])
            if turn and opening_stats['first_bank'] is None:
                prior = frames[turn-1][0]['observation']['players']
                paid = review.base_receipts(prior,frame,size)[side]
                if paid>0:
                    ours = 0.
                    own_homes = prior[side][1].values()
                    for wid,(pos,cargo,_) in prior[side][2].items():
                        a = (frame[side]['action'] or {}).get(wid)
                        if review.destination(pos,a,size) in own_homes:
                            ours += cargo
                    opening_stats.update(first_bank=round(paid,4),first_bank_turn=turn,
                                         first_bank_from_own_carriers=round(ours,4))
            if turn>=12 and all(opening_stats[k] is not None for k in ('first_bank','first_hire_step','first_opponent_hire_step')):
                break
        if row['name'] in ('Artemis','Apollo'):
            hire = opening_stats['first_hire_step']
            assert not any(frames[hire-1][side]['observation']['carbon'])
            worker = next(w for w in frames[hire][0]['observation']['players'][side][2].values() if w[2]=='PLANTER')
            first_p = worker[0]
            mine = initial['players'][side][1]
            enemy = initial['players'][1-side][1]
            other_open = frames[1][1-side]['action']
            other_roles = sorted(other_open[b][3:] for b in enemy if b in other_open)
            foe_p = next((cell for b,cell in enemy.items() if other_open.get(b)=='RECPLANTER'),None)
            tree_frame = next(f for f in frames if any(age==1 for _,age in f[0]['observation']['players'][side][3].values()))
            first_pos = next(pos for pos,age in tree_frame[0]['observation']['players'][side][3].values() if age==1)
            opening_stats.update(planter_base=first_p,first_tree_cell=first_pos,
                                 first_tree_distance_from_spawn=torus(first_pos,first_p,size),
                                 first_tree_distance_to_own=min(torus(first_pos,b,size) for b in mine.values()),
                                 first_tree_distance_to_foe=min(torus(first_pos,b,size) for b in enemy.values()),
                                 first_opponent_planter_base=foe_p,first_opponent_roles='+'.join(other_roles) or 'WAIT')
        row['startup'] = opening_stats
        sides[side] = row
    totals = [Counter(),Counter()]
    previous_owner = {}
    bases = [p[1].values() for p in initial['players']]
    distance = [[min(torus(x,b,size) for b in centers) for x in range(size**2)] for centers in bases]
    for t, frame in enumerate(frames[:76]):
        obs = frame[0]['observation']
        now = obs['players']
        received = review.base_receipts(prev,frame,size) if t else [0.,0.]
        for side, pl in enumerate(now):
            vals = totals[side]
            vals['base_receipts75'] += received[side]
            for tree in pl[3]:
                if tree not in previous_owner:
                    vals['new_trees75'] += 1
                    vals.setdefault('first_tree',t)
                elif previous_owner[tree] != side:
                    vals['seizes75'] += 1
                previous_owner[tree] = side
                vals['tree_income75'] += obs['trees'][tree][1]
            for cell, cargo, role in pl[2].values():
                if role == 'PLANTER':
                    vals.setdefault('first_planter',t)
                    vals['planter_turns75'] += 1
                else:
                    vals['collector_turns75'] += 1
                    vals['empty_pressure75'] += int(cargo<1 and distance[1-side][cell]<=2)
            if t==75 and sides[side]:
                for key in METRICS[:-1]:
                    if key.startswith('first_'):
                        # Missing opening timing is censored after 75 rather than recorded as turn 0.
                        sides[side][key] = vals.get(key)
                    elif key == 'cash_margin75':
                        sides[side][key] = round(pl[0]-now[1-side][0],2)
                    elif key == 'trees75':
                        sides[side][key] = len(pl[3])
                    else:
                        sides[side][key] = round(vals[key],2)
        prev = now
    return [row for row in sides if row]


def stats(rows):
    return dict(games=len(rows),wins=sum(r['won'] for r in rows),seat0=sum(r['seat']==0 for r in rows),
                medians={k:round(median(r[k] for r in rows if r.get(k) is not None),2)
                         for k in METRICS if any(r.get(k) is not None for r in rows)},
                opening=dict(Counter(r['opening'] for r in rows)))


def opening_profile(rows):
    stages = [r['startup'] for r in rows]
    rosters = [[median(r['roster'][t][m] for r in stages) for m in (0,1)] for t in range(13)]
    seq = Counter(tuple((t,action) for t,action in r['hire_sequence']) for r in stages)
    timely = Counter((r['step1_status'],r['overage_before'],r['overage_after']) for r in stages)
    receipts = [r for r in stages if r['first_bank'] is not None]
    hires = Counter((r['first_hire_step'],r['first_hire']) for r in stages)
    return dict(roster_median_steps_0_12=rosters,
                first_successful_hire=[dict(step=t,roles=h,games=n) for (t,h),n in sorted(hires.items())],
                most_common_hire_sequences_through_12=[dict(actions=list(k),games=n) for k,n in seq.most_common(5)],
                first_positive_base_receipt=dict(games=len(receipts),turn_median=median(r['first_bank_turn'] for r in receipts),
                         amount_median=round(median(r['first_bank'] for r in receipts),2),
                         all_from_own_incoming_cargo=sum(r['first_bank']==r['first_bank_from_own_carriers'] for r in receipts)),
                step1_timing=[dict(status=s,overage_start=a,overage_end=b,games=n) for (s,a,b),n in timely.items()])


def geometry_profile(rows):
    by_seat = {}
    for seat in (0,1):
        stages=[r['startup'] for r in rows if r['seat']==seat]
        data=Counter((r['first_opponent_roles'],r['first_opponent_planter_base'],r['planter_base']) for r in stages)
        by_seat[seat]=dict(n=len(stages), own_planter_base=dict(Counter(r['planter_base'] for r in stages)),
                           conditional=[dict(opponent_first_roles=k,opponent_initial_planter_cell=v,own_planter_cell=c,games=n)
                                        for (k,v,c),n in data.items()],
                           first_tree_distance_from_planter_base_median=median(r['first_tree_distance_from_spawn'] for r in stages),
                           first_tree_closer_to_foe=sum(r['first_tree_distance_to_foe']<r['first_tree_distance_to_own'] for r in stages))
    return by_seat


def matched(rows, names, same_seat):
    groups = defaultdict(lambda:defaultdict(list))
    for r in rows:
        key = (r['opponent_id'],r['seat'] if same_seat else None)
        groups[key][r['name']].append(r)
    complete = {k:v for k,v in groups.items() if all(n in v for n in names)}
    panels = []
    for key, group in sorted(complete.items()):
        opp = group[names[0]][0]['opponent']
        panels.append(dict(opponent=opp,opponent_id=key[0],seat=key[1],
                           strategies={n:stats(group[n]) for n in names}))
    avg = {n:{k:round(mean(p['strategies'][n]['medians'][k] for p in panels
                           if all(k in p['strategies'][strategy]['medians'] for strategy in names)),2)
              for k in METRICS} for n in names}
    return dict(strata=len(panels), strategy_counts={n:sum(p['strategies'][n]['games'] for p in panels) for n in names},
                equal_weight_mean_of_stratum_medians=avg, groups=panels)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/v6r-artemis.json')
    parser.add_argument('--games',type=Path,default=ROOT/'analysis/v6r-artemis-games.jsonl')
    args=parser.parse_args()
    data=args.data
    rows=[]
    for meta in json.loads((data/'index.json').read_text()):
        rows.extend(measure(meta,data))
    keys=[(r['match'],r['game'],r['name']) for r in rows]
    assert len(set(keys))==len(rows)
    names=list(NAMES.values())
    board=[{k:s.get(k) for k in ('id','name','author','is_public','board_rank','rating','wins','games')}
           for s in json.loads((data/'final-snapshot-board.json').read_text())['strategies'] if s['id'] in NAMES]
    out=dict(leaderboard=board, latest=REVISION,
             distinct_layouts=len({r['layout'] for r in rows}),distinct_games=len({r['match'] for r in rows}),
             all_games={n:stats([r for r in rows if r['name']==n]) for n in names},
             initial_layout_mirrored_on_both_axes=True,
             geometry_profiles={n:geometry_profile([r for r in rows if r['name']==n]) for n in ('Artemis','Apollo')},
             startup_profiles={n:opening_profile([r for r in rows if r['name']==n]) for n in names},
             matched_opponents=matched(rows,names,False),matched_opponent_and_seat=matched(rows,names,True))
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n')
    # The opening aggregates are retained above. The manifest only needs each game's identifying and analysis fields.
    args.games.write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='startup'},ensure_ascii=False)+'\n' for r in rows))


if __name__=='__main__':
    main()
