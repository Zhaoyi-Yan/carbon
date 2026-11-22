"""Recompute the frozen yan-v2 cohort from public match summaries and optional full replays."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import time


STRATEGY = '03b736af6aaf915fe90a48b0127e5cac'
ARCHIVE = Path(__file__).with_name('yan-v2-games.jsonl')
HORIZON = 18


def summary(match, game, cache):
    path = cache / f'{match}-{game}.json'
    if path.exists():
        return json.loads(path.read_bytes())
    url = f'https://117.72.210.134/api/summary/{match}/{game}'
    def page(u):
        data = subprocess.check_output(['curl', '-fsSL', '--retry', '4', '--max-time', '25', u])
        time.sleep(2.1)  # Published summary endpoint limit is 30 requests/minute.
        return json.loads(data)
    result = page(url)
    offset = result['next_events_offset']
    while offset is not None:
        chunk = page(f'{url}?events_offset={offset}')
        if chunk['events_offset'] != offset or chunk['total_events'] != result['total_events']:
            raise ValueError(f'inconsistent summary pages for {match}')
        result['events'].extend(chunk['events'])
        offset = chunk['next_events_offset']
    if len(result['events']) != result['total_events']:
        raise ValueError(f'incomplete summary for {match}')
    result.update(next_events_offset=None, complete=True)
    path.write_text(json.dumps(result))
    return result


def exposures(replay, me):
    """At planting, measure visible rival planters and allies from the preceding frame."""
    frames = replay['steps']
    last = len(frames)-1
    size = replay['configuration']['size']
    births = {}
    income = [[0., 0.], [0., 0.]]  # indexed by original and present owner
    for t in range(1, len(frames)):
        old = frames[t-1][0]['observation']['players']
        obs = frames[t][0]['observation']
        for side, pl in enumerate(obs['players']):
            for key, (pos, _) in pl[3].items():
                if key not in births:
                    def within_two(worker):
                        dr = abs(worker[0]//size-pos//size)
                        dc = abs(worker[0]%size-pos%size)
                        return min(dr, size-dr)+min(dc, size-dc) <= 2
                    rival = any(w[2] == 'PLANTER' and within_two(w) for w in old[1-side][2].values())
                    guard = any(w[2] == 'COLLECTOR' and w[1] < 5 and within_two(w) for w in old[side][2].values())
                    births[key] = [side, t, rival, guard, False]
                birth = births[key]
                if t-birth[1] <= HORIZON and side != birth[0]:
                    birth[4] = True
                income[birth[0]][side] += obs['trees'][key][1]
    strata = {}
    for origin, t, rival, guard, captured in births.values():
        if origin != me or t > last-HORIZON:
            continue
        key = 'far' if not rival else ('near_guard' if guard else 'near_exposed')
        counts = strata.setdefault(key, [0, 0])
        counts[0] += int(captured)
        counts[1] += 1
    return dict(own_plants_own_income=round(income[me][me], 2),
                own_plants_opponent_income=round(income[me][1-me], 2),
                opponent_plants_own_income=round(income[1-me][me], 2),
                opponent_plants_opponent_income=round(income[1-me][1-me], 2),
                nearby_at_plant=strata)


def measure(x, seed, map_sha, replay):
    meta = x['metadata']
    if len(x['events']) != x['total_events']:
        raise ValueError(f"incomplete cached event stream for {meta['id']}")
    own = next(k for k, v in meta['strategies'].items() if v['id'] == STRATEGY)
    seat = meta['order'].index(own)
    other = meta['order'][1-seat]
    if any(status != 'DONE' for status in meta['statuses'].values()):
        raise ValueError(f"abnormal finish in {meta['id']}")
    frames = {t: [p0, p1] for t, p0, p1, _ in x['samples']}
    last = max(frames)
    if last != 299 or abs(meta['scores'][own]-frames[last][seat][0]) > .02:
        raise ValueError(f"score/step mismatch in {meta['id']}")
    born = {}
    fast = plants = new_total = seizes = 0
    for step, kind, actor, *args in x['events']:
        if kind == 'plant':
            born[args[0]] = [actor, step, False]
            new_total += int(actor == seat)
            plants += int(actor == seat and step <= last-HORIZON)
        if kind == 'seize':
            seizes += int(actor == seat)
            birth = born[args[0]]
            if actor != birth[0]:
                if not birth[2] and birth[0] == seat and birth[1] <= last-HORIZON and step-birth[1] <= HORIZON:
                    fast += 1
                birth[2] = True
    snapshots = {}
    for t in (100, 150, 250, 280, 290, 299):
        a, b = frames[t][seat], frames[t][1-seat]
        snapshots[t] = dict(cash_margin=round(a[0]-b[0], 2), cargo=[a[4], b[4]], trees=[a[3], b[3]])
    row = dict(match=meta['id'], game=meta['game'], match_seed=seed, map_sha256=map_sha, seat=seat,
               opponent=meta['strategies'][other]['name'], opponent_id=meta['strategies'][other]['id'],
               scores=[meta['scores'][own], meta['scores'][other]], won=meta['winner']==own,
               early_seized=fast, eligible_new_trees=plants, all_plants=new_total, all_seizes=seizes,
               snapshots=snapshots)
    if replay is not None:
        row['replay_random_seed'] = replay['configuration']['randomSeed']
        row.update(exposures(replay, seat))
        rates = row['nearby_at_plant'].values()
        if (fast, plants) != (sum(c[0] for c in rates), sum(c[1] for c in rates)):
            raise ValueError('Lifetime/owner labels from full replay disagree with summary')
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, default=ARCHIVE, help='JSONL manifest with match, game, match_seed and map_sha256')
    parser.add_argument('--cache', required=True, type=Path, help='local summary cache; missing entries are fetched')
    parser.add_argument('--snapshot', type=Path, help='official snapshot directory with replays/*.json.gz')
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for line in args.games.read_text().splitlines():
        g = json.loads(line)
        match, game, seed = g['match'], g['game'], g['match_seed']
        p = args.snapshot/'replays'/f'{match}-{game}.json.gz' if args.snapshot else None
        r = None
        if p and p.exists():
            with gzip.open(p) as stream:
                r = json.load(stream)
        print(json.dumps(measure(summary(match, game, args.cache), seed, g['map_sha256'], r), ensure_ascii=False, separators=(',', ':')))
