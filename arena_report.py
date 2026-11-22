"""Summarize seat-swapped Carbon arena JSONL files, including incomplete and failed games."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def summarize(paths):
    names = {}
    seeds = {}
    games = {}
    settings = set()
    for path in paths:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row['schema_version'] != 1 or len(row['players']) != 2:
                        raise ValueError('unknown arena schema')
                    seats = tuple(p['name'] for p in row['players'])
                    if seats[0] == seats[1]:
                        raise ValueError('use separate names for the two seats')
                    for p in row['players']:
                        if names.setdefault(p['name'], p['sha256']) != p['sha256']:
                            raise ValueError(f"{p['name']} has multiple strategy hashes")
                    engine, steps, seed = row['engine_sha256'], row['episode_steps'], row['seed']
                    if engine:
                        settings.add((engine, steps))
                    fingerprint = row['map_sha256']
                    if fingerprint and seeds.setdefault(seed, fingerprint) != fingerprint:
                        raise ValueError(f'seed {seed} produced different maps')
                    key = (seed, seats)
                    if key in games and row != games[key]:
                        raise ValueError(f'conflicting runs for seed {seed} and seats {seats}; report the runs separately')
                    status = row['status']
                    if status not in ('completed', 'forfeit', 'double_forfeit', 'input_error', 'engine_error'):
                        raise ValueError(f'unknown status {status}')
                    if status == 'completed' and (not fingerprint or not engine or
                                                  len(row['cash']) != 2 or len(row['scores']) != 2 or
                                                  row['winner'] not in (None, 0, 1)):
                        raise ValueError('normal games require scores, cash and fingerprints')
                    games[key] = row
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f'{path}:{number}: {exc}') from exc
    if len(settings) > 1:
        raise ValueError('different referees or episode lengths; report the runs separately')
    if not games:
        raise ValueError('no game records')
    matches = defaultdict(lambda: defaultdict(dict))
    for (seed, seats), row in games.items():
        pair = tuple(sorted(seats))
        matches[pair][seed][seats.index(pair[0])] = row
    summaries = []
    for (left, right), maps in sorted(matches.items()):
        wins = draws = losses = unpaired = 0
        problems = Counter()
        failures = Counter()
        cash_margins = []
        for rows in maps.values():
            completed = {side: r for side, r in rows.items() if r['status'] == 'completed'}
            for r in rows.values():
                if r['status'] != 'completed':
                    problems[r['status']] += 1
                    for player in r['players']:
                        if player.get('failure'):
                            failures[(player['name'], player['failure']['kind'])] += 1
            if len(completed) < 2:
                unpaired += len(completed)
                continue
            two_seats = []
            for side, r in completed.items():
                draws += r['winner'] is None
                wins += r['winner'] == side
                losses += r['winner'] == 1-side
                two_seats.append(r['cash'][side] - r['cash'][1-side])
            cash_margins.append(statistics.mean(two_seats))
        summaries.append(dict(left=left, right=right, paired_maps=len(cash_margins),
                              wins=wins, draws=draws, losses=losses,
                              mean_cash_margin=statistics.mean(cash_margins) if cash_margins else None,
                              unpaired_completed=unpaired, faults=dict(sorted(problems.items())),
                              player_failures=[dict(name=name, kind=kind, count=count)
                                               for (name, kind), count in sorted(failures.items())]))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    try:
        results = summarize(args.files)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print('Left / Right | paired maps | W-D-L (left) | mean cash margin (left) | unpaired games | faults')
        for r in results:
            margin = r['mean_cash_margin']
            cash = '—' if margin is None else f'{margin:+.2f}'
            print(f"{r['left']} / {r['right']} | {r['paired_maps']} | "
                  f"{r['wins']}-{r['draws']}-{r['losses']} | {cash} | "
                  f"{r['unpaired_completed']} | {r['faults'] or '—'}")


if __name__ == '__main__':
    main()
