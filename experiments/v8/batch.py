"""Run the three frozen v8 arms and, when qualified, a second set of unused maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arena import Strategy, run_game
from benchmark import configure_engine, limit_math_threads
from scripts.selfplay import source_for
from economy import build

ARMS = ('trees', 'cargo', 'combined')


def seeds(stage):
    return [int.from_bytes(hashlib.sha256(f'yan-v8-payback-20261123/{stage}/{i}'.encode()).digest()[:4],
                           'big') % 1000000000 for i in range(4)]


def candidate_scores(rows):
    outcomes = [[] for _ in range(2)]
    for row in rows:
        if row['status'] != 'completed' or row['turns'] != 299 or not all(p['search_enabled'] for p in row['players']):
            return False
        i = next(i for i, p in enumerate(row['players']) if p['name'] != 'yan-v7')
        outcomes[i].append(row['scores'][i] - row['scores'][1-i])
    return all(len(v) == 4 and sum(x>0 for x in v) >= 3 for v in outcomes) and sum(map(sum, outcomes)) > 0


def stage(out, stage_name, arms, players, jobs):
    tasks = []
    for n, seed in enumerate(seeds(stage_name)):
        for j in range(len(arms)):
            arm = arms[(j+n) % len(arms)]
            for i in (n % 2, 1-n % 2):
                pair = [players[arm], players['yan-v7']]
                tasks.append((pair if i == 0 else pair[::-1], seed, 300, True))
    rows = {arm: [] for arm in arms}
    path = out / (stage_name + '.jsonl')
    with path.open('x') as file, ProcessPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(run_game, task) for task in tasks]):
            row = future.result()
            file.write(json.dumps(row, separators=(',', ':'))+'\n')
            file.flush()
            arm = next(p['name'] for p in row['players'] if p['name'] != 'yan-v7')
            rows[arm].append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    os.environ['CARBON_DEBUG'] = '1'
    limit_math_threads()
    configure_engine()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen, digest, data = source_for('yan-v7', ROOT)
    base = out / 'yan-v7.py'
    base.write_bytes(data)
    players = {frozen: Strategy(frozen,base,digest)}
    for arm in ARMS:
        source = build(data.decode(), arm).encode()
        path = out / ('yan-v8-'+arm+'.py')
        compile(source, str(path), 'exec')
        path.write_bytes(source)
        players[arm] = Strategy(arm, path, hashlib.sha256(source).hexdigest())
    protocol = dict(baseline_tag='v10-yan-v7-selfplay-prior', sha256={a:p.sha256 for a,p in players.items()},
                    screening=seeds('screening'), confirmation=seeds('confirmation'),
                    gate='At each stage, require at least 3/4 wins per seat and a positive mean margin. Only the combined arm can advance.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows = stage(out, 'screening', ARMS, players, args.jobs)
    if candidate_scores(rows['combined']):
        stage(out, 'confirmation', ('combined',), players, args.jobs)
    print('Batch complete.', flush=True)


if __name__ == '__main__':
    main()
