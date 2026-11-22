"""Compare generated actions, random streams and sampler CPU cost on recorded boards."""
import argparse
import gzip
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_replays import load_strategy


def audit(original, candidate):
    old = load_strategy(original)
    new = load_strategy(candidate)
    samples = []
    covered = set()
    for path in sorted((ROOT / 'replays/apollo').glob('*.json.gz'))[:4]:
        with gzip.open(path, 'rt') as stream:
            steps = json.load(stream)['steps']
        for side in (0, 1):
            tracker = old.BeliefTracker(old._Bel, side)
            previous = None
            for t, (cur, nxt) in enumerate(zip(steps, steps[1:])):
                players = cur[0]['observation']['players']
                carbon, known = tracker.step(players, t, cur[side]['observation']['carbon'])
                tracker.set_actions(players, nxt[side]['action'] or {})
                if t % 15 == 0 or t == len(steps)-2:
                    x = old._quant(old.encode(players, t, side, carbon, known, previous))
                    lw, lb = old._POLICY.forward(x, old._SID)
                    ids = list(players[side][2])
                    fixed = {key: (nxt[side]['action'] or {}).get(key) for key in ids[::4]}
                    for hold in (None, fixed):
                        for temperature in (0.65, 1., 2.5):
                            samples.append((players, side, lw / temperature, lb, t, hold))
                    covered.add((path.name, side, t))
                previous = players
    rng_old, rng_new = (np.random.RandomState(78) for _ in range(2))
    for i, (players, side, lw, lb, step, fixed) in enumerate(samples):
        before = old._sample(players, side, lw, lb, rng_old, step, fixed)
        after = new._sample(players, side, lw, lb, rng_new, step, fixed)
        if before != after or rng_old.random_sample() != rng_new.random_sample():
            raise ValueError(f'action or stream differs at sample {i}: {before} / {after}')
    times = {'original': [], 'batch': []}
    for order in (('original','batch'), ('batch','original'), ('original','batch')):
        for label in order:
            sampler = (old if label == 'original' else new)._sample
            rng = np.random.RandomState(102)
            start = time.process_time()
            for players, side, lw, lb, step, fixed in samples:
                sampler(players, side, lw, lb, rng, step, fixed)
            times[label].append(time.process_time()-start)
    return dict(boards=len(covered), matched_actions=len(samples), process_cpu_seconds=times,
                sampler_speedup=statistics.median(times['original']) / statistics.median(times['batch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.original, args.candidate), indent=2))
