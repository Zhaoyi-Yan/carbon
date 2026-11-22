"""Walk published matches forward from each player's view and score predicted return moves."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, limit_math_threads, load_strategy


def evaluate_seat(model, net, frames, me):
    side = 1 - me
    profile = model._ReturnProfile()
    tracker = model.BeliefTracker(model._Bel, me)
    counts = Counter()
    previous = None
    for step, (before, after) in enumerate(zip(frames, frames[1:])):
        players = before[0]['observation']['players']
        rival = players[side][2]
        profile.observe(step, rival)
        estimate, _ = tracker.step(players, step, before[me]['observation']['carbon'])
        view = model.encode(players, step, side, estimate, [1.] * model.N, previous)
        logits, _, _ = net(model._quant(view)[None].astype('float32'), model._SID)
        updated = profile.adjust(players, side, logits[0])
        lead = model._late_lead(players, me, step)
        counts['protect' if lead > .18 else 'chase' if lead < -.18 else 'neutral'] += 1

        # Only now consult the next frame. The live agent never sees its labels here.
        next_rival = after[0]['observation']['players'][side][2]
        for wid, (group, closer, base) in profile.forecasts.items():
            if wid not in next_rival:
                counts['unobservable'] += 1
                continue
            cell = rival[wid][0]
            probs = model._probs(updated[cell], model._TEMP)
            calibrated = sum(float(p) for p, dest in zip(probs, model._ROLLOUT_LANDING[cell]) if dest in closer)
            wanted = float(next_rival[wid][0] in closer)
            counts['scored'] += 1
            counts['with_pressure'] += group
            counts['changed_over_one_point'] += abs(calibrated - base) > .01
            counts['base_squared_error'] += (wanted - base) ** 2
            counts['adaptive_squared_error'] += (wanted - calibrated) ** 2
        tracker.set_actions(players, after[me]['action'] or {})
        previous = players
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-dir', type=Path, default=ROOT / 'replays/top-five/replays')
    parser.add_argument('--count', type=int, help='use only the first N archived matches')
    args = parser.parse_args()
    configure_engine()
    limit_math_threads()
    model = load_strategy('adaptive_candidate', ROOT / 'submission.py')
    net = model._net()
    if net is None:
        raise RuntimeError('A CPU PyTorch runtime is required')
    result = []
    totals = Counter()
    for path in sorted(args.replay_dir.glob('*.json.gz'))[:args.count]:
        with gzip.open(path) as f:
            replay = json.load(f)
        model.LAST_TURN = replay['configuration']['episodeSteps'] - 2
        seats = [evaluate_seat(model, net, replay['steps'], seat) for seat in range(2)]
        result.append(dict(replay=path.name, seats=[dict(s) for s in seats]))
        for counts in seats:
            totals.update(counts)
    if not totals['scored']:
        parser.error('No eligible worker transitions in the selected replay files')
    print(json.dumps(dict(games=result, totals=dict(totals),
                          baseline_brier=totals['base_squared_error']/totals['scored'],
                          adaptive_brier=totals['adaptive_squared_error']/totals['scored']), indent=2))


if __name__ == '__main__':
    main()
