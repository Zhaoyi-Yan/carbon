"""Run paired Carbon matches offline using the upstream zerosum_env interpreter."""

import argparse
import importlib.util
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def load_strategy(name, path, source=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def validate_actions(action, player):
    _, bases, workers, _ = player
    if not isinstance(action, dict):
        raise ValueError("agent must return a dictionary")
    for unit, command in action.items():
        legal = (("RECCOLLECTOR", "RECPLANTER") if unit in bases
                 else ("UP", "RIGHT", "DOWN", "LEFT") if unit in workers else ())
        if command not in legal:
            raise ValueError(f"invalid action {unit!r}: {command!r}")


def configure_engine():
    # Honour an explicit upstream checkout, otherwise prefer the bundled interpreter.
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    upstream = next((Path(p) for p in paths if p and (Path(p) / "zerosum_env").is_dir()), None)
    root = upstream or Path(__file__).resolve().parent / "vendor"
    if (root / "zerosum_env").is_dir() and str(root) not in sys.path[:1]:
        sys.path.insert(0, str(root))


def limit_math_threads():
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"


def new_game(seed, steps):
    import numpy as np
    from zerosum_env import utils
    from zerosum_env.envs.carbon import carbon

    random.seed(seed)
    np.random.seed(seed)
    fields = carbon.specification["configuration"]
    config = {key: value.get("default") if isinstance(value, dict) else value
              for key, value in fields.items()}
    config.update(episodeSteps=steps, randomSeed=seed)
    env = SimpleNamespace(configuration=utils.structify(config))
    state = [SimpleNamespace(
        observation=utils.structify({"step": 0, "player": player, "remainingOverageTime": 5}),
        reward=80, status="ACTIVE", action={},
    ) for player in range(2)]
    carbon.populate_board(state, env)
    return state, env


def observe(state, player, turn, remaining=5):
    from zerosum_env import utils

    board = state[0].observation
    return utils.structify({
        "step": turn, "player": player, "remainingOverageTime": remaining,
        "players": board.players, "trees": board.trees,
        "carbon": state[player].observation.carbon,
    })


def play_match(task):
    limit_math_threads()
    configure_engine()
    from zerosum_env.envs.carbon import carbon

    candidate_path, opponent_path, seed, seat, steps, min_searches, progress_every = task
    state, env = new_game(seed, steps)
    config = env.configuration
    policies = [None, None]
    policies[seat] = load_strategy("candidate", candidate_path)
    policies[1 - seat] = load_strategy("opponent", opponent_path)
    if min_searches and not getattr(policies[1 - seat], "_SEARCH_OK", False):
        raise RuntimeError(f"{opponent_path.name} loaded without search. Check the PyTorch CPU runtime.")
    label = f"{opponent_path.stem} seed={seed} seat={seat}"
    durations = []
    slow_steps = 0
    for turn in range(steps - 1):
        board = state[0].observation
        for player, entry in enumerate(state):
            obs = observe(state, player, turn)
            started = time.perf_counter()
            entry.action = policies[player].agent(obs, config)
            elapsed = time.perf_counter() - started
            validate_actions(entry.action, board.players[player])
            if player == seat:
                durations.append(elapsed)
                slow_steps += elapsed > 1.0
        carbon.compute_next_board(state, env)
        if progress_every and (turn + 1) % progress_every == 0:
            scores = [entry.reward for entry in state]
            print(f"{label} turn={turn + 1} scores={scores[seat]:.0f}/{scores[1 - seat]:.0f}", flush=True)
        if any(entry.status != "ACTIVE" for entry in state):
            break
    searches = getattr(policies[1 - seat], "_S_STATS", {}).get("searched", 0)
    score = state[seat].reward
    other = state[1 - seat].reward
    return (Path(opponent_path).stem, seed, seat, score, other, turn + 1,
            statistics.mean(durations), max(durations), slow_steps, searches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).with_name("submission.py"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--opponents", nargs="+", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--min-opponent-searches", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()
    if args.min_opponent_searches < 0 or args.progress_every < 0:
        parser.error("search and progress counts must be nonnegative")
    if args.steps < 2 or args.jobs < 1:
        parser.error("--steps must be at least 2 and --jobs must be positive")
    for path in (args.candidate, *args.opponents):
        if not path.is_file():
            parser.error(f"strategy file not found: {path}")

    tasks = [(args.candidate.resolve(), opponent.resolve(), seed, seat, args.steps,
              args.min_opponent_searches, args.progress_every)
             for opponent in args.opponents for seed in args.seeds for seat in range(2)]
    results = defaultdict(list)
    valid_searches = True
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
        for name, seed, seat, score, other, turn, average, worst, slow, searches in pool.map(play_match, tasks):
            margin = score - other
            results[name].append((margin, average, worst, slow))
            search_summary = ""
            if args.min_opponent_searches:
                sufficient = searches >= args.min_opponent_searches
                valid_searches &= sufficient
                search_summary = f" opponent_searches={searches} {'PASS' if sufficient else 'LOW'}"
            print(f"{name:20} seed={seed:<5} seat={seat} turn={turn:<3} "
                  f"score={score:9.2f}/{other:9.2f}  margin={margin:+9.2f}  "
                  f"mean={average * 1000:6.2f} ms max={worst * 1000:7.2f} ms{search_summary}", flush=True)

    all_won = True
    for name, matches in results.items():
        wins = sum(margin > 0 for margin, *_ in matches)
        draws = sum(margin == 0 for margin, *_ in matches)
        mean_margin = statistics.mean(margin for margin, *_ in matches)
        peak_ms = max(worst for _, _, worst, _ in matches) * 1000
        slow_steps = sum(slow for _, _, _, slow in matches)
        all_won &= wins == len(matches)
        print(f"{name:20} {wins}-{draws}-{len(matches) - wins - draws} W-D-L"
              f"  mean margin {mean_margin:+.2f}  max {peak_ms:.2f} ms  slow steps {slow_steps}")
    if not valid_searches:
        print(f"Some games completed fewer than {args.min_opponent_searches} opponent searches.")
        return 2
    return 0 if all_won else 1


if __name__ == "__main__":
    raise SystemExit(main())
