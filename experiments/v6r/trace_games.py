"""Run two predeclared maps, saving the decisions and post-turn cash of each new game."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arena import parse_strategies, run_game
from benchmark import configure_engine, limit_math_threads

SEEDS = (568554483, 171684640)
V6_SHA = "1995bbcc07592296e6fe6f816412acccb2d39ea2c79787cd937a210513f69afa"
V5_SHA = "0a413a11f80f6872a5c8f0d313061d7bf11121ae16fbdad97247ff3cb15242e5"
V6_REF = "9907fc560c5cc0cf5164014178351dbfb5d64c29:submission.py"
V5_REF = "v8-yan-v5-multistyle:submission.py"


def record_game(task):
    seats, seed, seat, dest = task
    limit_math_threads()
    configure_engine()
    from zerosum_env.envs.carbon import carbon

    compute = carbon.compute_next_board
    turns = []

    def traced(state, env):
        actions = [dict(entry.action) for entry in state]
        updated = compute(state, env)
        turns.append([actions, [p[0] for p in state[0].observation.players]])
        return updated

    carbon.compute_next_board = traced
    try:
        result = run_game((seats, seed, 300))
    finally:
        carbon.compute_next_board = compute
    path = dest / f"trace-{seed}-v6seat{seat}.json.gz"
    with path.open("xb") as archived:
        with gzip.GzipFile(fileobj=archived, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(dict(result=result, actions_and_post_cash=turns), separators=(",", ":")).encode())
    print(path.name, result["status"], result["scores"], flush=True)
    return result["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="optional local copy of the exact frozen v6")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    dest = args.raw_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.check_output(["git", "show", V5_REF], cwd=ROOT)
    source = args.candidate.read_bytes() if args.candidate else subprocess.check_output(["git", "show", V6_REF], cwd=ROOT)
    paths = []
    for name, content, sha in (("yan-v6", source, V6_SHA), ("yan-v5", baseline, V5_SHA)):
        if hashlib.sha256(content).hexdigest() != sha:
            parser.error(f"unexpected {name} SHA")
        path = dest / f"{name}-{sha[:12]}.py"
        if path.exists():
            if path.read_bytes() != content:
                parser.error(f"different contents already at {path}")
        else:
            path.write_bytes(content)
        paths.append(f"{name}={path}")
    players = parse_strategies(paths)
    tasks = [(players if seat == 0 else players[::-1], seed, seat, dest) for seed in args.seeds for seat in (0, 1)]
    if any((dest / f"trace-{s}-v6seat{i}.json.gz").exists() for _, s, i, _ in tasks):
        parser.error("trace output exists, supply a fresh directory")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        statuses = [job.result() for job in as_completed([pool.submit(record_game, task) for task in tasks])]
    return int(any(s != "completed" for s in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
