"""Offline Carbon round robin for trusted strategy files. Each seed is played from both seats.

Each player has its own persistent process per game. Local timing uses wall clock including
IPC, with a 20s initialization limit, 1s per move, 3s hard move limit, three slow moves
and 5s of total overtime. Host CPU/memory allocation and network access are not sandboxed.
The JSONL contains one record per finished match, including forfeits and engine errors.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

from benchmark import configure_engine, limit_math_threads, load_strategy, new_game, observe, validate_actions


@dataclass(frozen=True)
class Strategy:
    name: str
    path: Path
    sha256: str


def parse_strategies(values):
    entries = {}
    for value in values:
        name, sep, filename = value.partition("=")
        path = Path(filename if sep else value).resolve()
        name = name if sep else path.stem
        if not name or name in entries:
            raise ValueError(f"strategy names must be nonempty and unique: {name!r}")
        if not path.is_file():
            raise ValueError(f"strategy file missing: {path}")
        entries[name] = Strategy(name, path, hashlib.sha256(path.read_bytes()).hexdigest())
    if len(entries) < 2:
        raise ValueError("supply at least two strategies")
    return list(entries.values())


def agent_process(pipe, strategy, config, seed):
    # All inputs are local, trusted Python code. Restrict the return channel to JSON actions.
    def reply(kind, detail):
        pipe.send_bytes(json.dumps([kind, detail], allow_nan=False).encode())

    try:
        configure_engine()
        limit_math_threads()
        import numpy as np
        from zerosum_env.utils import structify

        source = strategy.path.read_bytes()
        if hashlib.sha256(source).hexdigest() != strategy.sha256:
            reply("input_error", "Strategy changed during the tournament")
            return
        random.seed(seed)
        np.random.seed(seed)
        module = load_strategy("arena_strategy", strategy.path, source=source)
        if not callable(getattr(module, "agent", None)):
            raise TypeError("Missing callable agent(observation, configuration)")
        config = structify(config)
        reply("ok", getattr(module, "_SEARCH_OK", None))
        while True:
            obs = pipe.recv()
            if obs is None:
                return
            action = module.agent(obs, config)
            try:
                validate_actions(action, obs.players[obs.player])
            except (ValueError, TypeError) as exc:
                reply("invalid_action", str(exc)[:600])
                return
            searched = getattr(module, "_S_STATS", {})
            count = searched.get("searched") if isinstance(searched, dict) else None
            reply("ok", [action, count if isinstance(count, int) else None])
    except (EOFError, BrokenPipeError):
        return
    except BaseException as exc:
        reply("error", f"{type(exc).__name__}: {str(exc)[:600]}")
    finally:
        pipe.close()


class PolicyFault(Exception):
    def __init__(self, kind, detail):
        super().__init__(detail)
        self.kind = kind


class Player:
    def __init__(self, ctx, strategy, config, seed):
        self.pipe, self.child_pipe = ctx.Pipe()
        self.process = ctx.Process(target=agent_process, args=(self.child_pipe, strategy, config, seed))
        self.remaining = 5.
        self.data = dict(name=strategy.name, sha256=strategy.sha256, init_seconds=None,
                         total_act_seconds=0., max_act_seconds=0., slow_steps=0,
                         calls=0, search_enabled=None, searches=None, failure=None)

    def close(self):
        self.child_pipe.close()
        self.pipe.close()
        if self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(.5)
            if self.process.is_alive():
                self.process.kill()
            self.process.join()
        self.process.close()

    def receive(self, deadline):
        try:
            if self.pipe.poll(max(0, deadline-time.monotonic())):
                kind, detail = json.loads(self.pipe.recv_bytes())
                if time.monotonic() <= deadline:
                    if kind == "ok":
                        return detail
                    raise PolicyFault(kind, detail)
        except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
            raise PolicyFault("process_exit", "Strategy process exited before replying") from exc
        raise PolicyFault("timeout", "Wall-clock deadline exceeded")

    def initialize(self):
        started = time.monotonic()
        self.process.start()
        self.child_pipe.close()
        try:
            self.data["search_enabled"] = self.receive(started+20)
        finally:
            self.data["init_seconds"] = time.monotonic()-started

    def act(self, obs):
        soft = 1.
        overtime = min(2., self.remaining) if self.data["slow_steps"] < 3 else 0.
        started = time.monotonic()
        try:
            try:
                self.pipe.send(obs)
            except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
                raise PolicyFault("process_exit", "Strategy process exited before acting") from exc
            action, self.data["searches"] = self.receive(started+soft+overtime)
            return action
        finally:
            elapsed = time.monotonic()-started
            self.data["calls"] += 1
            self.data["total_act_seconds"] += elapsed
            self.data["max_act_seconds"] = max(elapsed, self.data["max_act_seconds"])
            if elapsed > soft:
                self.data["slow_steps"] += 1
                self.remaining = max(0., self.remaining-(elapsed-soft))


def engine_fingerprint(carbon):
    package = Path(carbon.__file__).parent
    digest = hashlib.sha256()
    for name in ("carbon.py", "carbon.json", "helpers.py", "idgen.py"):
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def finish_forfeit(result):
    failed = [i for i, p in enumerate(result["players"]) if p["failure"]]
    if not failed:
        return False
    if any(result["players"][i]["failure"]["kind"] == "input_error" for i in failed):
        result["status"] = "input_error"
    elif len(failed) == 2:
        result["status"] = "double_forfeit"
    else:
        result["status"] = "forfeit"
        result["winner"] = 1-failed[0]
    return True


def run_game(task):
    seats, seed, steps = task
    result = dict(schema_version=1, seed=seed, episode_steps=steps, turns=0, map_sha256=None,
                  engine_sha256=None, status="engine_error", winner=None, scores=None, cash=None,
                  players=[dict(name=p.name, sha256=p.sha256, failure=None) for p in seats])
    try:
        limit_math_threads()
        configure_engine()
        from zerosum_env.envs.carbon import carbon

        result["engine_sha256"] = engine_fingerprint(carbon)
        state, env = new_game(seed, steps)
        board = state[0].observation
        opening = [board.full_carbon, [list(p[1].values()) for p in board.players]]
        result["map_sha256"] = hashlib.sha256(json.dumps(opening, separators=(",", ":")).encode()).hexdigest()
        ctx = mp.get_context("spawn")
        with ExitStack() as cleanup:
            players = []
            for seat, strategy in enumerate(seats):
                p = Player(ctx, strategy, dict(env.configuration), seed)
                cleanup.callback(p.close)
                players.append(p)
                result["players"][seat] = p.data
                try:
                    p.initialize()
                except PolicyFault as exc:
                    p.data["failure"] = dict(kind=exc.kind, phase="init", turn=None, detail=str(exc))
            if finish_forfeit(result):
                return result
            for turn in range(steps-1):
                for seat, p in enumerate(players):
                    try:
                        action = p.act(observe(state, seat, turn, p.remaining))
                        validate_actions(action, state[0].observation.players[seat])
                        state[seat].action = action
                    except (PolicyFault, ValueError) as exc:
                        kind = exc.kind if isinstance(exc, PolicyFault) else "invalid_action"
                        p.data["failure"] = dict(kind=kind, phase="act", turn=turn, detail=str(exc))
                if finish_forfeit(result):
                    return result
                carbon.compute_next_board(state, env)
                result["turns"] = turn+1
                if any(entry.status != "ACTIVE" for entry in state):
                    break
        scores = [entry.reward for entry in state]
        result.update(status="completed", scores=scores, cash=[p[0] for p in state[0].observation.players],
                      winner=None if scores[0] == scores[1] else int(scores[1] > scores[0]))
    except Exception as exc:
        result.update(status="engine_error", winner=None, detail=f"{type(exc).__name__}: {exc}")
    return result


def record_result(result, file, standings):
    file.write(json.dumps(result, allow_nan=False, ensure_ascii=False) + "\n")
    file.flush()
    players = [p["name"] for p in result["players"]]
    status, winner = result["status"], result["winner"]
    for seat, name in enumerate(players):
        if status == "completed":
            label = "draw" if winner is None else "win" if seat == winner else "loss"
        elif status == "forfeit":
            label = "won_by_forfeit" if seat == winner else "forfeited"
        else:
            label = status
        standings[name][label] += 1
    print(f"seed={result['seed']} {' / '.join(players)} {status} "
          f"{result['scores'] or [p['failure'] for p in result['players']]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", required=True, metavar="[NAME=]FILE")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file, one record per game")
    args = parser.parse_args()
    try:
        strategies = parse_strategies(args.strategies)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.jobs < 1 or not 2 <= args.steps <= 300:
        parser.error("jobs must be positive and steps must be in 2..300")
    if any(not 0 <= s < 2**32 for s in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique integers in 0..4294967295")
    tasks = [(seats, seed, args.steps) for pair in combinations(strategies, 2)
             for seed in args.seeds for seats in (pair, pair[::-1])]
    standings = {p.name: Counter() for p in strategies}
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("x", encoding="utf-8")
    except OSError as exc:
        parser.error(str(exc))
    invalid = False
    with output:
        if args.jobs == 1:
            for task in tasks:
                result = run_game(task)
                record_result(result, output, standings)
                invalid |= result["status"] != "completed"
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks)),
                                     mp_context=mp.get_context("spawn")) as pool:
                futures = [pool.submit(run_game, t) for t in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    record_result(result, output, standings)
                    invalid |= result["status"] != "completed"
    print(json.dumps(standings, sort_keys=True, ensure_ascii=False))
    return 2 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
