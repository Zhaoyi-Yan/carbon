"""Re-adjudicate captured actions with the pinned interpreter and reconcile cash and carried carbon."""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

from trace_games import ROOT, SEEDS, V5_SHA, V6_SHA

from arena import engine_fingerprint
from benchmark import configure_engine, limit_math_threads, new_game

CASH_RULES = {
    "player._cash = max(player._cash - rec_collector_cost, 0)": "hire_collector",
    "player._cash = max(player._cash - rec_planter_cost, 0)": "hire_planter",
    "worker.cell.tree.player._cash += worker.carbon": "tree_collision",
    "worker.cell.recrtCenter.player._cash += worker.carbon": "base_collision",
    "recrtCenter.player._cash += worker._carbon - configuration.smelt_cost": "bank",
    "worker.player._cash = worker.player._cash - player_plant_cost[worker.player_id]": "plant",
    "worker.player._cash -= configuration.seize_cost": "seize",
    "tree.player._cash += tree_carbon": "tree_yield",
    "player._cash = round(player._cash, 2)": "rounding",
}
CARGO_RULES = {
    "worker._carbon = worker._carbon * (1 - board.configuration.move_cost)": "move",
    "winner._carbon += worker.carbon": "loot",
    "worker._carbon = 0": "unload",
    "worker._carbon += delta_carbon": "harvest",
    "board.workers[worker_id]._carbon = max(board.workers[worker_id].carbon - absorbed_carbon, 0)": "enemy_tree",
}


def rule_lines(fn, rules):
    source, first = inspect.getsourcelines(fn)
    found = {first+i: rules[line.strip()] for i, line in enumerate(source) if line.strip() in rules}
    if sorted(found.values()) != sorted(rules.values()):
        raise ValueError("referee rules changed, review the source before replaying")
    return found


def fresh():
    return {name: [Counter(), Counter()] for name in ("cash", "cargo", "counts", "tree_sources")}


def add(total, step):
    for kind in total:
        for seat in (0, 1):
            total[kind][seat].update(step[kind][seat])


def snapshot(obs):
    return [dict(cash=p[0], cargo=sum(w[1] for w in p[2].values()), trees=len(p[3]),
                 collectors=sum(w[2] == "COLLECTOR" for w in p[2].values()),
                 planters=sum(w[2] == "PLANTER" for w in p[2].values())) for p in obs.players]


def check_near(a, b, what):
    if abs(a-b) > 1e-6:
        raise ValueError(f"{what}: {a} vs {b}")


def replay(archive):
    raw = archive.read_bytes()
    game = json.loads(gzip.decompress(raw))
    summary, steps = replay_actions(game["result"], game["actions_and_post_cash"], V6_SHA)
    summary["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    return summary, steps


def replay_actions(result, actions, candidate_sha):
    from zerosum_env.envs.carbon import carbon, helpers
    if result["engine_sha256"] != engine_fingerprint(carbon):
        raise ValueError("referee differs from game recorder")
    if result["status"] != "completed" or result["turns"] != len(actions):
        raise ValueError("incomplete recorded game")
    seat = [p["sha256"] for p in result["players"]].index(candidate_sha)
    if result["players"][1-seat]["sha256"] != V5_SHA:
        raise ValueError("baseline mismatch")
    state, env = new_game(result["seed"], result["episode_steps"])
    initial = snapshot(state[0].observation)
    start = prev = initial
    total, phase, phases = fresh(), fresh(), []
    board_code = helpers.Board.next.__code__
    cash_lines = rule_lines(helpers.Board.next, CASH_RULES)
    cargo_lines = rule_lines(helpers.Board.next, CARGO_RULES)
    source, first = inspect.getsourcelines(helpers.Board.next)
    delete_lines = [first+i for i, line in enumerate(source) if line.strip() == "board._delete_worker(worker)"]
    if len(delete_lines) != 2:
        raise ValueError("referee worker removal changed")
    death_keys = dict(zip(delete_lines, ("collision", "enemy_base")))
    old_delete = helpers.Board._delete_worker

    def player_write(p, attr, new):
        frame = sys._getframe(1)
        if attr == "_cash" and frame.f_code is board_code:
            key = cash_lines[frame.f_lineno]
            value = new - p.cash
            fields = frame.f_locals
            if key == "tree_yield":
                tree = fields["tree"]
                board = fields["board"]
                adjacent = [tree.position] + [tree.position.translate(d, board.configuration.size) for d in tree.surround()]
                taxed = sum(board.cells[pos].worker.carbon for pos in adjacent
                            if board.cells[pos].worker is not None and board.cells[pos].worker.is_collector
                            and board.cells[pos].worker.player_id != p.id)
                taxed *= board.configuration.collector_absorption_rate
                turn["tree_sources"][p.id]["opposing_cargo"] += taxed
                turn["tree_sources"][p.id]["ground"] += fields["tree_carbon"] - taxed
                turn["counts"][p.id]["eligible_tree_turns"] += 1
            if key == "bank" and fields["worker"].player_id != p.id:
                key = "bank_from_enemy"
            turn["cash"][p.id][key] += value
            if key in ("hire_collector", "hire_planter", "plant", "seize") or (key.startswith("bank") and value > 0):
                turn["counts"][p.id][key] += 1
        return object.__setattr__(p, attr, new)

    def worker_write(w, attr, new):
        frame = sys._getframe(1)
        if attr == "_carbon" and frame.f_code is board_code:
            key = cargo_lines[frame.f_lineno]
            if key == "loot" and frame.f_locals["worker"].player_id == w.player_id:
                key = "loot_from_own"
            turn["cargo"][w.player_id][key] += new - w.carbon
        return object.__setattr__(w, attr, new)

    def remove_worker(board, w):
        frame = sys._getframe(1)
        if frame.f_code is board_code:
            key = death_keys[frame.f_lineno]
            turn["cargo"][w.player_id][key] -= w.carbon
            turn["counts"][w.player_id][key + ("_collector" if w.is_collector else "_planter")] += 1
        return old_delete(board, w)

    trace_steps = []
    with ExitStack() as hooks:
        hooks.enter_context(patch.object(helpers.Player, "__setattr__", player_write))
        hooks.enter_context(patch.object(helpers.Worker, "__setattr__", worker_write))
        hooks.enter_context(patch.object(helpers.Board, "_delete_worker", remove_worker))
        for index, (decisions, expected) in enumerate(actions, start=1):
            for p, decision in zip(state, decisions):
                p.action = decision
            turn = fresh()
            carbon.compute_next_board(state, env)
            post = snapshot(state[0].observation)
            for i in (0, 1):
                check_near(post[i]["cash"], expected[i], f"recorded cash at turn {index}, seat {i}")
                for stock in ("cash", "cargo"):
                    change = sum(turn[stock][i].values())
                    check_near(post[i][stock], prev[i][stock]+change, f"{stock} at turn {index}, seat {i}")
                check_near(turn["cash"][i]["tree_yield"], sum(turn["tree_sources"][i].values()),
                           f"tree receipts at turn {index}, seat {i}")
                for name in ("trees", "collectors", "planters"):
                    turn["counts"][i][name + "_turns"] += post[i][name]
            add(total, turn)
            add(phase, turn)
            trace_steps.append(dict(turn=index, end=post, ledger=turn))
            if index in (75, 150, 225, len(actions)):
                phases.append(dict(last_turn=index, start=start, end=post, ledger=phase))
                phase, start = fresh(), post
            prev = post
    if [p.reward for p in state] != result["scores"] or [p["cash"] for p in prev] != result["cash"]:
        raise ValueError("terminal scores differ from recorder")
    return dict(seed=result["seed"], v6_seat=seat, original=None, rerun=result, opening=initial, terminal=prev,
                ledger=total, phases=phases), trace_steps


def contributions(cash, seat):
    groups = {
        "net_tree": ["tree_yield", "plant", "seize", "tree_collision"],
        "net_deposits": ["bank", "bank_from_enemy"],
        "base_collisions": ["base_collision"],
        "recruitment": ["hire_collector", "hire_planter"],
        "rounding": ["rounding"],
    }
    if (set(cash[0]) | set(cash[1])) - set().union(*groups.values()):
        raise ValueError("unclassified cash transaction")
    return {name:sum(cash[seat][key] - cash[1-seat][key] for key in keys) for name, keys in groups.items()}


def historic():
    originals = {}
    for name in ("v6-confirmation-response", "v6-response-followup"):
        path = ROOT / "analysis" / "arena" / f"{name}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["seed"] in SEEDS:
                shas = [p["sha256"] for p in row["players"]]
                if sorted(shas) != sorted((V6_SHA, V5_SHA)):
                    raise ValueError("different original strategies")
                key = (row["seed"], shas.index(V6_SHA))
                if key in originals:
                    raise ValueError("duplicate original")
                originals[key] = dict(source=str(path.relative_to(ROOT)), record=row)
    return originals


def rounded(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {k:rounded(v) for k,v in value.items()}
    if isinstance(value, list):
        return [rounded(x) for x in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, help="per-step accounting output directory, defaults to --raw-dir")
    parser.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    ledger_dir = args.ledger_dir or args.raw_dir
    ledger_dir.mkdir(parents=True, exist_ok=True)
    limit_math_threads()
    configure_engine()
    histories = historic()
    summaries = []
    for (seed, seat), original in sorted(histories.items()):
        if seed not in args.seeds:
            continue
        archive = args.raw_dir / f"trace-{seed}-v6seat{seat}.json.gz"
        report, steps = replay(archive)
        for key in ("seed", "episode_steps", "engine_sha256", "map_sha256", "layout_sha256"):
            if report["rerun"][key] != original["record"][key]:
                raise ValueError(f"original and rerun differ at {key}")
        report["original"] = original
        for section in [report] + report["phases"]:
            section["v6_minus_v5"] = contributions(section["ledger"]["cash"], seat)
            start = section.get("opening", section.get("start"))
            finish = section.get("terminal", section.get("end"))
            lead_change = finish[seat]["cash"] - finish[1-seat]["cash"] - start[seat]["cash"] + start[1-seat]["cash"]
            check_near(sum(section["v6_minus_v5"].values()), lead_change, "cash lead decomposition")
        summaries.append(rounded(report) | {"original": original, "rerun": report["rerun"]})
        with gzip.open(ledger_dir / f"ledger-{seed}-v6seat{seat}.json.gz", "wt") as out:
            json.dump(rounded(steps), out, separators=(",", ":"))
    if len(summaries) != 2 * len(args.seeds):
        raise ValueError("expected both seats on each selected map")
    args.summary.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
