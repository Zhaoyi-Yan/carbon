"""Separate the last planter purchase and first safe delivery in the Master7sE replay."""
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

from branch_replay import OUT, configure_engine, limit_math_threads, new_game, step
from close_replay import archive_state, referee_events, restore
from starvation_replay import read_replay


def compare(replay):
    frames = replay["steps"]
    config = replay["configuration"]
    new_game(config["randomSeed"], len(frames))
    from zerosum_env.envs.carbon import idgen

    initial = dict(seats=[{**frame, "action": frame["action"] or {}} for frame in frames[0]],
                   id_counters=dict(idgen._id_dict))
    state, env = restore(initial, config)
    for tick in range(86):
        step(state, env, [frame["action"] or {} for frame in frames[tick + 1]])
    snapshot = archive_state(state)
    assert state[0].observation.players == frames[86][0]["observation"]["players"]
    assert state[0].observation.full_carbon == frames[86][0]["observation"]["full_carbon"]

    result = []
    for hire, deliver in product((True, False), (False, True)):
        state, env = restore(snapshot, config)
        ledger = Counter()
        marks = []
        for tick in range(86, 98):
            players = state[0].observation.players
            actions = [dict(frame["action"] or {}) for frame in frames[tick + 1]]
            if tick == 86:
                assert actions[1]["player-1-recrtCenter-1"] == "RECPLANTER"
                if not hire:
                    del actions[1]["player-1-recrtCenter-1"]
            else:
                assert not (actions[1].keys() & players[1][1].keys()), "new historical hire needs an ID mapping"
            if not hire:
                actions[1].pop("player-1-worker-18", None)
            if deliver and tick in (90, 91):
                actions[1]["player-1-worker-11"] = "DOWN"
            events = []
            with referee_events(events):
                step(state, env, actions)
            for event, seat, amount, *_ in events:
                if seat == 1:
                    ledger[event] += amount
            players = state[0].observation.players
            if hire and not deliver:
                assert players == frames[tick + 1][0]["observation"]["players"]
            if tick + 1 in (87, 89, 92, 95, 98):
                cash, _, workers, trees = players[1]
                marks.append(dict(turn=tick + 1, cash=cash, workers=len(workers), trees=len(trees),
                                  opponent_cash=players[0][0], own_cargo=round(sum(w[1] for w in workers.values()), 2),
                                  gross_bank=round(ledger["bank"] + ledger["base_collision"], 2),
                                  tree_yield=round(ledger["tree_yield"], 2), plant=round(ledger["plant"], 2),
                                  seize=round(ledger["seize"], 2)))
        result.append(dict(hire_planter=hire, take_delivery=deliver, marks=marks))
    return dict(match="4063a34afedb6d4767a9118e1dee9e5d/1", seat=1,
                method="Official referee from the original turn 86. Replay both sides' other recorded moves through observation 98. Remove uncreated worker actions when a purchase is omitted. No other own recruit instruction occurs during the tested window.", arms=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    limit_math_threads()
    configure_engine()
    (OUT / "starvation-choices.json").write_text(json.dumps(compare(read_replay(args.replay)), indent=2) + "\n")
