"""From verified turn 190, compare ten-turn interventions with paired live continuations."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from branch_replay import BRANCH, HERE, INPUT, OUT, ROOT, SNAPSHOT, compact, digest, prefix, step, write
from benchmark import configure_engine, limit_math_threads, load_strategy, observe, validate_actions
from close_replay import referee_events, restore

SEEDS = (18011, 38039, 62047)
MODES = ("control", "tree", "bank", "intercept")
PROXIES = ("artemis", "apollo")
BUDGET = dict(width=4, depth=2, opponent_replies=1, tail_plies=3)
FROZEN_SHA = "e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0"
RAW = ROOT.parent / "v7-data" / "branch-jobs"
CASH_KEYS = {"bank", "hire_collector", "hire_planter", "plant", "seize", "tree_yield",
             "rounding", "tree_collision", "base_collision"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_release():
    path = ROOT / "yan-v6.py"
    if sha(path) != FROZEN_SHA:
        raise ValueError("The frozen QQ-v6 release has changed")
    src = path.read_text()
    start = src.index("    if _S_COST[0] > 0.0:", src.index("def _search("))
    end = src.index("    stages =", start)
    src = src[:start] + "    most, D2, M = 4, min(D2, 2), 1\n" + src[end:]
    for old, new in (
        ("+ 17 * me) % (2 ** 31)", "+ 17 * me + _BRANCH_SEED) % (2 ** 31)"),
        ("for tick in range(1,15):", "for tick in range(1,4):"),
        ("_torch.set_num_threads(2)", "_torch.set_num_threads(1)"),
        ("return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT", "return False"),
    ):
        if src.count(old) != 1:
            raise ValueError(old)
        src = src.replace(old, new)
    m = load_strategy("branch_frozen", path, src)
    m._time = SimpleNamespace(perf_counter=lambda: 0.)
    m._BRANCH_SEED = 0
    m._S_MMAX = 1
    if not m._SEARCH_OK or not m._RX:
        raise RuntimeError("Both the frozen network and its simulated referee are required")
    return m


def initialize():
    global POLICY, BASE_MEMORY, BASE_FC, BASE_STATS, PREFIX, CFG, ORIGINAL, CHECKPOINT, OPP_MEM
    limit_math_threads()
    configure_engine()
    os.environ["CARBON_DEBUG"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ORIGINAL = json.loads(INPUT.read_text())
    CHECKPOINT = json.loads(SNAPSHOT.read_text())
    PREFIX = prefix(ORIGINAL)
    CFG = {k:v for k,v in ORIGINAL["configuration"].items() if k != "randomSeed"}
    POLICY = fixed_release()
    search, choose = POLICY._search, POLICY._reply_choice
    # Replay executed own moves into belief and style trackers. Skip discarded decision-only search.
    POLICY._search = lambda obs, cfg, est, base, *args: base
    POLICY._reply_choice = lambda obs, cfg, est, base, *args: base
    for obs, acts in PREFIX:
        POLICY.agent(obs[0], CFG)
        POLICY._ST["tr"].set_actions(obs[0]["players"], acts[0])
    POLICY._search, POLICY._reply_choice = search, choose
    BASE_MEMORY, BASE_FC, BASE_STATS = copy.deepcopy((POLICY._ST, POLICY._FC_HISTORY, POLICY._S_STATS))
    OPP_MEM = {}


def new_opponent(name, seed):
    from surrogate import make
    key = (name, seed)
    if key not in OPP_MEM:
        obj = make(name, seed, CFG)
        for obs, acts in PREFIX:
            obj.observe(obs[1], acts[1])
        OPP_MEM[key] = obj
    return copy.deepcopy(OPP_MEM[key])


def marker(state, ledger, counts):
    cash = [p[0] for p in state[0].observation.players]
    flow = [{k:round(v, 6) for k,v in sorted(acc.items())} for acc in ledger]
    return dict(cash=cash, margin=round(cash[0]-cash[1],2), deposits=[a["bank"] for a in ledger],
                receipts=list(counts), flow=flow)


def run(job):
    import numpy as np
    from zerosum_env.envs.carbon import idgen
    from lanes import plan
    proxy, seed, mode, repeat, stop = job
    live = mode != "recorded"
    key = f"{proxy}-{seed}-{mode}" + ("-repeat" if repeat else "") + (f"-to{stop}" if stop!=299 else "")
    random.seed(seed)
    np.random.seed(seed)
    own = POLICY
    own._BRANCH_SEED = seed
    own._ST, own._FC_HISTORY, own._S_STATS = copy.deepcopy((BASE_MEMORY, BASE_FC, BASE_STATS))
    opp = new_opponent(proxy,seed) if live else None
    if live:
        own._torch.set_num_threads(1)  # surrogate loader can change global Torch threads
    state, env = restore(CHECKPOINT["snapshot"], ORIGINAL["configuration"])
    start_cash = [p[0] for p in state[0].observation.players]
    action_log = []
    ledger = [Counter(), Counter()]
    counts = [0,0]
    marks = {}
    edits = []
    original_search = own._search
    stock = None
    def capture(obs, cfg, est, *args):
        nonlocal stock
        stock = est
        return original_search(obs, cfg, est, *args)
    own._search = capture
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        with (RAW / (key+".progress")).open("w") as stream:
            for t in range(BRANCH, stop):
                if live:
                    obs = [observe(state, i, t) for i in (0,1)]
                    # A policy's internal simulations never get to advance referee IDs.
                    ids = dict(idgen._id_dict)
                    own._ST["rng"] = np.random.RandomState(seed + 1000003*(t+1))
                    stock = None
                    base = own.agent(obs[0], CFG)
                    assert stock is not None
                    actual = (plan(obs[0], CFG, stock, base, mode)
                              if mode != "control" and t < BRANCH+10 else base)
                    if actual != base:
                        changes = {k:[base.get(k),actual.get(k)] for k in sorted(base.keys()|actual.keys())
                                   if base.get(k)!=actual.get(k)}
                        assert len(changes)<=2
                        edits.append([t, changes])
                    own._ST["tr"].set_actions(obs[0]["players"], actual)
                    idgen._id_dict.clear(); idgen._id_dict.update(ids)
                    rival = opp.act(obs[1])
                    idgen._id_dict.clear(); idgen._id_dict.update(ids)
                    decisions = [actual, rival]
                else:
                    decisions = ORIGINAL["actions"][t]
                for a,p in zip(decisions,state[0].observation.players):
                    validate_actions(a,p)
                action_log.append(decisions)
                events = []
                with referee_events(events):
                    step(state, env, decisions)
                for kind,side,amount,*_ in events:
                    if kind in CASH_KEYS:
                        ledger[side][kind] += amount
                    if kind == "bank":
                        counts[side]+=1
                done = any(s.status != "ACTIVE" for s in state)
                if t+1 in (200,210,stop) or done:
                    rec = marker(state, ledger, counts)
                    for i in (0,1):
                        if abs(rec["cash"][i]-start_cash[i]-sum(ledger[i].values()))>1e-6:
                            raise ValueError(f"Cash accounting failed at {key}/{t}")
                    marks[str(t+1)] = rec
                    stream.write(compact([t+1,rec])+"\n"); stream.flush()
                if done:
                    break
        final = dict(key=key, proxy=proxy, seed=seed, mode=mode, start_step=BRANCH, last_step=t+1,
                     initial_cash=start_cash, milestones=marks, edits=edits, rewards=[s.reward for s in state],
                     search={k:own._S_STATS[k] for k in ("searched","aborted","cands","depth","rolls")} if live else {},
                     actions_sha256=digest(action_log), terminal_board_sha256=digest(
                         [state[0].observation.full_carbon, state[0].observation.players,
                          state[0].observation.trees, idgen._id_dict]))
        write(RAW / (key+".json"), dict(final,actions=action_log))
        return final
    finally:
        own._search = original_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1,4), default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--proxies", nargs="+", choices=PROXIES, default=PROXIES)
    parser.add_argument("--stop", type=int, default=299)
    parser.add_argument("--out", type=Path, default=OUT/"branch-results.json")
    a = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"]="0"
        os.execv(sys.executable,[sys.executable,*sys.argv])
    assert BRANCH < a.stop <= 299
    os.environ["PYTHONHASHSEED"]="0"
    limit_math_threads()
    jobs = [(p,s,m,False,a.stop) for p,s,m in product(a.proxies,a.seeds,a.modes)]
    repeat = (a.proxies[0],a.seeds[0],a.modes[0],True,a.stop)
    jobs += [("original",0,"recorded",False,a.stop), repeat]
    with ProcessPoolExecutor(a.workers,initializer=initialize) as pool:
        futures = [pool.submit(run,j) for j in jobs]
        results = []
        for f in futures:
            row = f.result()
            print(row["key"],row["milestones"][str(row["last_step"])]["margin"],flush=True)
            results.append(row)
    for k in ("actions_sha256","terminal_board_sha256","milestones","search"):
        if results[0][k] != results[-1][k]:
            raise ValueError(f"Reproduction failed for {k}")
    write(a.out,dict(budget=BUDGET, release_sha256=FROZEN_SHA,
                     module_sha256={p.name:sha(p) for p in (HERE/"lanes.py", HERE/"surrogate.py", HERE/"branch_run.py",ROOT/"experiments/v6r/checkpoints/grove.py")},
                     verified_repetition=results[-1]["key"], maps=1, results=results[:-1]))


if __name__=="__main__":
    main()
