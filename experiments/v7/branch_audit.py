"""Pack played actions and independently replay them against saved outcome and cash ledgers."""
import argparse
import gzip
import json
from pathlib import Path

from branch_replay import BRANCH, INPUT, OUT, SNAPSHOT, compact, configure_engine, digest, step, write
from branch_run import CASH_KEYS, RAW, marker
from close_replay import referee_events, restore
from collections import Counter

PACK = OUT/"branch-actions.json.gz"


def pack(rows, directory, path):
    logs = {r["key"]:json.loads((directory/(r["key"]+".json")).read_text())["actions"] for r in rows}
    path.write_bytes(gzip.compress(compact(logs).encode(),compresslevel=9,mtime=0))


def audit(rows, path):
    from zerosum_env.envs.carbon import idgen
    checkpoint = json.loads(SNAPSHOT.read_text())
    logs = json.loads(gzip.decompress(path.read_bytes()))
    first = {}
    total = 0
    for r in rows:
        state, env = restore(checkpoint["snapshot"],checkpoint["configuration"])
        key = r["key"]
        plays = logs[key]
        assert len(plays) == r["last_step"]-BRANCH and digest(plays)==r["actions_sha256"], key
        flow = [Counter(),Counter()]
        counts = [0,0]
        for turn,actions in enumerate(plays,BRANCH+1):
            events = []
            with referee_events(events):
                step(state,env,actions)
            for kind,side,amount,*_ in events:
                if kind in CASH_KEYS:
                    flow[side][kind]+=amount
                if kind=="bank":
                    counts[side]+=1
            if str(turn) in r["milestones"]:
                assert marker(state,flow,counts)==r["milestones"][str(turn)], (key,turn)
            total+=1
        b=state[0].observation
        assert digest([b.full_carbon,b.players,b.trees,idgen._id_dict])==r["terminal_board_sha256"], key
        assert [s.reward for s in state]==r["rewards"], key
        if r["mode"]!="recorded":
            group=(r["proxy"],r["seed"])
            first.setdefault(group,[]).append(plays[0][1])
    for group,moves in first.items():
        if any(a!=moves[0] for a in moves):
            raise ValueError(f"Proxy saw something different across arms before the fork: {group}")
    return dict(verified_trajectories=len(rows),verified_transitions=total,matching_first_opponent_actions_groups=len(first),
                compared="Actual referee bank, receipts and all cash flow at all horizons; final full stock, players, trees, reward and ID counters.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-from",type=Path,help="First pack raw per-game actions from this directory.")
    p.add_argument("--input",type=Path,default=OUT/"branch-results.json")
    p.add_argument("--actions",type=Path,default=PACK)
    p.add_argument("--out",type=Path,default=OUT/"branch-audit.json")
    args=p.parse_args()
    configure_engine()
    rows=json.loads(args.input.read_text())["results"]
    if args.pack_from:
        pack(rows,args.pack_from,args.actions)
    result=audit(rows,args.actions)
    write(args.out,result)
    print(result)
