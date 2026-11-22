"""One-step economic counterfactual on a recorded board, with the referee and fixed opponent action."""
from collections import Counter
from functools import lru_cache
from unittest.mock import patch
import sys
from trace_economy import CASH_RULES, CARGO_RULES, rule_lines


@lru_cache(None)
def rules():
    from zerosum_env.envs.carbon import helpers
    return rule_lines(helpers.Board.next,CASH_RULES),rule_lines(helpers.Board.next,CARGO_RULES)


def turn(observation, config, actions, side):
    from zerosum_env.envs.carbon import helpers,idgen
    from arena import validate_actions
    cash_lines,cargo_lines=rules()
    full=dict(observation)
    full["carbon"]=full["full_carbon"]
    for i,a in enumerate(actions):
        validate_actions(a,full["players"][i])
    entries=Counter()
    code=helpers.Board.next.__code__
    def write_cash(player,attr,value):
        frame=sys._getframe(1)
        if attr=="_cash" and frame.f_code is code and player.id==side:
            if cash_lines[frame.f_lineno]=="bank":
                entries["base_deposits"]+=value-player.cash
        return object.__setattr__(player,attr,value)

    def write_cargo(worker,attr,value):
        frame=sys._getframe(1)
        if attr=="_carbon" and frame.f_code is code and worker.player_id==side:
            kind=cargo_lines[frame.f_lineno]
            if kind=="harvest":
                entries["ground_harvest"]+=value-worker.carbon
        return object.__setattr__(worker,attr,value)

    # The recording starts midgame. Unique fresh identities do not affect a one-turn payroll.
    counters={f"player-{s}-{role}-":10000 for s in (0,1) for role in ("tree","worker","recrtCenter")}
    with patch.dict(idgen._id_dict,counters,clear=True),patch.object(helpers.Player,"__setattr__",write_cash),patch.object(helpers.Worker,"__setattr__",write_cargo):
        after=helpers.Board(full,config,actions).next().observation["players"]
    before=full["players"]
    entries.update(cash=after[side][0]-before[side][0],enemy_cash=after[1-side][0]-before[1-side][0],
                   cargo=sum(w[1] for w in after[side][2].values())-sum(w[1] for w in before[side][2].values()),
                   worker_change=len(after[side][2])-len(before[side][2]))
    return entries


def compare(frame, following, config, side, root, proposal):
    opponent=following[1-side]["action"] or {}
    observed=following[0]["observation"]["players"]
    teacher=[s["action"] or {} for s in following]
    full=frame[0]["observation"]
    known=turn(full,config,teacher,side)
    for key,i in (("cash",side),("enemy_cash",1-side)):
        if abs(known[key]-observed[i][0]+full["players"][i][0])>.005:
            raise ValueError("recorded economic branch disagrees with official referee")
    recorded_cargo = sum(w[1] for w in observed[side][2].values())-sum(w[1] for w in full["players"][side][2].values())
    if abs(known["cargo"]-recorded_cargo)>1e-4 or known["worker_change"]!=len(observed[side][2])-len(full["players"][side][2]):
        raise ValueError("recorded worker holdings disagree with official referee")
    branches=[]
    for act in (root,proposal):
        actions=[opponent,opponent]
        actions[side]=act
        branches.append(turn(full,config,actions,side))
    return {k:round(branches[1][k]-branches[0][k],6) for k in ("ground_harvest","base_deposits","cash","enemy_cash","cargo","worker_change")}


def validate_recorded(source, raw):
    import gzip
    import json
    from benchmark import configure_engine
    configure_engine()
    max_error=Counter()
    count=0
    for game in source["games"]:
        replay=json.loads(gzip.decompress((raw/game["file"]).read_bytes()))
        side=int(replay["rewards"][1]>replay["rewards"][0])
        for row in game["changes"]:
            t=row["t"]
            frame,following=replay["steps"][t:t+2]
            full=frame[0]["observation"];post=following[0]["observation"]["players"]
            predicted=turn(full,replay["configuration"],[p["action"] or {} for p in following],side)
            own=full["players"][side];enemy=full["players"][1-side]
            actual=dict(cash=post[side][0]-own[0],enemy_cash=post[1-side][0]-enemy[0],
                        cargo=sum(w[1] for w in post[side][2].values())-sum(w[1] for w in own[2].values()),
                        worker_change=len(post[side][2])-len(own[2]))
            for key,target in actual.items():
                err=abs(predicted[key]-target)
                max_error[key]=max(err,max_error[key])
                if err>1e-4:raise ValueError(f"recorded mismatch on {game['file']} at {t}: {key} {err}")
            count+=1
    return dict(teacher_steps=count,max_abs_error=dict(max_error))


def compact(source):
    games=[]
    for game in source["games"]:
        totals=Counter()
        for row in game["changes"]:
            totals.update(row["economy"])
        games.append(dict(file=game["file"],roots=game["roots"],proposed=game.get("proposed",0),
                          one_step_delta={k:round(v,4) for k,v in totals.items()}))
    return dict(method="On each recorded board, run the official next-step rule on the recorded teacher joint action, then on the frozen v6 root and its one-worker alternative. The opponent always plays their recorded next action. Differences are alternate minus frozen root. Samples are independent counterfactual branches from recorded board snapshots. Coverage assumes a free search slot. Aggregate one-step receipts do not measure final cash.",
                preserve_bank=source.get("preserve_bank",False),root_sha256=source["root_sha256"],games=games,all=source["all"],
                by_phase_agreement=source["phases"],
                one_step_delta=source["one_step_delta"],by_phase_one_step_delta=source["by_phase_one_step_delta"])


if __name__=="__main__":
    import argparse
    import json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--replays",type=Path,help="independently check the unmodified published actions and balances")
    a=p.parse_args();source=json.loads(a.source.read_text());summary=compact(source)
    if a.replays:summary["referee_check"]=validate_recorded(source,a.replays)
    a.output.write_text(json.dumps(summary,indent=2)+"\n")
