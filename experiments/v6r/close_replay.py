"""Rebuild both recorded panels and retain exact original endgame states and transactions."""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,limit_math_threads,new_game
from arena import engine_fingerprint
from trace_economy import CASH_RULES,CARGO_RULES,rule_lines
from harvest import _HV_DIST as DIST, _HV_MOVES as MOVES, _HV_COMMANDS as CMDS

PANELS=(("canopy","v6r-qual-02-canopy.jsonl"),("zenith","v6r-qual-03-zenith.jsonl"))
MILESTONES=(250,274,275,285)


def archive_state(state):
    from zerosum_env.envs.carbon import idgen
    return dict(seats=[dict(observation=copy.deepcopy(dict(s.observation)),status=s.status,reward=s.reward,action=dict(s.action)) for s in state],
                id_counters=dict(idgen._id_dict))


def restore(data,configuration):
    from zerosum_env import utils
    from zerosum_env.envs.carbon import idgen
    idgen.reset();idgen._id_dict.update(data["id_counters"])
    return [SimpleNamespace(observation=utils.structify(copy.deepcopy(s["observation"])),status=s["status"],reward=s["reward"],action=dict(s["action"])) for s in data["seats"]],SimpleNamespace(configuration=utils.structify(configuration))


def holdings(players,t):
    out=[]
    for p in players:
        homes=list(p[1].values());cc=[]
        for k,(pos,v,role) in p[2].items():
            if role=="COLLECTOR" and v>0:
                d=min(DIST[pos][h] for h in homes)
                cc.append(dict(id=k,pos=pos,cargo=v,minimum_steps=d,reachable=d<=299-t))
        out.append(dict(cash=p[0],cargo=sum(x["cargo"] for x in cc),reachable_cargo=sum(x["cargo"] for x in cc if x["reachable"]),carriers=cc))
    return out


def state_hash(state):
    return hashlib.sha256(json.dumps(archive_state(state),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def referee_events(events):
    from zerosum_env.envs.carbon import helpers
    code=helpers.Board.next.__code__
    cash_lines=rule_lines(helpers.Board.next,CASH_RULES)
    cargo_lines=rule_lines(helpers.Board.next,CARGO_RULES)
    old_delete=helpers.Board._delete_worker
    def on_cash(p,attr,v):
        frame=sys._getframe(1)
        if attr=="_cash" and frame.f_code is code:
            key=cash_lines[frame.f_lineno];value=v-p.cash
            if key=="bank" and value>0:
                fields=frame.f_locals;worker=fields["worker"]
                events.append([key,p.id,value,worker.id,worker.player_id,fields["recrtCenter"].id])
            elif key in ("hire_collector","hire_planter","plant","seize","tree_yield","rounding","tree_collision","base_collision"):
                events.append([key,p.id,value])
        return object.__setattr__(p,attr,v)
    def on_worker(w,attr,v):
        frame=sys._getframe(1)
        if attr=="_carbon" and frame.f_code is code:
            key=cargo_lines[frame.f_lineno];d=v-w.carbon
            if key=="loot" and d:
                other=frame.f_locals["worker"];events.append([key,w.player_id,d,w.id,other.id,other.player_id])
            elif key in ("harvest","enemy_tree") and d:events.append([key,w.player_id,d,w.id])
        return object.__setattr__(w,attr,v)
    def death(board,w):
        if sys._getframe(1).f_code is code:
            events.append(["death",w.player_id,w.carbon,w.id,str(w.occupation),w.position.to_index(15)])
        return old_delete(board,w)
    stack=ExitStack()
    stack.enter_context(patch.object(helpers.Player,"__setattr__",on_cash))
    stack.enter_context(patch.object(helpers.Worker,"__setattr__",on_worker))
    stack.enter_context(patch.object(helpers.Board,"_delete_worker",death))
    return stack


def read_records():
    for candidate,name in PANELS:
        with (ROOT/"analysis"/name).open() as f:
            for line in f:
                r=json.loads(line);yield candidate,r


def replay_all(out,snapshots):
    from zerosum_env.envs.carbon import carbon
    cases=[];states={};configuration=None
    for candidate,r in read_records():
        seat=next(i for i,p in enumerate(r["players"]) if p["name"]==candidate);key=f"{candidate}/{r['seed']}/{seat}"
        if r["status"]!="completed" or r["turns"]!=299 or engine_fingerprint(carbon)!=r["engine_sha256"]:raise ValueError(key)
        state,env=new_game(r["seed"],300)
        if configuration is None:configuration=dict(env.configuration);configuration.pop("randomSeed")
        names=[p["name"] for p in r["players"]]
        saved={};checks={};nverified=0;cash_flow=[Counter(),Counter()];deaths=[[],[]];receipts=[[],[]];births=[[],[]]
        previous=None;final_hash=None
        for t,(actions,expected_cash) in enumerate(r["actions_and_post_cash"]):
            pre=state[0].observation.players
            if t in MILESTONES:
                saved[str(t)]=archive_state(state);checks[str(t)]=holdings(pre,t)
            for p,a in zip(state,actions):p.action=a
            events=[]
            if t>=274:
                with referee_events(events):carbon.compute_next_board(state,env)
            else:carbon.compute_next_board(state,env)
            post=state[0].observation.players
            if [p[0] for p in post]!=expected_cash:raise ValueError(f"cash changed at {key}/{t}")
            nverified+=1
            if t>=274:
                for i,pl in enumerate(post):
                    for uid in pl[2].keys()-pre[i][2].keys():births[i].append(dict(turn=t,id=uid,role=pl[2][uid][2],pos=pl[2][uid][0]))
                for e in events:
                    kind,i,v=e[:3]
                    if kind=="bank":
                        _,_,_,uid,owner,base=e
                        opportunity=[]
                        if owner==i and previous and uid in previous[i][2] and uid in pre[i][2]:
                            prior_pos,prior_cargo,_=previous[i][2][uid];approach=pre[i][2][uid][0]
                            all_bases={q for player in previous for q in player[1].values()}
                            if approach not in all_bases:
                                for wid,(p,c,role) in previous[1-i][2].items():
                                    if role=="COLLECTOR" and c<prior_cargo and DIST[p][approach]<=1:
                                        opportunity.append(wid)
                        receipts[i].append(dict(turn=t,worker=uid,owner=owner,amount=v,base=base,
                                                prior_approach_lighter_interceptors=opportunity))
                    if kind=="death":
                        deaths[i].append(dict(turn=t,id=e[3],role=e[4],cargo=v,pos=e[5]))
                    elif kind not in ("harvest","enemy_tree","loot"):
                        cash_flow[i][kind]+=v
            previous=pre
        finish=state[0].observation.players
        if [s.reward for s in state]!=r["scores"] or [p[0] for p in finish]!=r['cash']:raise ValueError(key)
        checks["299"]=holdings(finish,299)
        for i in (0,1):
            opening=checks['274'][i]['cash']
            if abs(sum(cash_flow[i].values())+opening-finish[i][0])>1e-6:raise ValueError("cash ledger")
        final_hash=state_hash(state)
        # Replay from every saved state, including unit-ID counters, with the archived decisions.
        for st,snap in saved.items():
            branch,env2=restore(snap,dict(configuration,randomSeed=r["seed"]))
            for action,paid in r['actions_and_post_cash'][int(st):]:
                for p,a in zip(branch,action):p.action=a
                carbon.compute_next_board(branch,env2)
                if [p[0] for p in branch[0].observation.players]!=paid:raise ValueError("restored trajectory")
            if state_hash(branch)!=final_hash:raise ValueError("restored terminal board differs")
        states[key]=saved
        margin=r['cash'][seat]-r['cash'][1-seat]
        cases.append(dict(key=key,candidate=candidate,seed=r["seed"],seat=seat,players=names,sha256=[p["sha256"] for p in r["players"]],
                          layout=r["layout_sha256"],score=r["scores"],margin=round(margin,2),selected=margin<0 or abs(margin)<=350,
                          verified_steps=nverified,terminal_sha256=final_hash,milestones=checks,cash25=cash_flow,
                          deaths25=deaths,hires25=births,receipts25=receipts))
        print(key,"verified",flush=True)
    snapshots.write_text(json.dumps(dict(schema=1,engine_sha256=engine_fingerprint(carbon),source_sha256={f:hashlib.sha256((ROOT/"analysis"/f).read_bytes()).hexdigest() for _,f in PANELS},configuration=configuration,states=states),separators=(",",":"))+"\n")
    out.write_text(json.dumps(dict(cases=cases),separators=(",",":"))+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=HERE/"close-ledger.json");p.add_argument("--snapshots",type=Path,default=HERE/"close-states.json")
    a=p.parse_args();limit_math_threads();configure_engine();replay_all(a.out,a.snapshots)
