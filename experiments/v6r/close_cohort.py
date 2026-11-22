"""Follow carbon already on carriers at t=274 through deposits, captures, taxes, spills and the deadline."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch
from close_replay import HERE,DIST,CMDS,MOVES,holdings,configure_engine,limit_math_threads,restore,read_records
from trace_economy import CASH_RULES,CARGO_RULES,rule_lines


def track(record,snapshot,config):
    from zerosum_env.envs.carbon import helpers,carbon
    st,env=restore(snapshot,config)
    fractions={};start=[0.,0.];out=[Counter(),Counter()];history={};deadline={};cleared={}
    for side,p in enumerate(st[0].observation.players):
        for uid,(_,cargo,_) in p[2].items():
            fractions[uid]=[cargo if i==side else 0. for i in (0,1)]
            start[side]+=cargo
    code=helpers.Board.next.__code__;cash_rules=rule_lines(helpers.Board.next,CASH_RULES);worker_rules=rule_lines(helpers.Board.next,CARGO_RULES)
    old_delete=helpers.Board._delete_worker
    def dispose(uid,key,share=1.):
        z=fractions.get(uid)
        if not z:return
        for i in (0,1):
            moved=z[i]*share;z[i]-=moved;out[i][key]+=moved
    def write_cash(p,k,v):
        f=sys._getframe(1)
        if k=="_cash" and f.f_code is code:
            name=cash_rules[f.f_lineno]
            if name in ("bank","tree_collision","base_collision"):
                w=f.f_locals["worker"]
                # Both panels have zero smelt fees. Unload credits the depot owner, regardless of carrier.
                dispose(w.id,name+"_seat"+str(p.id))
                if name=="bank" and w.carbon>0:cleared[w.id]=t
        return object.__setattr__(p,k,v)
    def write_worker(w,k,v):
        f=sys._getframe(1)
        if k=="_carbon" and f.f_code is code:
            name=worker_rules[f.f_lineno]
            if name=="loot":
                other=f.f_locals["worker"];src=fractions.pop(other.id,[0.,0.]);dest=fractions.setdefault(w.id,[0.,0.])
                for i in (0,1):dest[i]+=src[i]
            if name in ("enemy_tree","move") and w.carbon>0 and v<w.carbon:
                key="enemy_tree_seat"+str(1-w.player_id) if name=="enemy_tree" else "move_loss"
                dispose(w.id,key,(w.carbon-v)/w.carbon)
        return object.__setattr__(w,k,v)
    def delete(board,w):
        if sys._getframe(1).f_code is code:
            dispose(w.id,"unbanked_spill_or_death");fractions.pop(w.id,None)
        return old_delete(board,w)
    assert config["smeltCost"]==0 and config["moveCost"]==0
    with ExitStack() as stack:
        stack.enter_context(patch.object(helpers.Player,"__setattr__",write_cash))
        stack.enter_context(patch.object(helpers.Worker,"__setattr__",write_worker))
        stack.enter_context(patch.object(helpers.Board,"_delete_worker",delete))
        for t,(acts,expected) in enumerate(record["actions_and_post_cash"][274:],274):
            pl=st[0].observation.players
            if t in (290,294,297,298):deadline[str(t)]=holdings(pl,t)
            for side,p in enumerate(pl):
                homes=list(p[1].values())
                for uid,(pos,v,role) in p[2].items():
                    if role!="COLLECTOR":continue
                    if not v:
                        cleared[uid]=t
                        continue
                    d=min(DIST[pos][h] for h in homes);a=acts[side].get(uid);q=MOVES[pos][CMDS.index(a)]
                    dst=min(DIST[q][h] for h in homes)
                    history.setdefault(uid,[]).append(dict(step=t,cargo=round(v,4),distance=d,remaining=299-t,
                                                          chosen=a,post_move_distance=dst))
            for obj,action in zip(st,acts):obj.action=action
            carbon.compute_next_board(st,env)
            if [p[0] for p in st[0].observation.players]!=expected:raise ValueError("cash trajectory differs")
    loaded=[[],[]]
    for side,p in enumerate(st[0].observation.players):
        for uid,(pos,v,role) in p[2].items():
            if uid in fractions:
                if sum(fractions[uid])>v+1e-6:raise ValueError("tag exceeds carrier load")
                dispose(uid,"still_carried_seat"+str(side))
            if role=="COLLECTOR" and v>0:
                trip=[v for v in history.get(uid,[]) if v["step"]>cleared.get(uid,273)]
                feasible=[v for v in trip if v["distance"]<=v["remaining"]]
                loaded[side].append(dict(id=uid,cargo=round(v,4),first_carried_on_final_trip=(trip[0] if trip else None),last_move_with_reachable_carried_stock=(feasible[-1] if feasible else None)))
    for side in (0,1):
        if abs(sum(out[side].values())-start[side])>1e-5:raise ValueError("missing original carbon")
    return dict(initial_carbon=[round(v,4) for v in start],disposition=[{k:round(v,4) for k,v in c.items() if v>1e-7} for c in out],
                late_reachability=deadline,terminal_workers=loaded)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--snapshots",type=Path,default=HERE/"close-states.json");p.add_argument("--out",type=Path,default=HERE/"close-cohort.json")
    a=p.parse_args();configure_engine();limit_math_threads();sn=json.loads(a.snapshots.read_text());cases={}
    for name,r in read_records():
        s=next(i for i,p in enumerate(r["players"]) if p["name"]==name);key=f"{name}/{r['seed']}/{s}"
        cases[key]=track(r,sn["states"][key]["274"],dict(sn["configuration"],randomSeed=r["seed"]))
    a.out.write_text(json.dumps(dict(method="Initially carried carbon at step 274 is tracked proportionally within each worker through capture, deposits, tree absorption and deaths. If it spills onto a ground cell, attribution stops at the spill. Probability-free reachability is toroidal Manhattan distance to the nearest own depot.",
                                 cases=cases),separators=(",",":"))+"\n")
