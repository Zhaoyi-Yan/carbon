"""Paired opening-economy ablation on the four declared development layouts."""
import copy
import hashlib
import json
import os
from collections import Counter
from statistics import mean

from branch_replay import HERE, OUT, ROOT, compact, digest, step, write
from branch_run import fixed_release, sha
from benchmark import configure_engine, limit_math_threads, new_game, observe, validate_actions
from close_replay import archive_state, referee_events, restore

MODES=("incumbent","mixed_now","double_reserve")
MARKS=(10,20,40,60,100,200,299)
OUTFILE=OUT/"opening-probe.json"


def load_policy():
    m=fixed_release()
    m._torch.set_num_threads(1)
    def capture(obs,cfg,est,action,lw,lb,*rest):
        m.base_logits=lb
        return action
    m._search=capture
    m._reply_choice=lambda obs,cfg,est,action,*rest:action
    return m


def variant(mode,obs,cfg,act,logits,ever_tree,m):
    if mode=="incumbent":
        return act
    p=obs["players"][obs["player"]]; bases=list(p[1].items())
    if obs["step"]==0:
        if mode=="double_reserve":
            return {b:"RECCOLLECTOR" for b,_ in bases}
        # The incumbent's joint role preference before enemy births are observed.
        (ba,pa),(bb,pb)=bases
        first=logits[pa,2]+logits[pb,1]>=logits[pa,1]+logits[pb,2]
        return {ba:"RECPLANTER" if first else "RECCOLLECTOR",bb:"RECCOLLECTOR" if first else "RECPLANTER"}
    if mode!="double_reserve" or ever_tree or obs["step"]>=40:
        return act
    action=dict(act)
    reserve=cfg["plantCost"]
    land={m.move(pos,m.WIDX.get(action.get(uid),0)) for uid,(pos,_,_) in p[2].items()}
    if not any(w[2]=="PLANTER" for w in p[2].values()):
        # Save toward one planter plus the game's protected first-tree cost.
        for uid,_ in bases:
            action.pop(uid,None)
        choices=[(float(logits[pos,2]),uid) for uid,pos in bases if pos not in land]
        if p[0]>=cfg["recPlanterCost"]+reserve and len(p[2])<cfg["workerLimit"] and choices:
            action[max(choices)[1]]="RECPLANTER"
    else:
        cash=p[0]
        for uid,pos in bases:
            if uid in action:
                charge=cfg["recPlanterCost" if action[uid]=="RECPLANTER" else "recCollectorCost"]
                if cash-charge<reserve:action.pop(uid)
                else:cash-=charge
    return action


def map_id(state):
    o=state[0].observation
    return hashlib.sha256(json.dumps([15,o.full_carbon,[sorted(p[1].values()) for p in o.players]],separators=(",",":")).encode()).hexdigest()


def create(seed,mode,seat):
    state,env=new_game(seed,300)
    cfg=dict(env.configuration);cfg.pop("randomSeed")
    return dict(seed=seed,mode=mode,seat=seat,config=dict(env.configuration),visible_cfg=cfg,layout=map_id(state),
                snapshot=archive_state(state),memory=[({},{}),({},{})],turn=0,done=False,marks={},
                first=[{},{}],ledger=[Counter(),Counter()],edited=[],joint_first=None)


def advance(g,end,m):
    import numpy as np
    from zerosum_env.envs.carbon import idgen
    if g["done"] or g["turn"]>=end:return
    state,env=restore(g["snapshot"],g["config"])
    events=[]
    with referee_events(events):
        for t in range(g["turn"],end):
            ids=dict(idgen._id_dict)
            joint=[]
            for side in (0,1):
                o=observe(state,side,t); cfg=g["visible_cfg"]
                m._ST,m._FC_HISTORY=g["memory"][side]
                m._ST["rng"]=np.random.RandomState((g["seed"]+1000003*(t+1)+31*side)%(2**31))
                a=m.agent(o,cfg)
                b=variant(g["mode"] if side==g["seat"] else "incumbent",o,cfg,a,m.base_logits,"tree" in g["first"][side],m)
                if a!=b:g["edited"].append([t,{k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}])
                validate_actions(b,o["players"][side]);joint.append(b)
                m._ST["tr"].set_actions(o["players"],b)
                idgen._id_dict.clear();idgen._id_dict.update(ids)
            if t==0:g["joint_first"]=joint
            events.clear()
            step(state,env,joint)
            for kind,s,amount,*rest in events:
                if kind in ("bank","base_collision","plant","seize","tree_yield","hire_collector","hire_planter"):
                    g["ledger"][s][kind]+=amount
                    if kind in ("bank","base_collision") and amount>0:
                        g["first"][s].setdefault("deposit",t+1)
            g["turn"]=t+1
            g["done"]=any(p.status!="ACTIVE" for p in state) or t+1==299
            ps=state[0].observation.players
            for s,p in enumerate(ps):
                first=g["first"][s]
                for _,_,role in p[2].values():first.setdefault(role.lower(),t+1)
                if p[3]:first.setdefault("tree",t+1)
            if t+1 in MARKS or g["done"]:
                g["marks"][str(t+1)]=[dict(cash=p[0],workers=len(p[2]),collectors=sum(w[2]=="COLLECTOR" for w in p[2].values()),
                                            trees=len(p[3]),inbank=round(g["ledger"][s]["bank"]+g["ledger"][s]["base_collision"],2),
                                            tree_yield=round(g["ledger"][s]["tree_yield"],2),
                                            tree_cost=round(-g["ledger"][s]["plant"]-g["ledger"][s]["seize"],2),
                                            cargo=round(sum(w[1] for w in p[2].values()),2),reward=state[s].reward) for s,p in enumerate(ps)]
            if g["done"]:break
    g["snapshot"]=archive_state(state)


def output(games,stage):
    records=[]
    for g in games:
        fields=("seed","mode","layout","turn","done","marks","first","edited","joint_first")
        for side in ((0,1) if g["mode"]=="incumbent" else (g["seat"],)):
            records.append(dict(**{k:g[k] for k in fields},seat=side))
    controls={(r["seed"],r["seat"]):r for r in records if r["mode"]=="incumbent"}
    groups={}
    for mode in MODES:
        metrics={}
        for t in MARKS:
            if t>stage:continue
            values=[]
            for r in records:
                if r["mode"]!=mode:continue
                c=controls[r["seed"],r["seat"]];s=r["seat"]
                board=r["marks"][str(min(t,r["turn"]))];orig=c["marks"][str(min(t,c["turn"]))]
                m=round(board[s]["cash"]-board[1-s]["cash"],2);cm=round(orig[s]["cash"]-orig[1-s]["cash"],2)
                values.append(dict(seat=s,margin=m,margin_delta=round(m-cm,2),win=board[s]["reward"]>board[1-s]["reward"],
                                   **{k:board[s][k]-orig[s][k] for k in ("cash","workers","trees","inbank","tree_yield")}))
            metrics[t]=dict(wins=sum(x["win"] for x in values),wins_by_seat=[sum(x["win"] for x in values if x["seat"]==s) for s in (0,1)],
                            mean_margin=round(mean(x["margin"] for x in values),2),
                            paired_improved=sum(x["margin_delta"]>0 for x in values),
                            mean_paired_delta={k:round(mean(x[k] for x in values),2) for k in ("margin_delta","cash","workers","trees","inbank","tree_yield")})
        groups[mode]=metrics
    write(OUTFILE,dict(release_sha256=sha(ROOT/"yan-v6.py"),probe_sha256=sha(HERE/"opening_probe.py"),stage=stage,
                       base="QQ-v6 root against itself; no forward search or opponent reply rerank; policy RNG shared by layout, seat and turn.",
                       arms={"incumbent":"Observe at step zero and recruit on step one with rival role information.",
                             "mixed_now":"Immediate collector and planter; role location assigned using root joint logits.",
                             "double_reserve":"Two immediate collectors. Until the first tree or turn 40, buy the first planter only with its planting budget intact at a free depot, and hold the planting budget against subsequent hires."},
                       note="One incumbent-vs-incumbent game per layout supplies both seat perspectives. Remaining arms play separate paired-seat games. Intermediate wins are cash leads, final wins use referee rewards.",
                       summary=groups,records=records))


def main():
    os.environ["CARBON_DEBUG"]="1";limit_math_threads();configure_engine()
    seeds=json.loads((HERE/"panels.json").read_text())["development"]
    assert len(set(seeds))==4
    model=load_policy()
    games=[]
    for seed in seeds:
        for mode in MODES:
            for s in ((0,) if mode=="incumbent" else (0,1)):
                games.append(create(seed,mode,s))
    assert len(set(g["layout"] for g in games))==4
    for stage in (60,299):
        for g in games:advance(g,stage,model)
        output(games,stage);print("finished through",stage,flush=True)


if __name__=="__main__":main()
