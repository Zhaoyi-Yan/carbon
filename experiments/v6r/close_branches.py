"""Replay the final 25 decisions under two public-information deviations with literal archived opponent actions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from close_replay import CMDS,DIST,HERE,MOVES,configure_engine,limit_math_threads,read_records,restore,referee_events


def endpoint(pos,action):
    return MOVES[pos][CMDS.index(action)]


def safe_options(own,rival,original,uid):
    pos,load,_=own[2][uid]
    reserved={endpoint(p,original.get(k)) for k,(p,_,_) in own[2].items() if k!=uid}
    reserved.update(p for k,p in own[1].items() if k in original)
    enemy_bases=set(rival[1].values())
    threats=[p for p,v,k in rival[2].values() if k=="COLLECTOR" and v<=load]
    out=[]
    for a,q in enumerate(MOVES[pos]):
        if q in reserved or q in enemy_bases:continue
        distance=min((DIST[q][p] for p in threats),default=15)
        if distance<=1:continue
        out.append((a,q,distance))
    return out


def early_bank(pl,s,t,original):
    own,rival=pl[s],pl[1-s];homes=list(own[1].values());left=299-t
    best=None
    for uid,(p,v,role) in own[2].items():
        if role!="COLLECTOR" or v<70:continue
        home=lambda q:min(DIST[q][h] for h in homes)
        d=home(p);slack=left-d
        if not (d>0 and 0<=slack<=6):continue
        if home(endpoint(p,original.get(uid)))<d:continue
        options=[(home(q),-safe,a) for a,q,safe in safe_options(own,rival,original,uid) if home(q)<d]
        if not options:continue
        a=min(options)[2];item=(v/(1+slack),uid,a,round(v,2),d)
        if best is None or item>best:best=item
    if best:return (best[1],best[2],dict(cargo=best[3],home_distance=best[4]))
    return None


def preblock(pl,s,t,original):
    own,rival=pl[s],pl[1-s];homes=set(rival[1].values());last=299-t
    best=None
    for foe,(p,load,kind) in rival[2].items():
        if kind!="COLLECTOR" or load<70:continue
        d=min(DIST[p][h] for h in homes)
        if not 2<=d<=min(last,6):continue
        # If two depots tie, the action cannot assume the carrier chooses either particular one.
        choices=[h for h in homes if DIST[p][h]==d]
        if len(choices)!=1:continue
        approach=[q for q in MOVES[choices[0]][1:] if DIST[p][q]==d-1 and q not in homes]
        if not approach:continue
        for uid,(pos,v,role) in own[2].items():
            if role!="COLLECTOR" or v>20 or v>=load:continue
            for dest in approach:
                reach=DIST[pos][dest]
                if reach>min(d-1,4):continue
                moves=[(DIST[q][dest],-danger,a) for a,q,danger in safe_options(own,rival,original,uid)
                       if DIST[q][dest]==max(0,reach-1)]
                if not moves:continue
                a=min(moves)[2]
                value=load/((1+reach)*len(approach))
                item=(value,uid,a,foe,dest,round(load,2))
                if best is None or item>best:best=item
    if best and CMDS[best[2]]!=original.get(best[1]):
        return (best[1],best[2],dict(enemy=best[3],approach=best[4],enemy_cargo=best[5]))
    return None


def cohorts(record,snap,cfg):
    from zerosum_env.envs.carbon import carbon
    st,env=restore(snap,cfg);tags=[{uid:("opening",uid) for uid in p[2]} for p in st[0].observation.players]
    for t,(acts,expected) in enumerate(record["actions_and_post_cash"][274:],274):
        before=st[0].observation.players
        for obj,a in zip(st,acts):obj.action=a
        carbon.compute_next_board(st,env)
        for side,p in enumerate(st[0].observation.players):
            for uid in p[2].keys()-before[side][2].keys():tags[side][uid]=(t+1,p[2][uid][0],p[2][uid][2])
        if [p[0] for p in st[0].observation.players]!=expected:raise ValueError("recorded branch")
    return tags


def run_rule(record,snap,cfg,candidate,seat,rule,ref_tags):
    from zerosum_env.envs.carbon import carbon
    st,env=restore(snap,cfg);edits=[];counters=[Counter(),Counter()]
    births=[{uid:("opening",uid) for uid in p[2]} for p in st[0].observation.players]
    cash_curve=[];flow=[Counter(),Counter()]
    for t,(acts,_) in enumerate(record["actions_and_post_cash"][274:],274):
        before=st[0].observation.players
        cmd=[dict(a) for a in acts]
        new=rule(before,seat,t,cmd[seat])
        if new:
            uid,a,detail=new
            edits.append(dict(turn=t,id=uid,original=cmd[seat].get(uid),alternative=CMDS[a],detail=detail))
            if a:cmd[seat][uid]=CMDS[a]
            else:cmd[seat].pop(uid,None)
        if cmd[1-seat]!=acts[1-seat]:raise ValueError("opponent commands changed")
        for side,entry in enumerate(st):
            entry.action=cmd[side]
            for uid in cmd[side]:
                if uid in before[side][1]:continue
                if uid not in before[side][2]:counters[side]["absent_order"]+=1
                elif births[side].get(uid)!=ref_tags[side].get(uid):counters[side]["different_birth_order"]+=1
        events=[]
        with referee_events(events):carbon.compute_next_board(st,env)
        for event in events:
            kind,who,amount=event[:3]
            if kind not in ("harvest","enemy_tree","loot","death"):
                flow[who][kind]+=amount
        for side,p in enumerate(st[0].observation.players):
            for uid in p[2].keys()-before[side][2].keys():births[side][uid]=(t+1,p[2][uid][0],p[2][uid][2])
        cash_curve.append([p[0] for p in st[0].observation.players])
        if any(v.status!="ACTIVE" for v in st):break
    scores=[x.reward for x in st];d=scores[seat]-scores[1-seat];old=record["scores"][seat]-record["scores"][1-seat]
    if not edits and abs(d-old)>.005:raise ValueError("untouched branch differs")
    return dict(original_margin=round(old,2),alternative_margin=round(d,2),change=round(d-old,2),scores=scores,
                edits=edits,command_mapping=counters,cash_by_step=cash_curve,cash25=flow)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--snapshots",type=Path,default=HERE/"close-states.json")
    p.add_argument("--output",type=Path,default=HERE/"close-counterfactual.json")
    a=p.parse_args();configure_engine();limit_math_threads();src=json.loads(a.snapshots.read_text());cases=[]
    for name,r in read_records():
        s=next(i for i,p in enumerate(r["players"]) if p["name"]==name);key=f"{name}/{r['seed']}/{s}"
        snap=src["states"][key]["274"];cfg=dict(src["configuration"],randomSeed=r["seed"])
        tags=cohorts(r,snap,cfg)
        result={k:run_rule(r,snap,cfg,name,s,fn,tags) for k,fn in (("early_bank",early_bank),("preblock",preblock))}
        record=dict(key=key,seat=s,**result)
        # Isolate the four specific deviations in the positive example after the two rules have been evaluated.
        if key=="canopy/428888107/0":
            record["retrospective_single_moves"]=[]
            for edit in result["preblock"]["edits"]:
                def one(pl,side,t,original,e=edit):
                    if t!=e["turn"] or e["id"] not in pl[side][2]:return None
                    return (e["id"],CMDS.index(e["alternative"]),dict(isolated_from_preblock=True))
                one_result=run_rule(r,snap,cfg,name,s,one,tags)
                record["retrospective_single_moves"].append(dict(edit=edit,branch=one_result))
        cases.append(record)
    a.output.write_text(json.dumps(dict(
        method="Original board at step 274; simulate every remaining decision through terminal with the official referee. Each alternative changes at most one own action per turn. Other own and all opponent commands are literal archived commands on the original IDs. Missing/reassigned-unit commands are counted. No player replans against the intervention.",
        early_bank="A loaded collector (>=70) with 0..6 spare moves before the last bank starts a collision-avoiding shortest path if the recorded move does not get closer.",
        preblock="An own collector carrying <=20 may precede a rival (>=70, 2..6 steps from its unique nearest depot) to a cell one step outside that depot, on a shortest route. Ties among possible approaches are penalized. Decisions use current public positions and the own joint move.",
        cases=cases),separators=(",",":"))+"\n")


if __name__=="__main__":main()
