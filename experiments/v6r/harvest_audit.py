"""Teacher-force the frozen v6 root on independent games; compare the proposed and root actions."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import harvest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import load_strategy,configure_engine
from export import original


def stats(samples):
    result=Counter();elapsed=[]
    for x in samples:
        result["roots"]+=1
        elapsed.append(x["micros"])
        if not x["changed"]:continue
        result["proposed"]+=1
        for name in ("teacher_exact_proposal","teacher_exact_root","teacher_class_proposal","teacher_class_root"):
            result[name]+=int(x[name])
        result["stay_proposals" if x["cls"]==0 else "return_proposals"]+=1
    return dict(result,median_microseconds=round(float(np.median(elapsed)),1) if elapsed else None,
                p95_microseconds=round(float(np.quantile(elapsed,.95)),1) if elapsed else None)


def evaluate(raw, outfile, n, partition, offset, credit=False, preserve_bank=False):
    import torch
    configure_engine()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen=original()
    # Keep the original network/decision rules and change only the local execution thread cap.
    module=load_strategy("harvest_frozen_root",ROOT/"yan-v6.py",frozen.replace("set_num_threads(2)","set_num_threads(1)").encode())
    weights=json.loads((HERE/"harvest-model.json").read_text())
    harvest._HV_MODEL=tuple(np.asarray(a,"f") for a in weights["weights"])
    harvest._HV_THRESH=weights["thresholds"]
    harvest._HV_PRESERVE_BANK=preserve_bank
    harvest._HV_GAPS=None  # capture all eligible alternatives to assess a gate without replaying the rules
    if credit:
        from harvest_credit import compare
        harvest._HV_GAPS=json.loads((HERE/"harvest-gate.json").read_text())["gaps"]
    report=json.loads((HERE/"harvest-validation.json").read_text())
    games=sorted((g for g in report["games"] if g["split"]==(1 if partition=="calibration" else 2)),key=lambda g:hashlib.sha256(("hv-audit/"+g["layout"]).encode()).digest())[offset:offset+n]
    per_game=[];all_samples=[]
    for i,g in enumerate(games):
        recorded=json.loads(gzip.decompress((raw/g["file"]).read_bytes()))
        frames=recorded["steps"];side=g["winner"];samples=[]
        def capture(obs,config,est,act,lw,lb,prior,started,home=()):
            if t%3!=i%3: return act
            start=time.perf_counter_ns()
            prop=harvest._hv_proposal(obs,module.LAST_TURN,act,lw,home)
            micros=(time.perf_counter_ns()-start)/1000
            record=dict(t=t,micros=micros,changed=prop is not None)
            if prop is not None:
                diff=[w for w in set(prop)|set(act) if prop.get(w)!=act.get(w)]
                assert len(diff)==1 and diff[0] not in home
                wid=diff[0]; own=players[side];pos,_,role=own[2][wid]
                assert role=="COLLECTOR"
                def label(move):
                    a=harvest._HV_COMMANDS.index(move);q=harvest._HV_MOVES[pos][a]
                    dist=lambda p:min(harvest._HV_DIST[p][b] for b in own[1].values())
                    return 0 if a==0 else int(2-(dist(q)<dist(pos)))
                gold=teacher.get(wid);p=prop.get(wid);a=act.get(wid)
                record.update(root_logit_gap=float(lw[pos][harvest._HV_COMMANDS.index(a)] - lw[pos][harvest._HV_COMMANDS.index(p)]),cls=label(p),teacher_exact_proposal=(p==gold),teacher_exact_root=(a==gold),
                              teacher_class_proposal=label(p)==label(gold),teacher_class_root=label(a)==label(gold))
                if credit:
                    record["economy"]=compare(frames[t],frames[t+1],config,side,act,prop)
            samples.append(record)
            return act
        module._search=capture
        module._ST.clear()
        for t in range(len(frames)-1):
            obs=dict(frames[t][side]["observation"])
            players=frames[t][0]["observation"]["players"]
            obs.update(players=players,trees=frames[t][0]["observation"].get("trees",{}),step=t,player=side)
            teacher=frames[t+1][side]["action"] or {}
            module.agent(obs,recorded["configuration"])
            # The next board reflects the teacher, so collision attribution uses the teacher's decisions.
            module._ST["tr"].set_actions(players,teacher)
        expected=sum(t%3==i%3 for t in range(len(frames)-1))
        assert len(samples)==expected
        per_game.append(dict(file=g["file"],**stats(samples),changes=[{k:v for k,v in x.items() if k not in ("micros","changed")} for x in samples if x["changed"]]))
        all_samples.extend(samples)
        print(i+1,len(games),stats(samples),flush=True)
    output=dict(preserve_bank=preserve_bank,partition=partition,offset=offset,root_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
        method="Winner-side teacher-forced full trajectories; frozen v6 root generates act0 at every turn, search disabled. Every third root is probed, using the same locked-home workers. Coverage assumes a free root-search slot. Live quota, search acceptance and win rate are unmeasured.",
        games=per_game,all=stats(all_samples),phases={str(lo)+"-"+str(hi):stats([v for v in all_samples if lo<=v["t"]<=hi]) for lo,hi in ((0,74),(75,149),(150,224),(225,298))})
    if credit:
        def economic(rows):
            combined=Counter()
            for row in rows:
                if row.get("changed"):combined.update(row["economy"])
            return {k:round(v,4) for k,v in combined.items()}
        output["one_step_delta"]=economic(all_samples)
        output["by_phase_one_step_delta"]={str(lo)+"-"+str(hi):economic([r for r in all_samples if lo<=r["t"]<=hi]) for lo,hi in ((0,74),(75,149),(150,224),(225,298))}
    outfile.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("raw",type=Path)
    p.add_argument("--games",type=int,default=12);p.add_argument("--output",type=Path,default=HERE/"harvest-root-audit.json")
    p.add_argument("--partition",choices=("calibration","test"),default="test");p.add_argument("--offset",type=int,default=0)
    p.add_argument("--one-step-credit",action="store_true")
    p.add_argument("--preserve-bank",action="store_true")
    a=p.parse_args();evaluate(a.raw,a.output,a.games,a.partition,a.offset,a.one_step_credit,a.preserve_bank)
