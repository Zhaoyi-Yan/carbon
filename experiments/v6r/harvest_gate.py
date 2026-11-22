"""Fit one old-root confidence gate per type on calibration games, then score unopened held-out games."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
GAPS=(-1.5,-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.,1.5,2.,3.)


def summarize(games, gaps, label=None, interval=(0,298)):
    def votes(game):
        rows=[r for r in game["changes"] if (label is None or r["cls"]==label) and interval[0]<=r["t"]<=interval[1]
              and gaps[r["cls"]] is not None and r["root_logit_gap"]<=gaps[r["cls"]]]
        return [len(rows)] + [sum(bool(r[k]) for r in rows) for k in ("teacher_exact_proposal","teacher_exact_root","teacher_class_proposal","teacher_class_root")]
    mat=np.array([votes(g) for g in games])
    sums=mat.sum(0).tolist()
    per_game=mat[:,1]-mat[:,2]
    return dict(n=sums[0],exact_candidate=sums[1],exact_root=sums[2],class_candidate=sums[3],class_root=sums[4],
                games_positive=int((per_game>0).sum()),games_negative=int((per_game<0).sum()),games_tied=int((per_game==0).sum()))


def calibrate(record):
    if record["partition"]!="calibration":raise ValueError("selection expects independent calibration games")
    selected=[];grid=[]
    for label in (0,1):
        scores=[summarize(record["games"],(gap,gap),label) for gap in GAPS]
        # Predefined cutoffs favor calibration support over a high score on tiny groups.
        accepted=[gap for gap,x in zip(GAPS,scores) if x["n"]>=25
                  and x["exact_candidate"]-x["exact_root"]>=.04*x["n"]
                  and x["class_candidate"]-x["class_root"]>=.04*x["n"]
                  and x["games_positive"]>x["games_negative"]]
        selected.append(max(accepted) if accepted else None)
        grid.append([dict(gap=g,**s) for g,s in zip(GAPS,scores)])
    return dict(gaps=selected,calibration=summarize(record["games"],selected),grid=grid,
                calibration_games=[g["file"] for g in record["games"]])


def test(result,record):
    if record["partition"]!="test" or record["offset"]!=12: raise ValueError("expected the unopened test block")
    if set(result["calibration_games"]) & {g["file"] for g in record["games"]}:raise ValueError("overlap")
    gaps=result["gaps"]
    result["test"]=dict(n_games=len(record["games"]),roots=record["all"]["roots"],performance=summarize(record["games"],gaps),
        ungated=summarize(record["games"],(float("inf"),float("inf"))),
        by_phase={str(lo)+"-"+str(hi):summarize(record["games"],gaps,interval=(lo,hi)) for lo,hi in ((0,74),(75,149),(150,224),(225,298))},
        by_intent=[summarize(record["games"],gaps,label) for label in (0,1)],
        median_microseconds=record["all"]["median_microseconds"],p95_microseconds=record["all"]["p95_microseconds"],
        games=[dict(file=g["file"],**summarize([g],gaps)) for g in record["games"]])
    per_game = result["test"]["games"]
    trial = np.array([[g["n"],g["exact_candidate"]-g["exact_root"]] for g in per_game])
    rng = np.random.default_rng(911)
    resamples = trial[rng.integers(len(trial),size=(4000,len(trial)))].sum(1)
    positive = resamples[:,0]>0
    result["test"]["paired_game_bootstrap_exact_delta_95"] = (np.quantile(resamples[positive,1]/resamples[positive,0],[.025,.975]).round(5).tolist() if positive.any() else None)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("calibrate","test"))
    p.add_argument("records",type=Path);p.add_argument("gate",type=Path)
    a=p.parse_args();data=json.loads(a.records.read_text())
    result=calibrate(data) if a.action=="calibrate" else test(json.loads(a.gate.read_text()),data)
    a.gate.write_text(json.dumps(result,indent=2)+"\n")
    print(result["gaps"],result["calibration"],result.get("test",{}).get("performance"))
