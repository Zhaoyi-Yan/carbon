"""Check production formulas, seat symmetry, terminal behavior and per-leaf cost on withheld public boards."""
import argparse
import gzip
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import numpy as np
import forecast as runtime
from forecast_data import components
from forecast_mix import NAMES,frame_data

HERE=Path(__file__).resolve().parent


def audit(raw,cache):
    model=np.load(HERE/"forecast-model.npz");mix=json.loads((HERE/"forecast-blend.json").read_text())
    runtime._FC_WEIGHTS=tuple(model[k] for k in NAMES)
    runtime._FC_BLEND=np.array(mix["blend"]["coef"]);runtime._FC_TERR=np.array(mix["legacy_territory"]["coef"])
    runtime._FC_ALPHA=np.array(mix["horizon_mix"])
    z=np.load(cache);games=json.loads(str(z["games"]))
    _,a,_,b=frame_data(z,model)
    expected=250*((a@runtime._FC_BLEND)*runtime._FC_ALPHA+(b@runtime._FC_TERR)*(1.-runtime._FC_ALPHA))
    old,_=components()
    boards=[]
    for gi,g in enumerate(games[:8]):
        replay=json.loads(gzip.decompress((raw/g["file"]).read_bytes()));frames=replay["steps"]
        histories={};tracker=old.BeliefTracker(old._Bel,0);rows=np.flatnonzero(z["G"]==gi)
        frame_idx={int(z["T"][r]):r for r in rows[::7]}
        for t,f in enumerate(frames[:-1]):
            obs=f[0]["observation"];pl=obs["players"];meta=obs["trees"]
            histories[t]=runtime._fc_summary(pl,meta);est,_=tracker.step(pl,t,obs["carbon"])
            tracker.set_actions(pl,frames[t+1][0]["action"] or {})
            if t not in frame_idx:continue
            row=frame_idx[t];board=dict(players=pl,trees=meta,carbon=est,step=t)
            history={k:v for k,v in histories.items() if t-30<=k<=t}
            oldval=old._v5_territory_value(board,0)
            computed=runtime._fc_heads(board,0,298,history,oldval)
            if not np.allclose(expected[row,0],computed,atol=.0015):raise ValueError((row,computed,expected[row,0]))
            if not np.allclose(-computed,runtime._fc_heads(board,1,298,history,-oldval),atol=.0015):raise ValueError("side sign")
            swapped=dict(board,players=pl[::-1])
            reversed_history={k:v[::-1] for k,v in history.items()}
            if not np.allclose(-computed,runtime._fc_heads(swapped,0,298,reversed_history,-oldval),atol=.0015):raise ValueError("seat permutation")
            simboard=dict(board,trees={},_st=SimpleNamespace(trees_meta=meta))
            if t>=7:
                truncated={k:v for k,v in history.items() if k<=t-7}
                if not np.allclose(computed,runtime._fc_heads(simboard,0,298,truncated,oldval),atol=.0015):raise ValueError("simulated leaf history / metadata")
            if not np.all(runtime._fc_heads(dict(board,step=299),0,298,history,oldval)==0):raise ValueError("terminal flow")
            boards.append((board,history,oldval))
    timings={"existing_v5":[],"public20_standalone":[],"forecast_with_known_v5":[],"v5_plus_forecast":[]}
    for _ in range(4):
        for board,history,val in boards:
            t0=time.perf_counter_ns();v=old._v5_territory_value(board,0);t1=time.perf_counter_ns()
            runtime._fc_heads(board,0,298,history,v);t2=time.perf_counter_ns()
            timings["existing_v5"].append((t1-t0)/1e6);timings["forecast_with_known_v5"].append((t2-t1)/1e6)
            timings["v5_plus_forecast"].append((t2-t0)/1e6)
            t3=time.perf_counter_ns();runtime._fc_predict(board,298,history);t4=time.perf_counter_ns()
            timings["public20_standalone"].append((t4-t3)/1e6)
    return dict(legal_boards=len(boards),games=8,checks=["vectorized training/live parity","same-board side signs","player permutation","simulated tree metadata and seven-step root history cutoff","zero future flow on terminal board"],milliseconds={
        k:dict(median=round(float(np.median(v)),4),p95=round(float(np.quantile(v,.95)),4),n=len(v)) for k,v in timings.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("raw",type=Path);p.add_argument("cache",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();r=audit(a.raw,a.cache);a.output.write_text(json.dumps(r,indent=2)+"\n");print(r)
