"""Extract legal features and forecasts from a frozen source archive with map-isolated holdout."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"
import numpy as np
import forecast
import assets

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import load_strategy
from export import original


def components():
    # Frozen v6 contains the unchanged v5 economic functions and historical weights.
    source=original().split("_V5_VALUE_MIX =",1)[0]
    m=load_strategy("forecast_v5",ROOT/"yan-v6.py",source.encode())
    params=json.loads((HERE/"assets-model.json").read_text())
    assets.np=np
    assets._V5_DIST=m._V5_DIST;assets._v5_home=m._v5_home;assets.LAST_TURN=298
    assets._R6_CARGO=np.asarray(params["cargo"]["coef"])
    assets._R6_TREE=np.asarray(params["tree"]["coef"])
    return m,params


def extract(raw,output,external_to=None,limit=None,source_revision=None):
    m,price=components()
    source=json.loads((raw/"index.json").read_text())
    games=[];x=[];y=[];baseline=[];step=[];group=[];actual=[]
    held=set(price["validation"]);asset_train=set(price["training"])
    prior=json.loads(external_to.read_text())["games"] if external_to else []
    seen_files={g["file"] for g in prior}
    seen_layouts={g["layout"] for g in prior}
    order=sorted(source,key=lambda r:hashlib.sha256(("forecast-external-1/"+r["file"]).encode()).digest()) if external_to else sorted(source,key=lambda r:r["file"])
    for r in order:
        if limit and len(games)>=limit:break
        if r["file"] in seen_files:continue
        path=raw/r["file"]
        packed=path.read_bytes();g=json.loads(gzip.decompress(packed));f=g["steps"];end=len(f)-1
        if end!=r["steps"] or g["configuration"]["size"]!=15 or g["configuration"]["episodeSteps"]!=300:
            raise ValueError(path)
        initial=f[0][0]["observation"]
        layout=hashlib.sha256(json.dumps([initial["full_carbon"],[sorted(p[1].values()) for p in initial["players"]]],separators=(",",":")).encode()).hexdigest()
        if layout in seen_layouts:continue
        if external_to:seen_layouts.add(layout)
        # The trained asset baseline must never see its evaluation games either.
        split=2 if external_to or path.name in held or path.name not in asset_train else (1 if int(hashlib.sha256(("fc2/"+layout).encode()).hexdigest()[:8],16)%6==0 else 0)
        if any(v["layout"]==layout and v["split"]!=split for v in games):raise ValueError("split layout overlap")
        game_idx=len(games);games.append(dict(file=r["file"],layout=layout,split=split,revision=r.get("revision") or source_revision,sha256=hashlib.sha256(packed).hexdigest(),turns=end))
        cash=[z[0]["observation"]["players"][0][0]-z[0]["observation"]["players"][1][0] for z in f]
        trackers=[m.BeliefTracker(m._Bel,i) for i in (0,1)]
        history={}
        for t in range(end):
            obs=f[t][0]["observation"];pl=obs["players"];meta=obs["trees"]
            cur=forecast._fc_summary(pl,meta);history[t]=cur
            sample=t%4==int(layout[:4],16)%4
            legal=[]
            for seat,tr in enumerate(trackers):
                visible=f[t][seat]["observation"]["carbon"]
                est,_=tr.step(pl,t,visible)
                if sample:legal.append(est)
                tr.set_actions(pl,f[t+1][seat]["action"] or {})
            if not sample:continue
            board=dict(players=pl,trees=meta,step=t)
            amount=pl[0][0]-pl[1][0]
            asset=assets._r6_asset_score(board,0)*3000-amount
            linear=m._v5_cash_only(board,0)*3000-amount
            targets=[cash[min(end,t+h)]-cash[t] for h in (20,50)]
            cand=forecast._fc_inputs(pl,meta,t,298,history)
            # v5 territory is evaluated on each viewer's legal belief, reporting both in seat-zero orientation.
            landscape=[]
            for seat in (0,1):
                board["carbon"]=legal[seat]
                landscape.append((1-2*seat)*3000*m._v5_territory_value(board,seat)-amount)
            x.append(cand);y.append(targets);step.append(t);group.append(game_idx)
            baseline.append([linear,asset,*landscape]);actual.append(min(299-t,end-t))
        if len(games)%30==0:print("extracted",len(games),flush=True)
    np.savez_compressed(output,X=np.array(x,"f"),Y=np.array(y,"f"),B=np.array(baseline,"f"),T=np.array(step,"i2"),
                        G=np.array(group,"i2"),L=np.array(actual,"i2"),games=json.dumps(games))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("raw",type=Path);p.add_argument("cache",type=Path)
    p.add_argument("--external-to",type=Path);p.add_argument("--limit",type=int);p.add_argument("--source-revision")
    a=p.parse_args();extract(a.raw,a.cache,a.external_to,a.limit,a.source_revision)
