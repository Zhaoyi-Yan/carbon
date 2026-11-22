"""Fit and audit a three-way collector preference using only the winning seat of each replay."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"
import numpy as np
import harvest

HERE = Path(__file__).resolve().parent
ELITE = {"dd1f117a64c676225f0c6333b1cdd1e7","226891089ba7f54b198c8f8dc771dfc2"}


def split_key(layout):
    key = int(hashlib.sha256(("harvest-1/"+layout).encode()).hexdigest()[:8],16)%100
    return 0 if key<70 else (1 if key<84 else 2)


def extract(raw, dest):
    xs, ys, groups, turns = [],[],[],[]
    games = []
    rows = json.loads((raw/"index.json").read_text())
    for i,r in enumerate(sorted(rows, key=lambda row:(row["id"],row["game"]))):
        path=raw/r["file"]
        g=json.loads(gzip.decompress(path.read_bytes()))
        frames=g["steps"]
        if len(frames)-1!=r["steps"] or g["rewards"]!=[r["scores"][k] for k in r["order"]]:
            raise ValueError(path.name)
        if r["winner"] not in r["order"]:
            continue
        side=r["order"].index(r["winner"])
        board=frames[0][0]["observation"]
        layout=hashlib.sha256(json.dumps([board["full_carbon"],[sorted(p[1].values()) for p in board["players"]]],separators=(",",":")).encode()).hexdigest()
        slot=len(games)
        games.append(dict(file=r["file"],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          layout=layout,split=split_key(layout),winner=side,teacher=r["strategies"][r["winner"]]["name"],
                          elite=r["strategies"][r["winner"]]["id"] in ELITE))
        for t in range(int(layout[:4],16)%4,len(frames)-1,4):
            own=frames[t][side]["observation"]
            players=frames[t][0]["observation"]["players"]
            obs=dict(players=players,carbon=own["carbon"],step=t,player=side)
            commands=frames[t+1][side]["action"] or {}
            homes=players[side][1].values()
            for uid,pos,cargo,home,x in harvest._hv_rows(obs,g["configuration"]["episodeSteps"]-2):
                act=harvest._HV_COMMANDS.index(commands.get(uid))
                dest_pos=harvest._HV_MOVES[pos][act]
                y=0 if act==0 else (1 if min(harvest._HV_DIST[dest_pos][h] for h in homes)<home else 2)
                xs.append(x);ys.append(y);turns.append(t);groups.append(slot)
        if (i+1)%50==0: print("read",i+1,flush=True)
    np.savez_compressed(dest,X=np.array(xs,"f"),Y=np.array(ys,"i1"),turn=np.array(turns,"i2"),group=np.array(groups,"i2"),games=json.dumps(games))
    return load(dest)


def load(dest):
    with np.load(dest) as data:
        return data["X"],data["Y"],data["turn"],data["group"],json.loads(str(data["games"]))


def verify_sources(raw, games):
    index = {r["file"]:r for r in json.loads((raw/"index.json").read_text()) if r["winner"] in r["order"]}
    if set(index) != {g["file"] for g in games}:
        raise ValueError("source and cache reference different games")
    for g in games:
        row = index[g["file"]]
        if hashlib.sha256((raw/g["file"]).read_bytes()).hexdigest() != g["source_sha256"]:
            raise ValueError(f"changed replay: {g['file']}")
        g["revision"] = row["revision"]


def behavior_examples(X, Y, t, test):
    mid = (t >= 75) & (t < 225)
    late_loaded = (t >= 225) & (X[:,1] >= .45)
    calm_light = (X[:,1] < .45) & (X[:,7]*7 >= 3)
    masks = {
        "mid_rich_calm_light": mid & calm_light & (X[:,0] >= .5),
        "mid_poor_calm_light": mid & calm_light & (X[:,0] < .2),
        "late_loaded_slack8plus": late_loaded & (X[:,4]*25 >= 8),
        "late_loaded_slack0to3": late_loaded & (X[:,4]*25 <= 3),
    }
    return {key:np.bincount(Y[m & test].astype(int), minlength=3).tolist() for key,m in masks.items()}


def fit(data, report, weights):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(91)
    X,Y,ts,group,games=data
    split=np.array([g["split"] for g in games])[group]
    x=torch.from_numpy(X);y=torch.from_numpy(Y.astype("int64"))
    counts=np.bincount(group)
    w=torch.from_numpy(np.array([(.65+.35*g["elite"])/max(n,1)**.5 for g,n in zip(games,counts)],"f")[group])
    w/=w.mean()
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,3))
    opt=torch.optim.AdamW(model.parameters(), lr=.007, weight_decay=.002)
    train=np.flatnonzero(split==0)
    rng=np.random.default_rng(91)
    best=None
    for epoch in range(38):
        for sel in np.array_split(rng.permutation(train),max(1,len(train)//1024)):
            loss=(torch.nn.functional.cross_entropy(model(x[sel]),y[sel],reduction="none")*w[sel]).mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            val=float(torch.nn.functional.cross_entropy(model(x[split==1]),y[split==1]))
        if best is None or val<best[0]:
            best=(val,epoch,[a.detach().clone() for a in model.parameters()])
    with torch.no_grad():
        for p,v in zip(model.parameters(),best[2]):p.copy_(v)
        probs=model(x).softmax(1).numpy()
    layers=[a.T.numpy() if a.ndim==2 else a.numpy() for a in best[2]]
    predicted=probs.argmax(1)
    threshold=[]
    for c in (0,1):
        # Use the calibration partition once to choose a high-precision operating point.
        vals=[(.58,.68),(.64,.72)][c]
        chosen=.90
        for th in np.arange(vals[0],.901,.02):
            ix=(split==1)&(predicted==c)&(probs[:,c]>=th)
            if ix.sum()>=35 and np.mean(Y[ix]==c)>=vals[1]:
                chosen=round(float(th),2);break
        threshold.append(chosen)
    meta=dict(architecture=[18,32,3],classes=["stay","inward","other"],thresholds=threshold,
              weights=[np.round(a,7).tolist() for a in layers])
    weights.write_text(json.dumps(meta,separators=(",",":"))+"\n")
    argmax=np.argmax(probs,1)
    assert [a.shape for a in layers]==[(18,32),(32,),(32,3),(3,)]
    def metrics(mask):
        n=int(mask.sum())
        if not n:return {"n":0}
        counts=np.bincount(Y[mask].astype(int),minlength=3)
        pred=argmax[mask]; gold=Y[mask]
        conf=np.bincount((gold*3+pred).astype(int),minlength=9).reshape(3,3)
        gated=[]
        for c in (0,1):
            ix=(pred==c)&(probs[mask,c]>=threshold[c])
            gated.append(dict(n=int(ix.sum()),precision=round(float(np.mean(gold[ix]==c)),4) if ix.any() else None))
        return dict(n=n,class_counts=counts.tolist(),accuracy=round(float(np.mean(pred==gold)),4),
                    confusion=conf.tolist(),gated=gated)
    game_scores=[]
    for i,g in enumerate(games):
        if g["split"]==2 and (group==i).any():
            game_scores.append(float(np.mean(argmax[group==i]==Y[group==i])))
    info=dict(method="Winning seats, observation at t and action at t+1, one of four turn phases per game; map-layout grouped 70/14/16 SHA partition; CPU, seed 91.",
              calibration_epoch=best[1]+1,calibration_cross_entropy=best[0],thresholds=threshold,
              partitions={name:dict(games=sum(g["split"]==i for g in games),layouts=len({g["layout"] for g in games if g["split"]==i}),metrics=metrics(split==i)) for i,name in enumerate(("train","calibration","test"))},
              test_game_macro_accuracy=round(float(np.mean(game_scores)),4),
              test_by_phase={str(lo)+"-"+str(hi):metrics((split==2)&(ts>=lo)&(ts<=hi)) for lo,hi in ((0,74),(75,149),(150,224),(225,298))},
              test_elite=metrics((split==2)&np.array([g["elite"] for g in games])[group]),
              test_behavior={"stay_at_cell_carbon":{}, "inward_at_cargo":{}},games=games)
    for ix,(lo,hi) in enumerate(((3,20),(20,50),(50,101))):
        mask=(split==2)&(X[:,0]*100>=lo)&(X[:,0]*100<hi);n=int(mask.sum())
        info["test_behavior"]["stay_at_cell_carbon"][f"{lo}-{hi}"]=[n,round(float(np.mean(Y[mask]==0)),4)]
    for lo,hi in ((0,30),(30,90),(90,200),(200,500)):
        mask=(split==2)&(X[:,1]*200>=lo)&(X[:,1]*200<hi);n=int(mask.sum())
        info["test_behavior"]["inward_at_cargo"][f"{lo}-{hi}"]=[n,round(float(np.mean(Y[mask]==1)),4)]
    info["test_behavior"]["context_class_counts"] = behavior_examples(X,Y,ts,split==2)
    report.write_text(json.dumps(info,indent=2)+"\n")
    print({k:(p["games"],p["metrics"]) for k,p in info["partitions"].items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("raw",type=Path);p.add_argument("cache",type=Path)
    p.add_argument("--report",type=Path,default=HERE/"harvest-validation.json");p.add_argument("--weights",type=Path,default=HERE/"harvest-model.json")
    a=p.parse_args()
    data = load(a.cache) if a.cache.exists() else extract(a.raw,a.cache)
    verify_sources(a.raw,data[-1])
    fit(data,a.report,a.weights)
