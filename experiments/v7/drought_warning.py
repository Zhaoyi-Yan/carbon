"""Learn an early warning for a new ten-turn drought using only past and current public cash and fleet."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from receipt_data import LAND
from receipt_check import real_target

DIST=np.asarray([[sum(min(abs(a-b),15-abs(a-b)) for a,b in zip(divmod(i,15),divmod(j,15))) for j in range(225)] for i in range(225)])
NAMES=('step','log_cash','own_cargo','rival_cash_edge','trees','foe_trees','workers','foe_workers','collectors',
       'next_tree_cash','remaining_tree_turns','wage_gap','short_safe_cargo','blocked_depots','unprotected_last_tree','recent_cash_change','recent_tree_loss','recent_roster_loss','opposing_cargo','recent_gross_income')


def scan(game,ident):
    frames=game['steps'];cfg=game['configuration'];fee=cfg.get('recCollectorCost',30);out=[]
    for seat in (0,1):
        flags=[];history=[]
        for t,frame in enumerate(frames):
            o=frame[0]['observation'];us,op=[o['players'][s] for s in (seat,1-seat)]
            workers=list(us[2].values());cargo=sum(v for p,v,j in workers if j=='COLLECTOR')
            flags.append(30<=t<=250 and us[0]<fee and cargo>=90 and op[0]-us[0]>=60)
            gross=sum(o['trees'][u][1] for u in us[3]);history.append((us[0],len(us[3]),len(us[2]),gross))
        begins={t for t in range(1,len(flags)-9) if not flags[t-1] and all(flags[t:t+10])}
        for t in range(20,min(241,len(frames)-19),2):
            if flags[t]:continue
            o=frames[t][0]['observation'];us,op=[o['players'][s] for s in (seat,1-seat)];cash=us[0]
            own=[(p,c) for p,c,k in us[2].values() if k=='COLLECTOR'];foe=[(p,c) for p,c,k in op[2].values() if k=='COLLECTOR']
            homes=list(us[1].values());cargo=sum(c for p,c in own)
            tree_cash=sum(min(6, max(0,50-age))*o['trees'][u][1] for u,(_,age) in us[3].items())
            left=sum(min(15,max(0,50-age)) for _,age in us[3].values())
            protected=0.;block=0
            for p,c in own:
                if c<20:continue
                threats=set(q for z,v in foe if v<=c for q in LAND[z])
                closed=sum(h in threats for h in homes);block=max(block,closed)
                d=min((DIST[p,h] for h in homes),default=16)
                if d<=4 and closed<len(homes):protected+=min(180,c)/(1.+.4*d)
            old=history[t-8];recent=history[t-7:t+1]
            pressured=sum(any(DIST[p,q]<=3 for q,v,job in op[2].values() if job=='PLANTER') for p,_ in us[3].values())
            values=[t/300,np.log1p(cash),min(900,cargo)/300,min(3000,max(-3000,op[0]-cash))/300,len(us[3]),len(op[3]),len(us[2]),len(op[2]),len(own),
                    np.log1p(tree_cash),left/15,max(0,60-cash-tree_cash)/30,protected/60,block,pressured if len(us[3])<=2 else 0,
                    np.clip(cash-old[0],-600,600)/100,old[1]-len(us[3]),old[2]-len(us[2]),sum(c for p,c in foe)/300,
                    sum(x[3] for x in recent)/50]
            onsets=sorted(i for i in begins if t<i<=t+10)
            out.append((ident,seat,t,values,bool(onsets),onsets[0] if onsets else None))
    return out


def measure(rows,p,threshold):
    y=np.array([r[4] for r in rows]);pred=p>=threshold
    advertised={(r[0],r[1],r[5]) for r in rows if r[4]}
    hit={(r[0],r[1],r[5]) for b,r in zip(pred,rows) if b and r[4]}
    return dict(frames=len(y),positive_frames=int(y.sum()),ap=round(float(average_precision_score(y,p)),5),
        alerts=int(pred.sum()),true_alert_frames=int((pred & y).sum()),available_onsets=len(advertised),warned_onsets=len(hit),
        false_alert_maps=len({r[0] for r,b in zip(rows,pred) if b and not r[4]}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('index',type=Path);parser.add_argument('target',type=Path);parser.add_argument('out',type=Path)
    a=parser.parse_args();roots=json.loads(a.index.read_text())['games'];tr=[];held=[];target=real_target(a.target)
    from receipt_data import layout
    key=layout(target);raw_target=scan(target,key)
    for r in roots:
        if r['layout']==key:continue
        g=json.loads(gzip.decompress((Path(r['source'])/'replays'/r['file']).read_bytes()))
        dataset=held if int(r['layout'][:8],16)%5==0 else tr
        dataset.extend(scan(g,r['layout']))
    y=np.array([r[4] for r in tr]);x=np.array([r[3] for r in tr]);vx=np.array([r[3] for r in held]);tx=np.array([r[3] for r in raw_target])
    results={};models={}
    for name,fields in [('cash',[1,2,3]),('assets',[1,2,3,4,5,6,7]),('warning',list(range(len(NAMES))))]:
        scale=StandardScaler().fit(x[:,fields]);model=LogisticRegression(C=.08,class_weight={0:1,1:12},max_iter=400).fit(scale.transform(x[:,fields]),y)
        predict=lambda z:model.predict_proba(scale.transform(z[:,fields]))[:,1]
        train=predict(x);p=predict(vx);target_p=predict(tx)
        threshold=float(np.quantile(train,.99))
        results[name]=dict(threshold=round(threshold,7),train=measure(tr,train,threshold),heldout=measure(held,p,threshold),
            original=[dict(turn=r[2],p=round(float(v),5),alarm=bool(v>=threshold),future_onset=r[5]) for r,v in zip(raw_target,target_p) if r[1]==1 and 44<=r[2]<=100])
        models[name]=dict(fields=[NAMES[k] for k in fields],mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coefficients=model.coef_[0].tolist(),bias=model.intercept_.item())
    d=dict(training_layouts=len({r[0] for r in tr}),independent_layouts=len({r[0] for r in held}),target_excluded=key,
           label='From a currently non-severe frame, a new run of at least ten consecutive severe frames begins in the next ten turns. Only legally visible history is used; all fitted thresholds correspond to the training top percentile.',results=results,models=models)
    a.out.write_text(json.dumps(d,indent=2)+'\n');print({k:v['heldout'] for k,v in results.items()},flush=True)
if __name__=='__main__':main()
