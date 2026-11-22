"""Independent ten-ply feedback check of selected held-out changes with neither future command tape."""
import argparse,copy,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from advantage_data import mod,manifest,prepare,footprint,landings,configure_engine,FIELDS
def fog(m,b,s):
    stock=np.zeros(225,np.float32)
    for p,v,k in b['players'][s][2].values():
        if k=='COLLECTOR':stock[p]=b['carbon'][p]
        else:
            for q in m.NB[p]:stock[q]=b['carbon'][q]
    return stock
def view(m,b,tr,s,previous):
    pl=b['players'];t=b['step'];est,known=tr.step(pl,t,fog(m,b,s))
    return m._quant(m.encode(pl,t,s,est,known,previous))
def trial(m,state,trackers,previous,side,actual_foe,commands,cfg,seed,styles):
    boards=[state,state];hist=[copy.deepcopy(trackers),copy.deepcopy(trackers)];prev=[previous,previous]
    income=np.zeros((2,2));sunk=np.zeros((2,2));stats=[[],[]];ended=[False,False]
    for j in range(10):
        if j:
            batch=np.stack([view(m,b,h[s],s,p) for b,h,p in zip(boards,hist,prev) for s in (0,1)])
            w,bb,_=m._V6R_TINY(batch,np.tile(styles,2))
        for i in range(2):
            if ended[i]:continue
            b=boards[i];pl=b['players'];t=b['step']
            if not j:orders=[commands[i],actual_foe] if side==0 else [actual_foe,commands[i]]
            else:
                orders=[m._sample(pl,s,w[2*i+s],bb[2*i+s],np.random.RandomState(seed+191*t+17*s),t) for s in (0,1)]
            for s in (0,1):hist[i][s].set_actions(pl,orders[s])
            homes={q:s for s,p in enumerate(pl) for q in p[1].values()}
            for q,group in landings(pl,orders,cfg).items():
                bank=homes.get(q)
                if bank is not None:income[i,bank]+=sum(w1[2] for w1 in group)
                if len(group)==1 and bank is None:continue
                cm=[w1 for w1 in group if w1[3]=='COLLECTOR'];low=min((w1[2] for w1 in cm),default=None)
                light=[w1 for w1 in cm if w1[2]==low];winner=light[0][0] if len(light)==1 else None
                for s,u,v,k in group:
                    if (bank is not None and bank!=s) or (bank is None and len(group)>1 and winner!=s):sunk[i,s]+=v
            prev[i]=pl;boards[i]=m._turn(b,cfg,orders)
            empty=[not p[2] and not p[3] and (not p[1] or p[0]<30) for p in boards[i]['players']]
            if j in (4,9) or any(empty):
                cash,cargo,near,fleet,trees=footprint(boards[i],side,m);terminal=int(empty[1-side])-int(empty[side])
                val=[cash,income[i,side]-income[i,1-side],cargo,near,fleet,trees,sunk[i,side]-sunk[i,1-side],
                     cash+.55*near+26*fleet+.55*trees+1200*terminal,terminal]
                stats[i].append(val)
                if any(empty):
                    while len(stats[i])<2:stats[i].append(val)
                    ended[i]=True
    return np.asarray(stats[1])-np.asarray(stats[0])

def main(a):
    configure_engine();m=mod(a.parent);report=json.loads(a.report.read_text())
    changes=json.load(gzip.open(a.changes));By=defaultdict(dict)
    for c in changes:By[c['panel'],c['seed'],c['side']][c['step']]=c
    rows=[];legalchecks=0
    for meta,g in manifest():
        k=meta['panel'],meta['seed'],meta['side']
        if k not in By:continue
        steps=By[k];s=meta['side'];cfg,states,obs=prepare(m,meta,g);h=[m.BeliefTracker(m._Bel,j) for j in (0,1)];prev=None
        for t in range(max(steps)+1):
            state=states[t]
            # Verify the reconstructed per-seat visibility against the actual official observation.
            assert np.allclose(fog(m,state,s),obs[t]['carbon'],atol=.0001);legalchecks+=1
            for j in (0,1):
                h[j].step(state['players'],t,fog(m,state,j))
            actual=g['actions_and_post_cash'][t][0]
            if t in steps:
                c=steps[t];out=defaultdict(list)
                for idx,style in enumerate(((22,21),(21,22))):
                    val=trial(m,state,h,prev,s,actual[1-s],[c['original'],c['suggested']],cfg,5437+idx*1721,style)
                    out[str(style)].append(val)
                multiple=np.stack([ds[0] for ds in out.values()])
                rows.append(dict(layout=meta['layout'],t=t,seed=meta['seed'],side=s,panel=meta['panel'],delta=multiple.round(4).tolist()))
            for j in (0,1):h[j].set_actions(state['players'],actual[j])
            prev=state['players']
    counts=defaultdict(int)
    for game in [json.loads(p.read_text()) for p in a.cache.glob('*.json')]:
        if game['split']=='test':counts[game['layout']]+=len(game['records'])
    collapsed=defaultdict(lambda:np.zeros((2,2,len(FIELDS))))
    for r in rows:collapsed[r['layout']]+=np.array(r['delta'])
    mapmeans=np.stack([collapsed[g]/n for g,n in counts.items()]);rng=np.random.RandomState(412);mapval=mapmeans.mean(1)
    goals=.3*mapval[:,0,7]+.7*mapval[:,1,7];boot=mapval[rng.randint(len(mapval),size=(6000,len(mapval)))].mean(1)
    out={'holdout_maps':len(counts),'all_eligible_roots':sum(counts.values()),'changed_roots':len(rows),'trials':2*len(rows),
         'future_policy':'Both teams re-decide from their own seat-visible fog and historical belief at each next state with the frozen tiny neural policy. Two common-random seeds and swapped expert IDs 21/22; only the first rival move is the real recording.',
         'fields':FIELDS,'horizons':[5,10],'official_visibility_checks':legalchecks,
         'per_style_map_equal_means':mapmeans.mean(0).round(4).tolist(),'mean_both':mapval.mean(0).round(4).tolist(),
         'score_delta':round(goals.mean(),4),'score_ci95':np.quantile(goals[rng.randint(len(goals),size=(6000,len(goals)))].mean(1),[.025,.975]).round(4).tolist(),
         'cash10_ci95':np.quantile(boot[:,1,0],[.025,.975]).round(4).tolist(),'events':rows}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(len(rows),out['mean_both'],out['score_delta'],out['cash10_ci95'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('cache',type=Path);p.add_argument('report',type=Path);p.add_argument('changes',type=Path);p.add_argument('output',type=Path)
    main(p.parse_args())
