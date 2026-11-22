"""Choose rerank strength on whole-map training folds, then evaluate the untouched two maps per panel."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from advantage import FEATURES
from advantage_data import FIELDS

WEIGHTS=(0.,.25,.50,.75,1.)
def flatten(rows):
    return [(g,d) for g in rows for d in g['records']]
def goal(d,mode):
    v=np.asarray(d[mode])
    return .3*v[:,0,7]+.7*v[:,1,7]
def train(records,clone=False):
    xs=[];zs=[];ys=[];wt=[]
    for g,d in records:
        f=np.asarray(d['features']);z=3000*np.asarray(d['oldscore']);n=len(z)
        value=-25*np.asarray(d['teacher_distance']) if clone else .5*(goal(d,'literal')+goal(d,'waypoint'))
        for i in range(n):
            for j in range(i):
                xs.append(f[i]-f[j]);zs.append(z[i]-z[j]);ys.append(value[i]-value[j])
                wt.append((1.5 if g['candidate_won'] else .5) / max(1,n*(n-1)/2) if clone else 1./max(1,n*(n-1)/2))
    x=np.stack(xs);z=np.array(zs);y=np.clip(ys,-500,500);wt=np.array(wt);wt/=wt.sum()
    # Robust paired regression. Preserve the incumbent search score as a positive prior.
    def loss_gamma(v):
        e=(z*v[0]-y)/65.
        return np.sum(wt*(np.sqrt(1+e*e)-1))+.01*(v[0]-1.)**2, np.array([np.sum(wt*z/65.*e/np.sqrt(1+e*e))+.02*(v[0]-1.)])
    gamma=float(minimize(loss_gamma,np.array([1.]),jac=True,bounds=[(.2,2.5)]).x[0])
    scales=np.maximum(3.5,np.sqrt((wt[:,None]*x*x).sum(0)))
    x=x/scales;target=(y-gamma*z)/65.
    def loss(w):
        e=x@w-target
        return np.sum(wt*(np.sqrt(1+e*e)-1))+.035*(w@w),x.T@(wt*e/np.sqrt(1+e*e))+.07*w
    w=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B').x*65./scales
    return w,gamma*3000.
def pick(d,coef,scale,weight):
    f=np.asarray(d['features']);z=np.asarray(d['oldscore'])
    pred=z+weight*np.clip((f-f[0])@coef,-55.,55.)/scale
    if weight==0:
        best0=max(d['eligible'],key=lambda c:pred[c]);best0=best0 if pred[best0]>pred[0] else 0
        assert best0==d['winner']
    best=max(d['eligible'],key=lambda c:pred[c])
    return best if pred[best]>pred[0] else 0
def evaluate(records,w,alpha,weight):
    rows=[]
    for g,d in records:
        base=d['winner'];k=pick(d,w,alpha,weight)
        result={}
        for mode in ('literal','waypoint'):
            y=np.asarray(d[mode]);result[mode]=y[k]-y[base];result['goal_'+mode]=float(goal(d,mode)[k]-goal(d,mode)[base])
        rows.append(dict(meta=g,decision=d,index=k,baseline=base,changed=k!=base,**result))
    return rows
def mapstats(rows,key):
    group=defaultdict(list)
    for x in rows:group[x['meta']['layout']].append(x[key])
    return np.stack([np.mean(vals,axis=0) for vals in group.values()])
def select(records,clone=False):
    cv_results={}
    gs={g['layout']:i%3 for v in sorted({g['panel'] for g,d in records}) for i,g in enumerate(sorted({g['layout']:g for g,d in records if g['panel']==v}.values(),key=lambda g:g['layout']))}
    folds=[]
    for j in range(3):
        fit=[(g,d) for g,d in records if gs[g['layout']]!=j];val=[(g,d) for g,d in records if gs[g['layout']]==j]
        folds.append((val,*train(fit,clone)))
    for weight in WEIGHTS:
        rows=[v for val,w,a in folds for v in evaluate(val,w,a,weight)]
        means=[mapstats(rows,'goal_'+mode) for mode in ('literal','waypoint')]
        cash=[mapstats(rows,mode)[:,1,0] for mode in ('literal','waypoint')]
        # Require both future-command conventions to agree; use training folds alone.
        score=min(v.mean()-.35*np.std(v)/np.sqrt(len(v)) for v in means)+.25*min(v.mean() for v in cash)
        cv_results[weight]=dict(score=float(score),literal=float(means[0].mean()),waypoint=float(means[1].mean()),
                                h10cash=[round(v.mean(),4) for v in cash],changed=sum(v['changed'] for v in rows))
    return max(WEIGHTS,key=lambda v:cv_results[v]['score']),cv_results
def stats(rows):
    if not rows:return {}
    rng=np.random.RandomState(97213);changed=[r for r in rows if r['changed']]
    out=dict(states=len(rows),changes=len(changed),maps=len({r['meta']['layout'] for r in rows}),modes={})
    for mode in ('literal','waypoint'):
        maps=mapstats(rows,mode);goalmap=mapstats(rows,'goal_'+mode);cv=maps[rng.randint(len(maps),size=(5000,len(maps)))].mean(1)
        out['modes'][mode]=dict(mean_across_maps=np.mean(maps,0).round(4).tolist(),
            mean_on_changed=(np.mean([r[mode] for r in changed],axis=0).round(4).tolist() if changed else None),
            primary_equal_map_mean=round(float(goalmap.mean()),4),primary_positive_maps=int((goalmap>.01).sum()),
            primary_negative_maps=int((goalmap<-.01).sum()),
            primary_ci95=np.quantile(goalmap[rng.randint(len(maps),size=(5000,len(maps)))].mean(1),[.025,.975]).round(4).tolist(),
            cash10_map_ci95=np.quantile(cv[:,1,0],[.025,.975]).round(4).tolist())
    return out
def main(a):
    games=[json.loads(p.read_text()) for p in sorted(a.cache.glob('*.json'))]
    assert len(games)==96 and len({g['layout'] for g in games})==48
    allrows=flatten(games);tr=[r for r in allrows if r[0]['split']=='train'];te=[r for r in allrows if r[0]['split']=='test']
    trainmaps={g['layout'] for g,d in tr};testmaps={g['layout'] for g,d in te}
    assert len(trainmaps)==36 and len(testmaps)==12 and not trainmaps&testmaps
    chosen,cv=select(tr);w,alpha=train(tr);_,clone_cv=select(tr,True);wbc,abc=train(tr,True)
    clone_weight=max(WEIGHTS,key=lambda v:clone_cv[v]['score'])
    testrows=evaluate(te,w,alpha,chosen);test=stats(testrows)
    supported=(chosen>0 and all(test['modes'][m]['primary_equal_map_mean']>0 and test['modes'][m]['mean_across_maps'][1][0]>=0 for m in ('literal','waypoint')))
    model={'coef':w.round(7).tolist(),'score_scale':round(alpha,5),'weight':chosen,'recommended_for_qualification':bool(supported),
           'feature_names':FEATURES,'training_maps':sorted(trainmaps)}
    a.model.write_text(json.dumps(model,indent=2)+'\n')
    bypanel={str(j):stats([r for r in testrows if r['meta']['panel']==j]) for j in range(1,7)}
    counts={str(v):sum(g['split']=='test' for g in games if g['panel']==v) for v in range(1,7)}
    report=dict(policy=a.parent.name,training_maps=sorted(trainmaps),testing_maps=sorted(testmaps),test_games_by_panel=counts,
                games=len(games),recorded_referee_transitions=sum(299 for g in games),training_states=len(tr),test_states=len(te),
                training_branch_count=sum(len(d['features']) for g,d in tr),test_branch_count=sum(len(d['features']) for g,d in te),
                paired_label='Same current state, observed rival first command, changed finalist joint. Frozen six-candidate, two-ply search; at most two finalists plus original incumbent. Subsequent historical commands replayed literally and with displaced survivors routed toward their recorded next squares. New births matched by side, depot, role and birth turn.',
                objective='30% score at step 5 plus 70% at step 10, averaged across two continuations. Score is cash margin plus .55 of distance-discounted held cargo, 26 per live worker, .55 of up to twelve turns of tree receipts, and 1200 for an early terminal advantage. Cash/deposits also reported separately.',
                selection='36 training layouts; two whole layouts held from each of six panels. Strength chosen by three training-only folds. Deploy only if weighted solvency label and cash margin at ten steps both improve in both held-out tape conventions.',
                features=list(FEATURES),fields=FIELDS,horizons=[5,10],chosen_weight=chosen,score_scale=alpha,training_cv=cv,eligible_for_game_test=bool(supported),
                heldout=test,by_panel=bypanel,
                winner_weighted_imitation={'weight':clone_weight,'cv':clone_cv,'heldout':stats(evaluate(te,wbc,abc,clone_weight))},
                predeclared_half_strength=stats(evaluate(te,w,alpha,.5)),training_groups=len(trainmaps))
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    archive=[]
    for r in testrows:
        if not r['changed']:continue
        g,d=r['meta'],r['decision'];archive.append(dict(panel=g['panel'],layout=g['layout'],seed=g['seed'],side=g['side'],step=d['t'],
            original=d['acts'][r['baseline']],suggested=d['acts'][r['index']],literal=r['literal'].round(4).tolist(),waypoint=r['waypoint'].round(4).tolist()))
    with gzip.open(a.changed,'wt') as f:json.dump(archive,f,separators=(',',':'))
    print(len(tr),len(te),chosen,supported,{k:(v['primary_equal_map_mean'],v['mean_across_maps'][1][0],v['primary_ci95']) for k,v in test['modes'].items()},len(archive),cv)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('parent',type=Path);p.add_argument('model',type=Path);p.add_argument('report',type=Path);p.add_argument('changed',type=Path);main(p.parse_args())
