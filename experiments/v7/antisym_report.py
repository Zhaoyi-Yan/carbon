"""Compare same-fog critic components and the actual retained half of each original proposal set."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
COLS={'single_raw':0,'dual_raw':3,'small_raw':4,'single':6,'dual':7,'small':8,'reference_small':9}
def bootstrap(by):
    groups=defaultdict(list)
    for g,v in by:groups[g].append(v)
    a=np.stack([np.mean(rs,0) for rs in groups.values()]);rng=np.random.RandomState(3692)
    ci=np.quantile(a[rng.randint(len(a),size=(3500,len(a)))].mean(1),[.025,.975],axis=0)
    return a.mean(0),ci
def gathered(folder):return [json.loads(path.read_text()) for path in sorted(folder.glob('*.json'))]
def archived(rows):
    results={}
    for phase in ('all','early','mid','late'):
        terms=defaultdict(list);cal=defaultdict(list);orient=defaultdict(dict)
        for r in rows:
            for j in r['rows']:
                if j.get('terminal'):continue
                if phase!='all' and phase!=('early' if j['t']<90 else 'mid' if j['t']<220 else 'late'):continue
                p=np.array(j['p']);g=r['layout'];s=j['s'];orient[g,j['t']][s]=p
                for horizon,y in [('35',j['future']),('final',j['final'])]:
                    for name,k in COLS.items():
                        if name=='reference_small':continue
                        value=p[k]*3000;err=value-y
                        base=p[0 if name.endswith('_raw') else 6]*3000
                        terms[name,horizon].append((g,[abs(err)-abs(base-y),abs(err),err,err*err,float((value>0)==(y>0)),err*(1-2*s)]))
        data={}
        for (name,h),items in terms.items():
            val,ci=bootstrap(items);fields=['delta_mae','mae','signed_bias','mse','sign_accuracy','seat0_bias']
            d=dict(zip(fields,val.round(4).tolist()));d['rmse']=round(float(np.sqrt(d.pop('mse'))),3)
            d['paired_ci95']=ci[:,0].round(3).tolist();d['boards']=len(items)
            data.setdefault(h,{})[name]=d
        asym=[]
        for (g,t),v in orient.items():
            if len(v)!=2:continue
            x,y=v[0],v[1]
            asym.append([abs(x[0]+y[0])*3000,abs(x[3]+y[3])*3000,abs(x[6]+y[6])*3000,abs(x[7]+y[7])*3000,abs(x[1]-x[2])*3000])
        if asym:data['seat_disagreement_MAE']=dict(zip(['single_raw','dual_raw','single_leaf','dual_leaf','enemy_big_vs_tiny'],np.mean(asym,0).round(3).tolist()))
        results[phase]=data
    return results
def ordered(score,num):
    keep=max(2,int(np.ceil(num*.5)))
    return [0]+[int(i)+1 for i in np.argsort(-score[1:],kind='stable')[:keep-1]]
def branch(rows):
    stats=defaultdict(list);pair=defaultdict(list);err=[];events=[]
    for r in rows:
        for z in r['rows']:
            p=np.array(z['p']);y=np.array(z['y']);g=r['layout'];n=len(p)
            if n<3:continue
            base=p[:,6]-p[:,10];old=ordered(base,n);ob=int(np.argmax(base))
            ids=np.triu_indices(n,1)
            for name,k in [('dual',7),('small',8),('reference_small',9)]:
                score=p[:,k]-p[:,10];new=ordered(score,n);nb=int(np.argmax(score));changed=set(new)!=set(old)
                for fb,mode in enumerate(('literal','waypoint')):
                    for t,h in enumerate((10,35)):
                        for u,obj in enumerate(('cash','assets')):
                            goal=y[fb,:,t,u];best=goal.max();current=goal[old].max()
                            gap=goal[ids[0]]-goal[ids[1]]
                            use=np.abs(gap)>=5
                            bm=base[ids[0]]-base[ids[1]];nm=score[ids[0]]-score[ids[1]]
                            if use.any():
                                a=np.mean(np.sign(bm[use])==np.sign(gap[use]))
                                b=np.mean(np.sign(nm[use])==np.sign(gap[use]))
                                pair[name,mode,h,obj].append((g,[b-a,a,b]))
                            stats[name,mode,h,obj].append((g,[float(changed),float(ob!=nb),goal[new].max()-current,best-goal[new].max(),best-current,
                                                         goal[nb]-goal[ob],float(goal[new].max()==best),float(current==best)]))
                if changed:events.append(dict(method=name,panel=r['panel'],seed=r['seed'],side=r['side'],t=z['t'],before=old,after=new))
            tf=p[:,7]-p[0,7];small=p[:,9]-p[0,9]
            err.extend((3000*abs(tf-small)).tolist())
    report={}
    for (name,mode,h,obj),v in stats.items():
        a,ci=bootstrap(v);root=report.setdefault(name,{}).setdefault(mode,{}).setdefault(str(h),{})
        f='changed_keep changed_pick delta_best_retained new_regret original_regret delta_top_pick new_retains_best original_retains_best'.split()
        root[obj]=dict(zip(f,a.round(5).tolist()),roots=len(v),paired_retained_95=ci[:,2].round(4).tolist(),paired_pick_95=ci[:,5].round(4).tolist())
        pp=pair.get((name,mode,h,obj))
        if pp:
            pv,pc=bootstrap(pp);root[obj].update(pair_diff=round(pv[0],5),pair_diff_ci95=pc[:,0].round(5).tolist(),base_pair=round(pv[1],5),new_pair=round(pv[2],5),pair_roots=len(pp))
    return report,events,float(np.mean(err)) if err else None

def main(data):
    games=gathered(data/'antisym-archive');branches=gathered(data/'antisym-branch');assert len(games)==21 and all(r['visibility_revision']==2 for r in games)
    assert len(branches)==36 and all(r['pool_revision']==3 for r in branches)
    a={fold:archived([r for r in games if r['fold']==fold]) for fold in ('train','test')}
    b={};events=[]
    for fold in ('train','test'):
        rr=[r for r in branches if r['split']==fold]
        q,e,err=branch(rr);events.extend(e);b[fold]=dict(metrics=q,roots=sum(len(r['rows']) for r in rr),candidates=sum(len(z['actions']) for r in rr for z in r['rows']),terminal_leaves=sum(v is not None for r in rr for z in r['rows'] for v in z['terminal']),layouts=len({r['layout'] for r in rr}),relative_small_error_cash=err)
    report=dict(archives=[{k:r[k] for k in ('id','layout','fold')} for r in games],
       branches=[{k:r[k] for k in ('name','seed','layout','side','version','split','panel')} for r in branches],
       definitions='Neural estimates compare routed large own vs large enemy vs embedded small enemy, from own reconstructed carbon and opponent visibility estimate. Cooked scores all keep the same horizon banking, 29% economic and 26% short-cash blends. An independent scorer does not learn or select weights. Future observations only provide labels.',
       real=a,counterfactual=b,kept_set_changes=events,
       branch_limits='Original local-v7 command is always eligible; up to six genuine pre-pruning generated candidates, fixed real simultaneous opposing action, all candidates evaluated on same legally estimated next board. Subsequent counterfactuals use the actual pre-action referee state for labels. Continuations use once the recorded tape and once feedback chasing the historical waypoint. They are counterfactual simulator returns, not an observed alternate game or win rate. Candidate choices truncate to the first six; protected candidate 0 is always retained.')
    (ROOT/'analysis/v7/antisym-value.json').write_text(json.dumps(report,indent=2)+'\n')
    print('public heldout',a['test']['all'])
    for f,d in b.items():
        print(f,d['roots'],d['layouts'])
        for k,v in d['metrics'].items():print(k,v['waypoint']['35']['cash'],v['literal']['35']['cash'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
