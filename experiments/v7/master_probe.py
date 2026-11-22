"""One-turn paired referee probes on held-out early-liquidity and blocked-carrier observations."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from master_data import NAMES,ROOT
from master_fit import predict
import master
from localhead_data import load
from receipt_data import LAND,ACT,targets
from benchmark import validate_actions

def economic(m,pl,board,actions,side,config):
    # Absolute true stock only enters this offline referee transition.
    nxt=m._turn(board,config,actions)
    p=pl[side];after=nxt['players'][side];sites=set(p[1].values());gross=0.;cargo_lost=0.
    for j,team in enumerate(pl):
        for u,(pos,load,kind) in team[2].items():
            q=LAND[pos][ACT[actions[j].get(u,'NONE')]]
            if q in sites:gross+=load
            if j==side and load>0 and u not in after[2] and q not in sites:cargo_lost+=load
    return np.array([gross,after[0]-p[0],sum(v for q,v,k in after[2].values()),len(after[2])-len(p[2]),len(after[3])-len(p[3]),cargo_lost])
def main(a):
    rows=[r for r in json.loads((a.cache/'index.json').read_text()) if r['fold']=='test']
    with np.load(a.model) as z: model={k:z[k] for k in z.files}
    m=load(a.parent);master._ROLLOUT_LANDING=m._ROLLOUT_LANDING;master.WIDX=m.WIDX
    all_=defaultdict(list);max_error=0;count=0;joints=[];hires={}
    for row in rows:
        rgroup=row['group']
        with np.load(a.cache/row['cache']) as z:data={k:z[k] for k in z.files}
        with gzip.open(row['file']) as f:g=json.load(f)
        s=row['side'];te=NAMES.index(row['teacher']);frames=g['steps']
        deposits=targets(g)[1] if a.base_only else None
        first=[k for p,v,k in frames[1][0]['observation']['players'][s][2].values()]
        select=2 if first.count('COLLECTOR')>=2 else (1 if not first.count('COLLECTOR') else 0)
        for t in np.unique(data['wt']):
            if t<2 or t>m.LAST_TURN-12:continue
            obs=frames[t][0]['observation'];pl=obs['players'];p=pl[s];foe=pl[1-s]
            wi=np.flatnonzero(data['wt']==t);bi=np.flatnonzero(data['bt']==t);early=t<=75 and p[0]<145
            blocked=data['blocked'][wi].any() and (p[0]<180 or p[0]+150<foe[0])
            if a.base_only and not (early and len(bi) and len(p[2])<10 and p[0]>=30):continue
            if not (early or blocked):continue
            board={'step':int(t),'player':s,'players':pl,'trees':obs.get('trees',{}),'carbon':obs['full_carbon']}
            acts=[z['action'] or {} for z in frames[t+1]]
            verify=m._turn(board,g['configuration'],acts)['players']
            expected=frames[t+1][0]['observation']['players']
            max_error=max(max_error,max(abs(a[0]-b[0]) for a,b in zip(verify,expected) if b[2] or b[0]))
            z0=np.zeros((225,5),np.float32);b0=np.zeros((225,3),np.float32)
            wt=z0.copy();bt=b0.copy();pw=[v[0] for v in p[2].values()];pb=list(p[1].values())
            z0[pw]=data['wz'][wi,0]
            if len(bi):b0[pb]=data['bz'][bi,0]
            for k in (0,1,2):
                idx=np.flatnonzero(data['wr'][wi]==k) if k<2 else np.arange(len(bi))
                if not len(idx):continue
                vals=data['wx'][wi[idx]] if k<2 else data['bx'][bi[idx]]
                z=predict((model['w'+str(k)],model['s'+str(k)]),vals.astype(np.float32),np.full(len(idx),select))*.65
                if k<2:wt[np.asarray(pw)[idx]]=z
                else:bt[np.asarray(pb)[idx]]=z
            draws=[]
            for seed in (193,811,2621):
                values=[]
                baseline=m._sample(pl,s,z0,b0,np.random.RandomState(seed+1009*t),int(t))
                if a.base_only:
                    proposed=master._ms_recruits({'players':pl,'player':s},baseline,bt[pb]/.65)
                    other=proposed or baseline
                    assert all(other.get(u)==baseline.get(u) for u in p[2])
                    truth=[acts[s].get(u) for u in p[1]]
                    old=[baseline.get(u) for u in p[1]];new=[other.get(u) for u in p[1]]
                    if proposed:
                        for i,hire in enumerate(truth):
                            if hire!='RECCOLLECTOR' or old[i]==new[i] or hire not in (old[i],new[i]):continue
                            label='alternative' if new[i]==hire else 'baseline'
                            key=(row['match'],int(t),pb[i],label)
                            if key in hires:continue
                            next_workers=frames[t+1][0]['observation']['players'][s][2]
                            born=[u for u,(q,v,role) in next_workers.items() if u not in p[2] and q==pb[i] and role=='COLLECTOR']
                            if len(born)>1:raise ValueError('Ambiguous individual recruit')
                            u=born[0] if born else None
                            sums=[]
                            for h in (10,20,35,50):
                                amounts=[ledger[u][1] for ledger in deposits[t:t+h] if u and u in ledger and ledger[u][0]==s]
                                alive=bool(u and u in frames[min(int(t+h),len(frames)-1)][0]['observation']['players'][s][2])
                                sums.append([round(sum(amounts),4),alive])
                            hires[key]=dict(match=row['match'],map=rgroup,turn=int(t),depot=pb[i],uid=u,group=label,
                                           cumulative_own_bank_survival=sums)
                    joints.append([rgroup,int(proposed is not None),old==truth,new==truth,
                                  sum(z1==z2 for z1,z2 in zip(old,truth))/len(truth),
                                  sum(z1==z2 for z1,z2 in zip(new,truth))/len(truth)])
                else:other=m._sample(pl,s,wt,bt,np.random.RandomState(seed+1009*t),int(t))
                for action in (baseline,other):
                    validate_actions(action,p)
                    pair=list(acts);pair[s]=action;values.append(economic(m,pl,board,pair,s,g['configuration']))
                draws.append(values[1]-values[0])
            d=np.mean(draws,0)
            for name in ('all','early' if early else 'late_blocked',row['teacher']):
                all_[name].append((row['group'],d))
            count+=1
    keys=['gross_received','cash_change','remaining_worker_cargo','worker_count_change','tree_count_change','lost_cargo_outside_own_depot']
    report={}
    for n,vs in all_.items():
        ds=defaultdict(list)
        for map_,v in vs:ds[map_].append(v)
        maps=np.stack([np.mean(vs1,0) for vs1 in ds.values()])
        rng=np.random.RandomState(1987);boot=maps[rng.randint(len(maps),size=(3000,len(maps)))].mean(1)
        report[n]=dict(roots=len(vs),maps=len(ds),difference=dict(zip(keys,np.mean([v for k,v in vs],0).round(4))),
                       macro_map95={k:a.tolist() for k,a in zip(keys,np.quantile(boot,[.025,.975],axis=0).T.round(4))})
    out=dict(holdout_game_cases=len(rows),roots=count,max_frozen_transition_cash_error=max_error,sampling='Three shared seeded actual _sample draws per strategy and frame. Same recorded rival joint response, full stock reserved for the referee, frozen original root worker/base heads as comparator. Compared the proposed extra candidate directly, without search selection or later adaptation.',results=report)
    if a.base_only:
        vals=np.asarray([j[1:] for j in joints],float);changed=vals[vals[:,0]>0]
        out['scope']='Base-only candidate: preserved all frozen baseline worker commands, changed by projected affordable two-depot joint choice.'
        out['joint_commands']={'samples':len(vals),'changed':len(changed),'all':[round(float(v),4) for v in vals[:,1:].mean(0)],
                               'when_changed':[round(float(v),4) for v in changed[:,1:].mean(0)],
                               'fields':['baseline_exact_joint','alternative_exact_joint','baseline_per_depot_hit','alternative_per_depot_hit']}
        cohorts={}
        for label in ('alternative','baseline'):
            items=[r for r in hires.values() if r['group']==label]
            if not items:continue
            vals=np.array([r['cumulative_own_bank_survival'] for r in items])
            cohorts[label]={'unique_factual_collector_hires':len(items),'maps':len({r['map'] for r in items}),
                           'mean_gross':vals[:,:,0].mean(0).round(2).tolist(),'median_gross':np.median(vals[:,:,0],0).round(2).tolist(),
                           'fraction_recovered_hiring_cost':(vals[:,:,0]>=30).mean(0).round(3).tolist(),
                           'survival':vals[:,:,1].mean(0).round(3).tolist()}
        out['factual_hires']={'windows':[10,20,35,50],'groups':cohorts,'events':list(hires.values()),
                            'interpretation':'Unique archived actual collector hires correctly anticipated by exactly one alternative in a differing draw; full subsequent actions are the recorded policies. Observational individual receipts, not a counterfactual team profit.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['roots'],max_error,report['all'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('model',type=Path);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--base-only',action='store_true')
    main(p.parse_args())
