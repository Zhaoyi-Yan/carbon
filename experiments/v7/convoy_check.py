"""Out-of-map public-path evaluation, with the observing player's fog and a shared neural first-ply prior."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import convoy as c
import convoy_fit as ft
from convoy_fit import ROOT
sys.path.append(str(ROOT/'experiments/v6r'))
from localhead_data import load as load_policy

def reference(pos,previous,ctx,mode):
    home=ctx[0];out=np.full((len(pos),5),.02 if mode=='shortest' else .04)
    d=c._CONVOY_L[pos]
    progress=home[d]==np.maximum(0,home[pos]-1)[:,None]
    h=.9 if mode=='shortest' else (.33 if ctx[-1]>=.65 else .1)
    out+=h*progress/progress.sum(1,keepdims=True)
    if mode=='inertia':out[np.arange(len(pos)),previous]+=.8-h
    return out
def baseline(pos,prior,ctx,name):
    state=c._CONVOY_L[pos];prev=np.arange(5);mass=prior;maps=[]
    for h in range(3):
        maps.append(np.bincount(state,weights=mass,minlength=225))
        if h<2:
            mass=(mass[:,None]*reference(state,prev,ctx,name)).ravel();state=c._CONVOY_L[state].ravel();prev=np.tile(np.arange(5),len(mass)//5)
    return maps

def risk_module(obs,base,work,foework,locked,clock):
    start=time.perf_counter();joint=c._convoy_plan(obs,base,work,foework,locked)
    clock.append(1000*(time.perf_counter()-start))
    return joint
def main(cache,out,parent):
    ft.geometry();model=json.loads(c.FIT.read_text());c._CONVOY_W=np.array(model['weights'],np.float32)
    train,rows=ft.splits(cache);assert model['validation_layouts']==[r['layout'] for r in rows]
    m=load_policy(parent);c.WACT=m.WACT;c.WIDX=m.WIDX;c.LAST_TURN=m.LAST_TURN
    measures=defaultdict(list);timings=[];actions=[]; maprecords=defaultdict(list)
    for i,row in enumerate(rows):
        g=ft.read(row);steps=g['steps'];cfg=g['configuration'];me=i%2;op=1-me
        picks={q for q in (40,89,141,194,253) if q+3<len(steps)}
        m._ST.clear();m._FC_HISTORY.clear();m._ST['tr']=m.BeliefTracker(m._Bel,me);m._ST['prev']=None
        m._ST['rng']=np.random.RandomState(9021)
        def capture(o,cfg,est,base,work,lb,prev,start,locked):
            info['base']=base.copy();info['work']=work.copy();info['locked']=set(locked);return base
        m._search=capture
        for t in range(max(picks)+1):
            obs={k:v for k,v in steps[t][0]['observation'].items() if k!='full_carbon'}
            obs.update(steps[t][me]['observation']);obs.pop('full_carbon',None);obs['player']=me
            pl=obs['players']
            if t in picks or t+1 in picks:
                info={};m._act(obs,cfg,None)
            else:
                m._fc_track(obs)
                tr=m._ST['tr']; est,known=tr.step(pl,t,obs['carbon'])
                m._sense_tree_carbon(tr.bel,obs,cfg,m._ST['prev'],est,known)
            m._ST['tr'].set_actions(pl,steps[t+1][me]['action'] or {});m._ST['prev']=pl
            if t not in picks:continue
            foe=m._ST['v5_history']['opponent'][0]
            joint=risk_module(obs,info['base'],info['work'],foe,info['locked'],timings)
            if joint is not None:
                diff=[u for u in pl[me][2] if info['base'].get(u)!=joint.get(u)]
                assert len(diff)==1 and diff[0] not in info['locked']
                uid=diff[0];p,load,_=pl[me][2][uid]
                before=info['base'].get(uid,'NONE');after=joint.get(uid,'NONE')
                landing=[c._CONVOY_L[p,m.WIDX[a]] for a in (before,after)]
                rival={c._CONVOY_L[q,m.WIDX.get((steps[t+1][op]['action'] or {}).get(v),0)] for v,(q,w,k) in pl[op][2].items() if k=='COLLECTOR' and w<=load}
                actions.append(dict(layout=row['layout'],t=t,side=me,uid=uid,load=load,before=before,after=after,
                                    actual_losing_enemy_lands=[q in rival for q in landing]))
            for uid,(pos,load,k) in pl[op][2].items():
                if k!='COLLECTOR':continue
                group='loaded' if load>=65 else 'light'
                z=foe[pos].astype(float);z-=z.max();prior=np.exp(z);prior/=prior.sum()
                ctx=c._convoy_context(pl,op,pos,load)
                route=ft.observed(steps,t,op,uid,pos)
                paths=dict(fitted=c._convoy_paths(pos,prior,ctx)[1],
                           shortest=baseline(pos,prior,ctx,'shortest'),inertia=baseline(pos,prior,ctx,'inertia'))
                for h in range(3):
                    measures[group,h,'survived'].append(float(len(route)>h))
                    if len(route)<=h:continue
                    q=route[h][0]
                    for name,arr in paths.items():
                        p=arr[h];assert abs(sum(p)-1)<1e-4
                        vals=[-np.log(max(1e-8,p[q])),int(q==p.argmax()),p[q],float(p@p-2*p[q]+1)]
                        measures[group,h,name].append(vals)
                        if h==2 and group=='loaded':maprecords[row['layout'],name].append(float(vals[0]))
    perf={}
    for g0 in ('loaded','light'):
        perf[g0]={h+1:dict(at_risk=len(measures[g0,h,'survived']),observed=int(sum(measures[g0,h,'survived'])),
                      **{k:dict(zip(('nll','hit1','prob_true','brier'),np.mean(measures[g0,h,k],axis=0).round(5))) for k in ('fitted','shortest','inertia')}) for h in range(3)}
    permap={name:[float(np.mean(maprecords[r['layout'],'fitted'])-np.mean(maprecords[r['layout'],name])) for r in rows
                  if maprecords[r['layout'],name]] for name in ('shortest','inertia')}
    def uncertainty(data):
        rng=np.random.RandomState(19);a=np.asarray(data)
        return dict(maps=len(a),improves=int((a<0).sum()),mean=round(a.mean(),5),
                    bootstrap_ci=np.percentile(rng.choice(a,(4000,len(a))).mean(1),[2.5,97.5]).round(5).tolist())
    report=dict(validation_maps=len(rows),training_maps=len(train),training_transitions=model['examples'],target_excluded=model['target_excluded'],
                parent=parent.name,history='Legal seat-visible carbon updated every historical frame. Root net called on each sampled frame and predecessor; earlier style state sparse.',
                metric='Conditional on worker still alive; paths forecast positions at 1,2,3 plies. Same neural first action in all three methods.',
                metrics=perf,loaded_h3_nll_by_map={n:uncertainty(v) for n,v in permap.items()},roots=len(timings),
                ms_p50_p95_p99=np.percentile(timings,[50,95,99]).round(4).tolist(),proposals=len(actions),changes=actions)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('metrics','loaded_h3_nll_by_map','roots','proposals','ms_p50_p95_p99')}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('out',type=Path);p.add_argument('parent',type=Path)
    a=p.parse_args();main(a.cache,a.out,a.parent)
