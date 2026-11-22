"""Replay-drive the root and time one extra tactical joint. Analysis instrumentation is not deployed."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import types
import numpy as np
from master_spatial import build
from master_spatial_fit import ROOT,PARENT
from starvation_replay import read_replay
from receipt_data import layout
from receipt_manifest import semantic_layout

TARGET='03100a95e8647c51a4d642fe08d856dd'
def readobs(f,s):
    obs={**f[0]['observation'],**f[s]['observation'],'player':s}
    obs.pop('full_carbon',None)
    return obs
def external(data):
    groups=json.loads((data/'master-worker/index.json').read_text());used={r['group'] for r in groups}
    blocked={semantic_layout(read_replay(Path(r['file']))) for r in groups}
    used.update(json.loads((data/'distill-cache/index.json').read_text())['excluded_layouts'])
    rows=[(r['mid'],Path(r['path']),r['s'],r['name']) for r in json.loads((ROOT/'analysis/v7/master-spatial-external.json').read_text())['sources']]
    for _,p,_,_ in rows:
        g=read_replay(p);used.add(layout(g));blocked.add(semantic_layout(g))
    for r in json.loads((data/'distill-public/index.json').read_text()):
        if len(rows)==10:break
        m=r['match'];p=data/'distill-public'/r['file']
        if m in (TARGET,'4063a34afedb6d4767a9118e1dee9e5d') or any(m==a[0] for a in rows):continue
        g=read_replay(p);key=layout(g);can=semantic_layout(g)
        if key in used or can in blocked:continue
        used.add(key);blocked.add(can);rows.append((m,p,r['seat'],'published QQ-v6'))
    assert len(rows)==10
    return rows
def instrument(code):
    marker="    _mx_extra=None\n    if most>=3"
    code=code.replace(marker,"    _AUD['before']=[dict(c) for c in cands]\n    _AUD['fixed']=dict(fixed or {})\n    _AUD['most']=most\n    _AUD['before_ms']=_time.perf_counter()\n"+marker)
    marker="    if len(cands) < 2:\n        _S_STATS['single']"
    code=code.replace(marker,"    _AUD['proposal_ms']=1000*(_time.perf_counter()-_AUD['before_ms'])\n    _AUD['proposed']=_mx_extra\n    _AUD['after']=[dict(c) for c in cands]\n"+marker)
    m=types.ModuleType('expert_tactical_probe');m._AUD={}
    exec(compile(code.replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)'),'<tactical_probe>','exec'),m.__dict__)
    m._torch.set_num_threads(1)
    return m
def delta(a,b):
    return {k:[a.get(k,'STAY'),b.get(k,'STAY')] for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
def evaluate(m,g,s,t,full_history,seq,production=False):
    m._ST.clear();m._AUD.clear();cfg=g['configuration'];frames=g['steps'];m.LAST_TURN=cfg['episodeSteps']-2
    search=m._search;reply=m._reply_choice;m._search=lambda obs,cfg,est,act,*args:act;m._reply_choice=lambda obs,cfg,est,act,*args:act
    # Full previous online inputs for the screenshot. Other ten maps mirror the belief exactly and warm six root observations.
    for i in range(t):
        obs=readobs(frames[i],s);act=frames[i+1][s]['action'] or {};pl=obs['players']
        if full_history or i>=t-6:
            if production and i>=t-5:m._search=search
            m._act(obs,cfg,time.perf_counter())
        else:
            if i==0:
                m._ST['tr']=m.BeliefTracker(m._Bel,s);m._ST['prev']=None;m._ST['rng']=np.random.RandomState(m._SEED)
            e,k=m._ST['tr'].step(pl,i,obs['carbon']);m._sense_tree_carbon(m._ST['tr'].bel,obs,cfg,m._ST['prev'],e,k)
            m._ST['rng'].random_sample(len(pl[s][1])+len(pl[s][2]))
        m._ST['tr'].set_actions(pl,act);m._ST['liquidity_action']=act;m._ST['prev']=pl
    obs=readobs(frames[t],s);actual=frames[t+1][s]['action'] or {};roles=m._mx_roles(obs)
    args=[];root_elapsed=[]
    def capture(*a):
        root_elapsed.append(time.perf_counter()-a[7]);args.extend(a);return a[3]
    m._search=capture
    m._act(obs,cfg,time.perf_counter());m._search=search;m._reply_choice=reply
    # Freeze the same modest search plan, sufficient for real simulations; alternate evaluation order.
    old=(m._S_K,m._S_D2,m._S_ABORT)
    if not production:m._S_K=6;m._S_D2=min(2,old[1]);m._S_ABORT=3
    cost=m._S_COST[:] if production else [.001,.001,.001,.001]
    fn=m._mx_tactical;comparison={}
    for use in ([False,True] if seq%2==0 else [True,False]):
        m._S_COST[:]=cost;m._AUD.clear();m._mx_tactical=fn if use else (lambda *a,**k:None)
        tic=time.perf_counter();args[7]=tic-root_elapsed[0] if production else tic;choice=search(*args);elapsed=time.perf_counter()-tic
        comparison[use]=dict(before=m._AUD['before'],after=m._AUD['after'],proposal=m._AUD['proposed'],fixed=m._AUD['fixed'],
                             most=m._AUD['most'],proposal_ms=m._AUD['proposal_ms'],end=choice,total_ms=elapsed*1000)
    m._mx_tactical=fn;m._S_K,m._S_D2,m._S_ABORT=old
    off,on=comparison[False],comparison[True]
    assert on['before']==on['after'][:len(on['before'])]
    if not production:assert off['after']==on['before']
    base=on['before'][0];extra=on['proposal'];changed=delta(base,extra) if extra is not None else {}
    assert set(changed)<=set(roles) and not set(changed)&set(on['fixed'])
    players=obs['players'];own,foe=players[s],players[1-s]
    context=m._ST['v5_history']['mx_context']
    return dict(step=t,seat=s,budget='normal' if production else 'fixed',planner_unit_cost=cost,input_root_ms=round(root_elapsed[0]*1000,3),disabled_width=len(off['before']),seed=cfg.get('randomSeed'),layout=layout(g),cash=own[0],units=[len(own[2]),len(foe[2])],old_gate=10<=t<=185 and own[0]<60 and len(own[3])<=2 and len(foe[2])>=len(own[2])+2,
       active=roles,locked=on['fixed'],frozen_expert22_cached=context[2] is not None,original_candidates=len(on['before']),after=len(on['after']),proposed=extra is not None,unique=len(on['after'])>len(on['before']),
       changed=changed,actual_moves_on_changed={k:actual.get(k,'STAY') for k in changed},selected_proposal=extra is not None and on['end']==extra,
       search_changed=delta(off['end'],on['end']),proposal_ms=round(on['proposal_ms'],3),baseline_ms=round(off['total_ms'],3),with_ms=round(on['total_ms'],3),total_delta_ms=round(on['total_ms']-off['total_ms'],3))
def main(data,target,mode,production):
    os.environ['CARBON_DEBUG']='1';code=build(PARENT.read_text(),mode=mode);m=instrument(code)
    if production:
        result=evaluate(m,read_replay(target),0,190,True,0,True)
        dest=ROOT/('analysis/v7/master-spatial-'+mode+'-production.json');dest.write_text(json.dumps(result,indent=2)+'\n');print('production',result,flush=True);return
    rows=external(data);out=[];schedule=(52,73,88,109,124,142,158,173,194,215)
    targetrow=evaluate(m,read_replay(target),0,190,True,0);print('target',targetrow,flush=True)
    for i,(mid,path,s,name) in enumerate(rows):
        g=read_replay(path);upper=len(g['steps'])-2;step=min(upper,schedule[i]);row=evaluate(m,g,s,step,False,i+1);row.update(match=mid,observed=name);out.append(row)
        print('external',i+1,step,row['unique'],row['proposal_ms'],row['total_delta_ms'],flush=True)
    timing={k:{'p50':round(float(np.median([r[k] for r in out])),3),'p95':round(float(np.quantile([r[k] for r in out],.95)),3)} for k in ('proposal_ms','total_delta_ms','baseline_ms','with_ms')}
    report=dict(parent_sha=hashlib.sha256(PARENT.read_bytes()).hexdigest(),candidate_sha=hashlib.sha256(code.encode()).hexdigest(),
        target_match=TARGET,target=targetrow,archive=out,timing=timing,method='Options differ by a single appended tactical joint; all old candidates are asserted identical and in the same order, and changes to bases or protected units are rejected. width six plus one, depth two, planner budget 3s; single CPU, enabled/disabled ordering alternates. Inputs legal player carbon, archived actual historical orders. Target warms all 190 root observations, other maps have exact fog/movement tape and six previous root router updates. Search comparison ends before downstream safeguards. Total timings include normal local CPU noise and the additional simulations; no whole-game claim.')
    (ROOT/('analysis/v7/master-spatial-'+mode+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(timing,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('target',type=Path);p.add_argument('--mode',choices=('tactical','tactical-secondary'),default='tactical');p.add_argument('--production-target',action='store_true');a=p.parse_args();main(a.data,a.target,a.mode,a.production_target)
