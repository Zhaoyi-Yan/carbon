"""One legal historical decision per distinct map; compare the original and reconstructed Harbormaster."""
import argparse
import base64
from collections import Counter,defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import dual_weights as pack
from distill_data import load,ROOT
from receipt_data import layout
from receipt_manifest import semantic_layout
from benchmark import validate_actions

def sample_obs(old,s):
    o={k:v for k,v in old[0]['observation'].items() if k!='full_carbon'}
    o.update(old[s]['observation']);o.pop('full_carbon',None);o['player']=s
    return o
def error(p,q):
    d=np.abs(np.asarray(p)-np.asarray(q)).ravel()
    return dict(n=len(d),mae=round(float(d.mean()),8),rmse=round(float(np.sqrt((d**2).mean())),8),
                p95=round(float(np.quantile(d,.95)),8),max=round(float(d.max()),8))
def summary(parts):return dict(correct=int(sum(parts)),total=len(parts),rate=round(float(np.mean(parts)),7))
def main(data,n):
    base=ROOT/'yan-v6.py';harbor=ROOT/'experiments/v7/checkpoints/harbormaster.py'
    snippet=pack.build_frozen_snippet(base)+(data/'dual-weights-init.py').read_text()
    handle=compile(snippet,'combined-weights','exec');tim=[];base_b64=pack.encoded(base)
    for repeat in range(5):
        begin=time.perf_counter();ns={};exec(handle,ns);mid=time.perf_counter()
        new_b64=ns['_dual_harbor_weights'](ns['_WEIGHTS']);end=time.perf_counter()
        assert ns['_WEIGHTS']==base_b64
        tim.append([mid-begin,end-mid,end-begin])
    original=pack.arrays(pack.encoded(harbor));recovered=pack.arrays(new_b64);frozen=pack.arrays(base_b64)
    small_exact=[];deltas=[];drift=[]
    for k,a in original.items():
        b=recovered[k];assert a.shape==b.shape and a.dtype==b.dtype
        if np.array_equal(a,b):small_exact.append(k)
        else:
            assert a.ndim>=2 and a.size>=4096
            deltas.extend((a-b).ravel());drift.extend((a-frozen[k]).ravel())
    delta=np.array(deltas);raw=np.array(drift)
    report=dict(frozen_npz_sha=hashlib.sha256(base64.b64decode(base_b64)).hexdigest(),original_harbor_npz_sha=hashlib.sha256(base64.b64decode(pack.encoded(harbor))).hexdigest(),
               new_npz_sha=hashlib.sha256(base64.b64decode(new_b64)).hexdigest(),frozen_base64_byte_equal=True,
               parent_source_sha={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in (base,harbor)},
               snippet_chars=len(snippet),frozen_chars=len(pack.build_frozen_snippet(base)),extra_chars=len((data/'dual-weights-init.py').read_text()),
               time_s={'columns':['frozen_base85_to_b64','differential_to_npz_base64','both'],'five':np.round(tim,4).tolist()},
               exact_target_layers=small_exact,lossy_scalar_count=len(delta),weight_error=error(delta,np.zeros_like(delta)),
               normalized_error_vs_original_drift=round(float(np.sqrt(np.sum(delta**2)/np.sum(raw**2))),6))
    model=load(harbor);val={k:a.astype('f') for k,a in pack.arrays(pack.encoded(harbor,'_VALUE')).items()}
    real=(model._POLICY,model._net());pp=model._Policy(new_b64);approx=(pp,model._build(pp,val))
    original_router=model._v5_root;stats={};h={}
    def capture_root(o,est,x,prev):
        h['x']=x.copy()
        return original_router(o,est,x,prev)
    model._v5_root=capture_root
    def pre(o,cfg,est,acts,lw,lb,*args):
        h.update(root=dict(acts),w=lw,b=lb)
        return acts
    model._search=pre;agreements=defaultdict(list);differences=defaultdict(list);changes=[];roster=[]
    rows=json.loads((data/'receipts-cache/index.json').read_text())['games'];seen=set()
    rows=sorted(rows,key=lambda r:hashlib.sha256(('dual-harbor-bitpack-v1'+r['layout']).encode()).hexdigest())
    forbidden=set(json.loads((data/'distill-cache/index.json').read_text())['excluded_layouts'])
    for row in rows:
        if row['layout'] in forbidden or row['end']<80:continue
        path=ROOT/row['source']/'replays'/row['file']
        with gzip.open(path) as f:g=json.load(f)
        assert layout(g)==row['layout']
        canon=semantic_layout(g)
        if canon in seen:continue
        seen.add(canon);frames=g['steps'];cfg=g['configuration'];s=int(canon[:2],16)%2
        t=2+int(canon[3:11],16)%(len(frames)-4);model.LAST_TURN=cfg['episodeSteps']-2
        tr=model.BeliefTracker(model._Bel,s);prev=None;tape=np.random.RandomState(model._SEED)
        for j in range(t):
            o=sample_obs(frames[j],s);pl=o['players'];est,known=tr.step(pl,j,o['carbon'])
            model._sense_tree_carbon(tr.bel,o,cfg,prev,est,known)
            act=frames[j+1][s]['action'] or {};tr.set_actions(pl,act);prev=pl
            tape.random_sample(len(pl[s][1])+len(pl[s][2]))
        obs=sample_obs(frames[t],s);pl=obs['players'];my=pl[s]
        options=[]
        for pol,net in (real,approx):
            model._POLICY=pol;model._NET=net;model._RNET=net
            model._ST.clear();model._FC_HISTORY.clear()
            rng=np.random.RandomState();rng.set_state(tape.get_state())
            model._ST.update(tr=copy.deepcopy(tr),prev=prev,rng=rng)
            h.clear();cmd=model._act(obs,cfg,None);validate_actions(cmd,my)
            sid=model._ST['v5_history']['simulation_ids'][s]
            x=h['x'];sc=model._scalars(pl,t,s)
            w,b,v=net(x[None],np.array([sid]),sc[None])
            options.append(dict(root=h['root'],cmd=cmd,work=h['w'],base=h['b'],raw_w=w[0],raw_b=b[0],v=float(v[0]),sid=int(sid)))
        a,b=options;assert a['sid']==b['sid']
        uid=list(my[2]);bid=list(my[1]);units={'worker':(uid,[my[2][u][0] for u in uid],'work','raw_w'),
                                              'base':(bid,[my[1][u] for u in bid],'base','raw_b')}
        for label,(keys,loc,k,rk) in units.items():
            if not keys:continue
            for stage in ('root','cmd'):
                score=[a[stage].get(u)==b[stage].get(u) for u in keys]
                agreements[stage+'_'+label].extend(score)
                for key,match in zip(keys,score):
                    if a[stage].get(key):
                        agreements[stage+'_'+label+'_reference_nonidle'].append(match)
                if label=='worker':
                    for key,match in zip(keys,score):
                        if my[2][key][1]>=65:agreements[stage+'_loaded_workers'].append(match)
            agreements['greedy_'+label].extend(a[k][loc].argmax(1)==b[k][loc].argmax(1))
            agreements['raw_greedy_'+label].extend(a[rk][loc].argmax(1)==b[rk][loc].argmax(1))
            differences['root_'+label].extend((a[k][loc]-b[k][loc]).ravel())
            differences['raw_'+label].extend((a[rk][loc]-b[rk][loc]).ravel())
        for stage in ('root','cmd'):
            agreements[stage+'_whole'].append(a[stage]==b[stage])
        differences['value'].append(a['v']-b['v'])
        if a['root']!=b['root'] or a['cmd']!=b['cmd']:
            changes.append(dict(layout=row['layout'],replay=row['file'],s=s,t=t,root0=a['root'],root1=b['root'],cmd0=a['cmd'],cmd1=b['cmd']))
        roster.append(dict(layout=row['layout'],canonical=canon,replay=row['file'],t=t,seat=s,sid=a['sid']))
        if len(roster)%32==0:print(len(roster),n,flush=True)
        if len(roster)>=n:break
    assert len(roster)==n
    report.update(layouts=n,agreement={k:summary(v) for k,v in agreements.items()},
                  activation_error={k:error(d,np.zeros_like(d)) for k,d in differences.items()},
                  value_3000unit_mae=round(float(np.abs(differences['value']).mean()*3000),5),
                  sid_counts=Counter(r['sid'] for r in roster),changes=changes,states=roster,
                  method='One predetermined decision and one seat per public map, excluding canonical duplicate maps. Historical private own carbon fog, actual own actions and all base/worker random draws replayed from turn zero. Router initialized at sampled root for each model with identical snapshot. The exact Harbormaster _act uses full four-orientation proposals, legal same-quantile root action and cash/postprocessing; root search and final tactical reply skipped for this numerical comparison. Native value/worker/base heads use selected own style. No complete match.')
    (ROOT/'analysis/v7/dual-weights-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['agreement'],report['activation_error'],report['time_s'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('--states',type=int,default=320)
    a=p.parse_args();main(a.data,a.states)
