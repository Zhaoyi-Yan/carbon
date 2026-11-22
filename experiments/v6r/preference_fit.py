"""Fit a pairwise planter-state preference on partial-view, one-turn counterfactuals."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import preference

ROOT=Path(__file__).resolve().parents[2]
TEACHERS={'226891089ba7f54b198c8f8dc771dfc2','dd1f117a64c676225f0c6333b1cdd1e7'}


def fingerprint(players):
    return [(round(p[0],2),sorted((q,round(c,3),r) for q,c,r in p[2].values()),sorted(p[3].values())) for p in players]


def sample(meta,d,model):
    raw=json.load(gzip.open(d/meta['file']))
    frames=raw['steps'];cfg=raw['configuration']
    me=next(i for i,k in enumerate(meta['order']) if meta['strategies'][k]['id'] in TEACHERS)
    h=hashlib.sha256(meta['id'].encode()).digest()
    offset=h[1]%15
    tracker=model.BeliefTracker(model._Bel,me)
    samples=[];audits=0; rejected=0
    for t,fr in enumerate(frames[:-1]):
        if t>185:break
        o=fr[0]['observation'];pl=o['players']
        estimate,known=tracker.step(pl,t,fr[me]['observation']['carbon'])
        actions=[dict(r['action'] or {}) for r in frames[t+1]]
        tracker.set_actions(pl,actions[me])
        if t<12 or (t-12-offset)%22:continue
        ids=sorted(k for k,w in pl[me][2].items() if w[2]=='PLANTER')
        if not ids:continue
        wid=ids[(h[2]+t)%len(ids)]
        actual=model.WIDX.get(actions[me].get(wid),0);group=[]
        root=dict(step=t,player=me,players=pl,trees=o['trees'],carbon=estimate,remainingOverageTime=5)
        if audits==0:
            # This branch only verifies the factual referee state; its full map never reaches features.
            ref=model._turn(dict(root,carbon=o['full_carbon']),cfg,actions)
            if fingerprint(ref['players'])!=fingerprint(frames[t+1][0]['observation']['players']):
                rejected+=1
                continue
            audits+=1
        for j in range(5):
            trial=[dict(a) for a in actions]
            if j:trial[me][wid]=model.WACT[j]
            else:trial[me].pop(wid,None)
            post=model._turn(root,cfg,trial)
            group.append(preference._pref_features(root,post,me,wid))
        prev=frames[t-1][0]['observation']['players']
        x=model.encode(pl,t,me,estimate,known,prev)
        x=np.clip(np.rint(x*100.),0,255)*.01
        lw=model._POLICY.forward(x,model._SID)[0]
        prior=lw[pl[me][2][wid][0]]
        samples.append((group,actual,prior))
    return samples,audits,rejected


def train(x,l,priors=None):
    positive=x[np.arange(len(x)),l]
    mask=np.arange(x.shape[1])[None,:]!=l[:,None]
    z=(positive[:,None,:]-x)[mask]
    offset=0. if priors is None else (priors[np.arange(len(l)),l,None]-priors)[mask]
    ridge=.017 if priors is None else .065
    w=np.zeros(x.shape[-1]);scale=np.maximum(z.std(axis=0),.16);a=z/scale
    for step in range(360):
        y=1/(1+np.exp(np.clip(offset+a@w,-28,28)))
        w+=.20*(a.T@y/len(y)-ridge*w)
    return w/scale


def score_logits(s,l):
    target=s[np.arange(len(l)),l];d=target[:,None]-s
    mask=np.arange(s.shape[1])[None,:]!=l[:,None]
    ties=np.isclose(d,0,atol=1e-8)
    best=s.max(axis=1)
    top=np.isclose(target,best,atol=1e-8)/np.isclose(s,best[:,None],atol=1e-8).sum(axis=1)
    return dict(n=len(l),top1=round(float(top.mean()),4),pair_agreement=round(float(((d>1e-8)+ties*.5)[mask].mean()),4))


def score(x,l,w):
    return score_logits(x@w,l)


METHOD = {
    'question':'Can one-turn action-conditioned planter position features add predictive information to a frozen v5 policy head?',
    'data':'90 full games by pinned Artemis/Apollo, deterministic game-level split. For each position one planter and five actions, all other recorded actions fixed. The map for model features comes only from the corresponding seat belief.',
    'factual_audit':'One factual transition per map reproduced from full referee input. Cash, sorted worker role/position/cargo and tree position/age must match the recorded next frame. Full map is never used in learned features.',
    'unit':'top1 ties receive split credit; pair_agreement counts a tie as half. Fixed positional weights are in preference.py. Per-game holdout applies to the new preference model. The existing v5 policy already learned from historical teacher replays.',
    'deployment':'Standalone diagnostic proposes one alternative direction only for a moving planter, using one simulated opponent reply estimated by v5. It preserves original stationary planting, locked worker moves, friendly landings and current recruitment reservations.',
    'decision':'Archive as an experiment. It beats the hand-written feature weights but the added learned residual does not improve overall held-out direction accuracy on top of the existing policy. No complete arena evaluation was run.',
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--maps',type=int,default=90)
    a=p.parse_args()
    sp=importlib.util.spec_from_file_location('ref',ROOT/'yan-v5.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    preference.NB=m.NB;preference.DIST=m.DIST
    rows=[r for r in json.loads((a.data/'index.json').read_text()) if any(s['id'] in TEACHERS for s in r['strategies'].values())]
    rows.sort(key=lambda r:hashlib.sha256(('planter-pref-v1:'+r['id']).encode()).digest())
    groups={'train':[],'holdout':[]}; valid_names=[]; tags=collections.defaultdict(list);audit=reject=0
    for i,row in enumerate(rows[:a.maps]):
        fold='holdout' if i%3==0 else 'train'
        g,n,r=sample(row,a.data,m)
        if fold=='holdout':
            teacher=next(s['name'] for s in row['strategies'].values() if s['id'] in TEACHERS)
            valid_names.extend([teacher]*len(g))
        groups[fold].extend(g);audit+=n;reject+=r
        tags[fold].append(dict(match=row['id'],seat=next(i for i,k in enumerate(row['order']) if row['strategies'][k]['id'] in TEACHERS),positions=len(g)))
    def arrays(g):return np.array([x for x,y,logits in g]),np.array([y for x,y,logits in g]),np.array([logits for x,y,logits in g])
    tx,tl,tp=arrays(groups['train']);vx,vl,prior=arrays(groups['holdout']);w=train(tx,tl)
    result=dict(method=METHOD,features=preference.FEATURES,weights=w.tolist(),maps=tags,
                replay_transition_checks_passed=audit,failed_factual_checks_skipped=reject,
                training=score(tx,tl,w),validation=dict(soft_fixed=score(vx,vl,np.array(preference.FIXED)),
                learned=score(vx,vl,w),always_stay=float((vl==0).mean())))
    for idx,name in enumerate(('stay','move')):
        keep=(vl==0) if idx==0 else (vl!=0)
        result['validation'][name]={k:score(vx[keep],vl[keep],v) for k,v in [('fixed',np.array(preference.FIXED)),('learned',w)]}
    moving=vl!=0
    result['validation']['movement_direction_only']={
        'fixed':score(vx[moving,1:],vl[moving]-1,np.array(preference.FIXED)),
        'learned':score(vx[moving,1:],vl[moving]-1,w),
        'frozen_v5_policy_head':score_logits(prior[moving,1:],vl[moving]-1)}
    tm=tl!=0
    residual=train(tx[tm,1:],tl[tm]-1,tp[tm,1:])
    result['residual_weights']=residual.tolist()
    baseline=prior[moving,1:]
    combined=baseline + vx[moving,1:]@residual
    def extra(mask):
        target=vl[moving][mask]-1
        before=baseline[mask]
        after=combined[mask]
        def nll(s):
            m=s.max(axis=1,keepdims=True)
            return round(float(np.mean(np.log(np.exp(s-m).sum(axis=1))-(s-m)[np.arange(len(target)),target])),4)
        return dict(policy=score_logits(before,target),policy_plus_preference=score_logits(after,target),
                    policy_nll=nll(before),combined_nll=nll(after),
                    changed=int((np.argmax(before,axis=1)!=np.argmax(after,axis=1)).sum()))
    result['validation']['residual']=dict(all=extra(np.ones(moving.sum(),bool)),
                      by_teacher={teacher:extra(np.array(valid_names)[moving]==teacher) for teacher in sorted(set(valid_names))})
    # The deployable rule proposes only a change among directions already chosen to move.
    a.output.write_text(json.dumps(result,indent=1)+'\n');print(result['validation'],flush=True)

if __name__=='__main__':
    main()
