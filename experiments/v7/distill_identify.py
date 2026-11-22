"""Fit non-v6 replies using only opposite seats on training layouts; evaluate prospective class gating."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from distill import _ds_logprob
from distill_fit import load,fit
from distill_id import _ds_identify
import distill_id
from receipt_data import LAND


def pool(rows):
    full=np.concatenate([v for _,v in rows]);gr=defaultdict(list)
    for k,v in rows:gr[k].append(v)
    maps=np.array([np.concatenate(v).mean(0) for v in gr.values()])
    boot=np.random.RandomState(3402).choice(maps[:,0],(5000,len(gr))).mean(1)
    return dict(maps=len(gr),units=len(full),base_nll=round(float(full[:,1].mean()),5),
                gated_nll=round(float(full[:,2].mean()),5),base_hit1=round(float(full[:,3].mean()),5),
                gated_hit1=round(float(full[:,4].mean()),5),base_mass=round(float(full[:,5].mean()),5),
                gated_mass=round(float(full[:,6].mean()),5),gated_weight=round(float(full[:,7].mean()),4),
                gated_active=round(float((full[:,7]>0).mean()),4),macro_gain=round(float(maps[:,0].mean()),5),
                improves=int((maps[:,0]<0).sum()),paired95=np.quantile(boot,[.025,.975]).round(5).tolist())
def main(a):
    ix=json.loads((a.cache/'index.json').read_text());pos=json.loads(a.pos.read_text())
    neg_ix=json.loads((a.neg/'index.json').read_text());trainids=set(pos['train_layouts'])
    assert {r['group'] for r in neg_ix}<=trainids
    ds=[load(a.neg,r) for r in neg_ix];cat=lambda k:np.concatenate([d[k] for d in ds])
    x,z,y,role=map(cat,('x','z','y','role'));wc=[];ss=[]
    for k in (0,1):
        use=role==k;zr=z[use];st=max(.05,fit(zr[:,:,None],y[use],np.zeros_like(zr),.0005)[0])
        ss.append(st);wc.append(fit(x[use],y[use],st*zr,.035))
    ncoef=np.array(wc);ns=np.array(ss);pc=np.array(pos['coefficients']);ps=np.array(pos['scale'])
    if a.light:
        remove=[1,2,25]  # The two expert logits and their cargo interaction.
        x[:,:,remove]=0.
        posdata=[load(a.cache,r) for r in ix['records'] if r['fold']=='train' and not r['negative']]
        xp=np.concatenate([p['x'] for p in posdata]);xp[:,:,remove]=0.
        zp=np.concatenate([p['z'] for p in posdata]);rp=np.concatenate([p['role'] for p in posdata]);yp=np.concatenate([p['y'] for p in posdata])
        for k in (0,1):
            i=role==k;ncoef[k]=fit(x[i],y[i],ns[k]*z[i],.035)
            j=rp==k;pc[k]=fit(xp[j],yp[j],ps[k]*zp[j],.014)
    a.weights.write_text(json.dumps(dict(scale=ns.round(7).tolist(),coefficients=ncoef.round(7).tolist(),
                                 layouts=[r['group'] for r in neg_ix],negative_actions=len(y),positive_scale=ps.round(7).tolist(),positive_coefficients=pc.round(7).tolist(),extra_net=not a.light),separators=(',',':'))+'\n')
    distill_id._ROLLOUT_LANDING=[list(v) for v in LAND]
    result=defaultdict(list)
    for r in ix['records']:
        if r['fold']!='test':continue
        assert r['group'] not in trainids
        g=load(a.cache,r);roles=g['role'];z=g['z'];x=g['x'];y=g['y'];ii=np.arange(len(y))
        if a.light:x[:,:,remove]=0.
        positive=np.einsum('nak,nk->na',x,pc[roles])+ps[roles,None]*z
        generic=np.einsum('nak,nk->na',x,ncoef[roles])+ns[roles,None]*z
        odds=_ds_logprob(positive)-_ds_logprob(generic);weights=np.zeros(len(y));mem={}
        for t in np.unique(g['t']):
            here=np.flatnonzero(g['t']==t)
            units={int(g['uid'][i]):[int(g['pos'][i])] for i in here}
            alpha=_ds_identify(mem,int(t),units);weights[here]=alpha
            mem['pending']={int(g['uid'][i]):(int(g['pos'][i]),odds[i]) for i in here}
        baseline=z/.4
        proposal=positive*np.sqrt(1./(.4*ps[roles,None]))
        gated=(1.-weights[:,None])*baseline+weights[:,None]*proposal
        q0=_ds_logprob(baseline)[ii,y];q1=_ds_logprob(gated)[ii,y]
        meas=np.column_stack((q0-q1,-q0,-q1,baseline.argmax(1)==y,gated.argmax(1)==y,np.exp(q0),np.exp(q1),weights))
        who='other' if r['negative'] else 'v6'
        for collection in (who,r['source']+'-'+who):result[collection].append((r['group'],meas))
    report=dict(negative_games=len(neg_ix),negative_workers=len(cat('y')),negative_training_layouts=[r['group'] for r in neg_ix],
                positive_training_layouts=sorted(trainids),cohorts={k:pool(v) for k,v in result.items()},
                caveat='Exploratory gate revision after reviewing the first gate on these same held-out maps. All fit parameters exclusively from training layouts. Need fresh downstream games.')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(report['cohorts'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('neg',type=Path)
    p.add_argument('pos',type=Path);p.add_argument('weights',type=Path);p.add_argument('report',type=Path)
    p.add_argument('--light',action='store_true',help='Use only already-available root logits and board features.')
    main(p.parse_args())
