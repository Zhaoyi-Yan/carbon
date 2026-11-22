"""Fit factual own-UID gross receipts on complete-map splits, evaluate held-out first actions."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import carrier_action as impl
from receipt_data import layout,targets,ACT

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sig(x):
    return hashlib.sha256(x.encode()).hexdigest()

def canonical(obs):
    a=np.asarray(obs['full_carbon']).reshape(15,15);homes=[list(p[1].values()) for p in obs['players']]
    h=np.zeros(225,dtype=np.uint8)
    for s in (0,1):h[homes[s]]=s+1
    h=h.reshape(15,15);a=np.rint(a*10000).astype('<i4')
    v=[]
    for flip in (False,True):
        for k in range(4):
            c=np.rot90(a,k);p=np.rot90(h,k)
            if flip:c=c[:,::-1];p=p[:,::-1]
            v.extend((c.tobytes()+p.tobytes(),c.tobytes()+np.where(p,3-p,0).astype(np.uint8).tobytes()))
    return hashlib.sha256(min(v)).hexdigest()

def collect(game):
    f=game['steps'];_,events=targets(game);allx=[];ally=[];key=[];pockets=[]
    for t in range(5,len(f)-1,4):
        common=f[t][0]['observation'];pl=common['players']
        for s in (0,1):
            own=pl[s];ob=dict(players=pl,player=s,step=t,carbon=f[t][s]['observation']['carbon'])
            ctx=impl._ca_board(ob);action=f[t+1][s]['action'] or {}
            for uid,(pos,load,role) in own[2].items():
                if role!='COLLECTOR' or load<60:continue
                move=ACT[action.get(uid) or 'NONE'];xx,keys=impl._ca_inputs(ctx,uid,[move])
                future=[evt[uid][1] if uid in evt and evt[uid][0]==s else 0 for evt in events[t:t+20]]
                allx.append(xx[0]);ally.append([sum(future[:10]),sum(future)]);key.append(keys[0][0]);pockets.append(load)
    return np.asarray(allx,'f'),np.asarray(ally,'f'),np.asarray(key,np.int32),np.asarray(pockets,'f')

def export(model,out,k):
    trees=[a[0].nodes for a in model._predictors];nt=len(trees);n=max(len(t) for t in trees)
    f=np.zeros((nt,n),np.int16);th=np.zeros((nt,n),'f8');ch=np.zeros((nt,n,2),np.int16);v=np.zeros((nt,n),'f8')
    for i,t in enumerate(trees):
        for j,p in enumerate(t):
            f[i,j]=p['feature_idx'];th[i,j]=p['num_threshold'];v[i,j]=p['value']
            ch[i,j]=[j,j] if p['is_leaf'] else [p['left'],p['right']]
    out.update({'f'+k:f,'t'+k:th,'c'+k:ch,'v'+k:v,'b'+k:model._baseline_prediction.ravel()[0]})

def metrics(y,p):
    return dict(n=len(y),mae=np.mean(np.abs(y-p),axis=0).round(3).tolist(),
                rmse=np.sqrt(np.mean((y-p)**2,axis=0)).round(3).tolist(),
                actual_mean=y.mean(0).round(3).tolist(),prediction_mean=p.mean(0).round(3).tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--train',type=int,default=128);ap.add_argument('--test',type=int,default=36)
    ap.add_argument('--model',type=Path,default=HERE/'carrier-action-model.npz');ap.add_argument('--out',type=Path,default=ROOT/'analysis/v7/carrier-action-factual.json')
    a=ap.parse_args();a.cache.mkdir(exist_ok=True,parents=True)
    raw=json.loads(gzip.decompress((ROOT/'analysis/v7/branch-breakout-replays.json.gz').read_bytes()));exclude={v['configuration']['randomSeed'] for v in raw.values()}|{205864169}
    banned={canonical(v['initial']['seats'][0]['observation']) for v in raw.values()}
    arch=json.load(open(ROOT/'analysis/v7/branch-original.json'));
    if 'initial' in arch:banned.add(canonical(arch['initial']['seats'][0]['observation']))
    rows=sorted(json.load(open(a.manifest))['games'],key=lambda x:sig('carrier-action-v1'+x['layout']))
    seen=set(banned);seeds=set(exclude);maps=[]
    for r in rows:
        if r['seed'] in seeds or r['end']<50:continue
        f=Path(r['source'])/'replays'/r['file']
        if not f.exists():continue
        with gzip.open(f) as z:g=json.load(z)
        fingerprint=canonical(g['steps'][0][0]['observation'])
        if fingerprint in seen:continue
        assert r['layout']==layout(g);seen.add(fingerprint);seeds.add(r['seed'])
        path=a.cache/(fingerprint+'-current.npz')
        if not path.exists():
            x,y,key,loads=collect(g);np.savez_compressed(path,x=x,y=y,key=key,loads=loads)
        with np.load(path) as tensors:
            if len(tensors['y'])==0:continue
        maps.append(dict(layout=r['layout'],canonical=fingerprint,seed=r['seed'],file=r['file']))
        if len(maps)%32==0:print('maps',len(maps),flush=True)
        if len(maps)==a.train+a.test:break
    assert len(maps)==a.train+a.test and not ({m['seed'] for m in maps}&exclude)
    def join(group):
        cells=[dict(np.load(a.cache/(r['canonical']+'-current.npz'))) for r in group]
        return {k:np.concatenate([p[k] for p in cells]) for k in cells[0]},np.repeat(np.arange(len(group)),[len(p['y']) for p in cells])
    tr,_=join(maps[:a.train]);test,groups=join(maps[a.train:]);x=tr['x'];z=test['x'];mod={'depth':4,'support':np.bincount(tr['key'],minlength=96)}
    specs={};data={}
    for name,n in [('action_agnostic',16),('action_conditional',z.shape[1])]:
        pred=[]
        for h,k in enumerate(('10','20')):
            reg=HistGradientBoostingRegressor(max_iter=66,max_depth=4,learning_rate=.10,max_leaf_nodes=14,min_samples_leaf=36,l2_regularization=16,early_stopping=False,random_state=831)
            reg.fit(x[:,:n],tr['y'][:,h]);p=reg.predict(z[:,:n]);pred.append(p)
            if name=='action_conditional':
                export(reg,mod,k);assert np.max(abs(np.maximum(p,0)-impl._ca_eval(mod,z,k)))<.003
        yhat=np.maximum(0,np.asarray(pred).T);yhat[:,1]=np.maximum(yhat[:,1],yhat[:,0]);data[name]=yhat
    y=test['y'];immediate=(z[:,19]>0);masks={'all':np.ones(len(y),bool),'no_immediate_depot':~immediate,
        'under_130':z[:,0]<1.3,'moving_away':z[:,17]<0,'stay':z[:,16]>0,'heavy_near_threat':(z[:,0]>=1.2)&(z[:,2]<=2)}
    for key,at in masks.items():specs[key]={n:metrics(y[at],pred[at]) for n,pred in data.items() if at.any()}
    delta=np.abs(y-data['action_conditional'])-np.abs(y-data['action_agnostic'])
    md=np.asarray([delta[groups==i].mean(0) for i in range(a.test)])
    rng=np.random.default_rng(1744);bootstrap=md[rng.integers(0,a.test,(4000,a.test))].mean(1)
    np.savez_compressed(a.model,**mod)
    res=dict(train_maps=maps[:a.train],test_maps=maps[a.train:],excluded_seeds=sorted(exclude),excluded_canonical=sorted(banned),
             train_workers=len(x),test_workers=len(y),stride=4,unit_eligibility='Current collector holds >=60; future own-UID gross deliveries at our own base (multiple arrivals counted); engine termination truncates future.',
             labels='Recorded next action is steps[t+1][seat].action; all numeric features use only current observation and first move; resources use filtered carbon from the same seat at t; full_carbon used only to identify and split initial maps.',
             comparisons=specs,per_map_delta_mae=md.round(3).tolist(),equal_map_delta_mae=md.mean(0).round(3).tolist(),
             equal_map_bootstrap_95=np.percentile(bootstrap,[2.5,97.5],axis=0).round(3).tolist(),
             model_sha=hashlib.sha256(a.model.read_bytes()).hexdigest(),model_bytes=a.model.stat().st_size,
             caution='Observed first actions selected by replay policies. Factual prediction scores and action-conditioned deltas do not identify the result of intervening on that action.')
    a.out.write_text(json.dumps(res,indent=1)+'\n');print('done',len(x),len(y),specs['all'],flush=True)
if __name__=='__main__':main()
