"""Reconstruct only the first eight completed V7 tournament batches into legal per-seat critic features."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,base64,hashlib,io,json,re
from pathlib import Path
import numpy as np
from branch_replay import HERE,OUT,ROOT,step,board_digest
from benchmark import configure_engine,new_game,observe,load_strategy

SOURCE=HERE/'checkpoints/lifeline.py'
FROZEN='9c5bdda7b8d1a9cc33af4cd0fe03edcb4a34a2d42e027d9b06fed519ff6b949d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
    if sha(SOURCE)!=FROZEN:raise ValueError('unexpected Lifeline checkpoint')
    src=SOURCE.read_text();weights=dict(np.load(io.BytesIO(base64.b64decode(re.search(r'^_VALUE = "([^"]+)"',src,re.M)[1]))))
    module=load_strategy('selfplay_value_data_lifeline',SOURCE,src.replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)'))
    module._torch.set_num_threads(1);return module,weights


class Features:
    def __init__(self,m,side):
        self.m=m;self.s=side;self.track=m.BeliefTracker(m._Bel,side);self.prev=None;self.pressure=None;self.expert=None

    def history(self,obs,cfg,econ,extract):
        m=self.m;s=self.s;pl=obs['players'];t=obs['step'];delta=econ[s]-econ[1-s]
        raw=.8*max(0.,-delta[5]/4.)+.7*max(0.,-delta[7])+.4*max(0.,delta[3]/3.)
        self.pressure=.84*(raw if self.pressure is None else self.pressure)+.16*raw
        self.expert=21 if self.pressure>(1.20 if self.expert==21 else 1.80) and t>32 and pl[s][0]<pl[1-s][0]+350 else 22
        stock,known=self.track.step(pl,t,obs['carbon']);stock,known=m._sense_tree_carbon(self.track.bel,obs,cfg,self.prev,stock,known)
        if not extract:return None
        # Simulated search leaves use the full *estimated* stock after the root belief update.
        self.root_stock=stock
        x=m._v5_pair(pl,t,stock,self.prev)[s]
        sid=m._v7_shipping_id(self.expert,float(m._v7_shipping_gap(x)))
        return x,sid,m._scalars(pl,t,s)

    def mark(self,obs,action):
        self.track.set_actions(obs['players'],action);self.prev=obs['players']


def features(m,weights,samples):
    x=np.array([z[0] for z in samples],dtype='f');sid=np.array([z[1] for z in samples],np.int64);sc=np.array([z[2] for z in samples],dtype='f')
    net=m._net();h=net(x,sid,hidden=True)
    z=np.maximum(h@weights['v1.weight'][:,:,0,0].T+weights['v1.bias'],0)
    inp=np.column_stack((z.mean(1),z.max(1),sc));latent=np.maximum(inp@weights['v2.weight'].T+weights['v2.bias'],0)
    old=sc[:,0]+(latent@weights['v3.weight'].T)[:,0]+weights['v3.bias'][0]
    return latent,old,sc


def replay(m,weights,row):
    st,env=new_game(row['seed'],300);cfg=env.configuration;agents=[Features(m,i) for i in (0,1)];inputs=[];times=[];sides=[]
    cash=[[v[0] for v in st[0].observation.players]];tape=row['actions_and_post_cash']
    for t,(cmd,expected) in enumerate(tape):
        states=[observe(st,s,t) for s in (0,1)];marg,_=m._v5_observe(states[0]['players'],states[0]['trees'],t,298)
        sample=t>=5 and (t-5)%9==row['seed']%9
        for s,tr in enumerate(agents):
            r=tr.history(states[s],cfg,marg,sample)
            if r is not None:inputs.append(r);times.append(t);sides.append(s)
            tr.mark(states[s],cmd[s])
        step(st,env,cmd);actual=[v[0] for v in st[0].observation.players]
        if actual!=expected:raise ValueError((row['seed'],t,actual,expected))
        cash.append(actual)
    assert [v.reward for v in st]==row['scores']
    cash=np.asarray(cash);end=len(tape);tt=np.array(times);ss=np.array(sides);margin=cash[:,0]-cash[:,1]
    all_h=np.column_stack([margin[np.minimum(tt+h,end)]*(1-2*ss) for h in (20,52,100)])
    yy=all_h@np.array([.28,.52,.20])/3000.
    chunks=[features(m,weights,inputs[i:i+16]) for i in range(0,len(inputs),16)]
    return dict(h=np.concatenate([v[0] for v in chunks]),old=np.concatenate([v[1] for v in chunks]),sc=np.concatenate([v[2] for v in chunks]),
                y=yy.astype('f'),horizons=all_h.astype('f'),win=(np.array(row['scores'])[ss]>np.array(row['scores'])[1-ss]),t=tt,s=ss,
                active=np.array([v[1] for v in inputs]),turns=len(tape))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);a=p.parse_args();a.cache.mkdir(exist_ok=True,parents=True)
    configure_engine();m,weights=load();layouts={};manifest=[];excluded=[]
    for batch in range(1,9):
        source=OUT/f'{batch:02d}-qualifying.jsonl'
        for i,line in enumerate(source.read_text().splitlines()):
            g=json.loads(line);layout=g['layout_sha256']
            if g['status']!='completed':excluded.append([batch,g['seed'],i,g['status']]);continue
            assert layouts.setdefault(layout,batch)==batch
            name=f'{batch:02d}-{i:02d}-{g["seed"]}.npz';dest=a.cache/name
            if not dest.exists():np.savez_compressed(dest,**replay(m,weights,g))
            manifest.append(dict(batch=batch,layout=layout,seed=g['seed'],file=name,trajectory=i,strategies=[v['name'] for v in g['players']]))
        print('reconstructed batch',batch,'completed total',len(manifest),flush=True)
    split=list(manifest);assert len(layouts)==64 and len(manifest)==127 and len(excluded)==1
    unique_train={r['layout'] for r in split if r['batch']<=6};test={r['layout'] for r in split if r['batch']>=7};assert not unique_train&test
    (a.cache/'index.json').write_text(json.dumps(dict(maps=split,excluded=excluded,files={str(i):sha(OUT/f'{i:02d}-qualifying.jsonl') for i in range(1,9)}),indent=1)+'\n')

if __name__=='__main__':main()
