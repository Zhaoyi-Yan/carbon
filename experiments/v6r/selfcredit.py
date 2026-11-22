"""A conservative collector return proposal learned from subsequent verified personal deposits."""
import argparse,hashlib,inspect,json
from pathlib import Path
import numpy as np

PLANES=(2,3,4,5,6,7,8,10,12,13,18,35,36,39,40,42,43,44,45,46,47,50,51)
ROOT=Path(__file__).resolve().parents[2]


def _selfcredit_features(v):
    # v is [worker, action, public plane]; the first of five actions keeps its position.
    at={p:i for i,p in enumerate(_SC_PLANES)}
    def col(p):return v[:,:,at[p]]
    load=np.expm1(np.clip(7*col(3)[:,0],0,7))
    go=14*col(39);foe=14*col(40)
    threatened=(col(35)>.5)&(np.expm1(7*col(36))<=load[:,None]+1.)
    close=(col(42)>.01)&(np.expm1(7*col(43))<=load[:,None]+1.)
    t=col(18)[:,0]
    cargo=np.log1p(load)/5.
    home=go[:,0]/7.;rich=np.minimum(load/120.,2.);late=np.clip((t-.77)*5,0,1)
    s=np.column_stack([cargo,rich,home,home*home,threatened[:,0],close[:,0],t,late,
                       col(50)[:,0],col(51)[:,0],rich*home,late*home])
    bs=np.broadcast_to(s[:,None,:],(len(s),5,s.shape[1]))
    progress=go[:,:1]-go
    halt=np.broadcast_to(np.eye(5,dtype=np.float32)[0],go.shape)
    terms=[progress,progress*rich[:,None],progress*late[:,None],go<.12,threatened,
           close,close*rich[:,None],threatened*rich[:,None],halt,halt*rich[:,None],col(10),
           col(8),col(5),col(6),foe<.12,np.minimum(foe,5.)/5.,(go<.12)*rich[:,None]]
    return np.concatenate((bs,np.stack(terms,axis=2)),axis=2).astype(np.float32),load


def _selfcredit_adjust(x,workers):
    planes=x.reshape(54,225)
    pos=np.flatnonzero(planes[2]>.5)
    if not len(pos):return workers
    v=planes.T[_SC_LAND[pos]][:,:,_SC_PLANES]
    feats,loads=_selfcredit_features(v)
    # The return model's credit labels are defined only on cargo-carrying workers.
    active=loads>=35.
    if not active.any():return workers
    view=feats[active]
    logits=view@_SC_W+_SC_B
    q=1/(1+np.exp(-np.clip(logits,-16,16)))
    # Cap the change to the teacher logit scale; other workers retain their policy.
    delta=.42*(q-q.mean(axis=1,keepdims=True))
    out=workers.copy();out[pos[active]]+=delta
    return out


def build(source,fit):
    code=source.decode()
    mark='def _act(obs, configuration=None, t0=None):'
    point='        lw,lb = _v6_response(obs,est,configuration,lw,lb)'
    if code.count(mark)!=1 or code.count(point)!=1:raise ValueError('Expected elite root')
    add=('_SC_PLANES = '+repr(PLANES)+'\n_SC_LAND = np.array([[move(p,a) for a in range(5)] for p in range(225)])\n'
         +'_SC_W = np.asarray('+repr(fit['weights'])+',dtype=np.float32)\n_SC_B = '+repr(fit['intercept'])+'\n')
    code=code.replace(mark,add+'\n'+inspect.getsource(_selfcredit_features)+'\n'+inspect.getsource(_selfcredit_adjust)+'\n'+mark)
    code=code.replace(point,point+'\n        lw=_selfcredit_adjust(x,lw)')
    compile(code,'<selfcredit>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('model',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();text=build(a.parent.read_bytes(),json.loads(a.model.read_text())['model'])
    with a.output.open('x') as f:f.write(text)
    print(hashlib.sha256(text.encode()).hexdigest())
