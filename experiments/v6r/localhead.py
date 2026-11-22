"""Add an actor-only residual scoring the source and five reachable cells at the root."""
import argparse
import base64
import hashlib
import inspect
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PLANES=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,20,35,36,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53)


def _localhead_features(x,hidden,rows,pos,landing):
    dest=landing[pos]
    features=x.reshape(len(x),54,225).transpose(0,2,1)
    target=features[rows[:,None],dest][:,:,_LOCAL_PLANES]
    source=features[rows,pos]
    direction=np.broadcast_to(np.eye(5,dtype=np.float32),(len(pos),5,5))
    extra=np.concatenate((target,source[:,25:30,None],source[:,30:35,None],direction),axis=2)
    return hidden[rows,pos],hidden[rows[:,None],dest],extra


def _localhead_delta(x,h,worker_logits):
    features=x.reshape(1,54,225)
    out=np.zeros((225,5),np.float32)
    for kind,plane in enumerate((2,4)):
        pos=np.flatnonzero(features[0,plane]>.5)
        if not len(pos):continue
        a,b,c=_localhead_features(features,h[None],np.zeros(len(pos),int),pos,_LOCAL_LAND)
        w=_LOCAL_PARAMS
        z=a@w['wa'][kind]+w['bias'][kind]
        z=np.tanh(z[:,None,:]+b@w['wb'][kind]+c@w['wc'][kind])
        delta=(z@w['wo'][kind])[:,:,0]
        # Correct the incumbent at the target cell, before its four rotations are averaged.
        logits=worker_logits[pos]
        out[pos]=(delta-delta.mean(axis=1,keepdims=True))+(w['scale'][kind]-1.)*(logits-logits.mean(axis=1,keepdims=True))
    return out


def build(text,archive):
    anchor='def _v5_rotated_heads(x, opponent_x, mine, rival):'
    prior="            workers += (hidden @ ww + prior['hw.b'])[:, actions] * .25"
    if text.count(anchor)!=1 or text.count(prior)!=1:raise ValueError('Expected rotated elite root')
    packed=base64.b64encode(archive).decode()
    init=("_LOCAL_PLANES = "+repr(PLANES)+"\n_LOCAL_LAND = np.asarray([[move(p,a) for a in range(5)] for p in range(225)])\n"
          "with np.load(io.BytesIO(base64.b64decode('"+packed+"')),allow_pickle=False) as _lh:\n    _LOCAL_PARAMS = dict(_lh)\n")
    text=text.replace(anchor,init+'\n'+inspect.getsource(_localhead_features)+'\n'+inspect.getsource(_localhead_delta)+'\n'+anchor)
    text=text.replace(prior,"""            worker_heads=hidden @ ww + prior['hw.b']
            if side==0:
                # h and input must use the same rotated coordinate system as the landing table.
                native=latent[2*i]
                native_workers=np.empty_like(worker_heads)
                native_workers[cells]=worker_heads
                correction=_localhead_delta(batch[i][0],native,native_workers)
                worker_heads+=correction[cells]
            workers+=worker_heads[:,actions]*.25""")
    compile(text,'<localhead>','exec')
    return text


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('parent',type=Path);a.add_argument('model',type=Path);a.add_argument('output',type=Path);args=a.parse_args()
    parent=args.parent.read_text();model=args.model.read_bytes()
    with np.load(args.model,allow_pickle=False) as z:
        wanted=str(z['backbone'])
    import re
    base=base64.b64decode(re.search(r'^_WEIGHTS = "([^"]+)"',parent,re.M)[1])
    if hashlib.sha256(base).hexdigest()!=wanted:raise ValueError('Incorrect elite backbone weights')
    result=build(parent,model)
    with args.output.open('x') as f:f.write(result)
    print(hashlib.sha256(result.encode()).hexdigest())
