"""One unmodified Lifeline backbone, with Master-only channel corrections after every residual layer."""
import argparse
import base64
import inspect
import io
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from master_full import HERE,ROOT,PARENT,original,take_data
from master_multi import elite_data,master_domain,build as shared_build

class Modulated(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        for p in base.parameters():p.requires_grad_(False)
        self.extra=nn.Parameter(base.style.weight[[22,25,26]].detach().clone())
        self.gain=nn.Parameter(torch.zeros(3,7,96));self.shift=nn.Parameter(torch.zeros(3,7,96))
    def forward(self,x,ids,s):
        a=ids>=27;k=(ids-27).clamp(0,2)
        base=self.base
        e=torch.where(a[:,None],self.extra[k],base.style(ids.clamp_max(26)))
        g=1+self.gain[k]*a[:,None,None];b=self.shift[k]*a[:,None,None]
        def change(z,layer):
            return (z*g[:,layer,:,None,None]+b[:,layer,:,None,None]).relu()
        def square(z):return F.pad(z,(1,1,1,1),mode='circular')
        h=change((base.stem(square(x))+e[:,:,None,None]).relu(),0)
        for j,(c1,glob,c2) in enumerate(base.res):
            z=c1(square(h)).relu();z=z+glob(z.mean((2,3)))[:,:,None,None]
            h=change((h+c2(square(z))).relu(),j+1)
        cells=h.flatten(2).transpose(1,2).contiguous()
        project=lambda m:F.linear(cells,m.weight.flatten(1),m.bias)
        v=project(base.pool).relu()
        val=base.v3(base.v2(torch.cat((v.mean(1),v.amax(1),s),1)).relu())[:,0]+s[:,0]
        return torch.cat((project(base.workers),project(base.bases)),2),val

def _mfi_rows(sid):
    ids=np.asarray(sid).reshape(-1)
    if np.max(ids)<27:return None
    global _MFI_GAIN,_MFI_SHIFT
    if not isinstance(_MFI_GAIN,_torch.Tensor):
        _MFI_GAIN=_torch.from_numpy(_MFI_GAIN);_MFI_SHIFT=_torch.from_numpy(_MFI_SHIFT)
    index=np.where(ids>=27,ids-26,0)
    return _MFI_GAIN[index],_MFI_SHIFT[index]

def build(code,weights=HERE/'master_multi_film_weights.npz'):
    code=shared_build(code,weights)
    with np.load(weights) as z:
        f=io.BytesIO();np.savez_compressed(f,g=z['gain'],b=z['shift'])
    payload=base64.b64encode(f.getvalue()).decode()
    helper="with np.load(io.BytesIO(base64.b64decode("+repr(payload)+")),allow_pickle=False) as _mfi_data:\n    _MFI_GAIN=np.concatenate((np.ones((1,7,96),np.float32),1.+_mfi_data['g']))\n    _MFI_SHIFT=np.concatenate((np.zeros((1,7,96),np.float32),_mfi_data['b']))\n"+inspect.getsource(_mfi_rows)+'\n'
    marker='class _TorchNet:'
    def replace(a,b):
        nonlocal code
        if code.count(a)!=1:raise ValueError('Unrecognized Lifeline neural implementation: '+a)
        code=code.replace(a,b)
    replace(marker,helper+marker)
    torch_loop='            for c1w, c1b, gw, gb, c2w, c2b in self.blocks:'
    wino_loop='            for U1, b1, gwT, gb, U2, b2 in self.blocks:'
    for loop,size,after,transform,which in [
        (torch_loop,'len(self.emb)','                h = _torch.relu(h + self._conv(t, c2w, c2b))','h*mfi[0][:,J,:,None,None]+mfi[1][:,J,:,None,None]','c1w, c1b, gw, gb, c2w, c2b'),
        (wino_loop,'len(self.emb)','                h = _torch.relu(h + self._conv(t[self.win], U2, b2, n))','h*mfi[0][:,J,:]+mfi[1][:,J,:]','U1, b1, gwT, gb, U2, b2')]:
        rep='            mfi=_mfi_rows(sid) if '+size+'>=30 else None\n            if mfi is not None:h=_torch.relu('+transform.replace('J','0')+')\n            for jj,('+which+') in enumerate(self.blocks):'
        replace(loop,rep);replace(after,after+'\n                if mfi is not None:h=_torch.relu('+transform.replace('J','jj+1')+')')
    return code

def load_film(weights,device):
    net=Modulated(original()).to(device)
    with np.load(weights) as z,torch.no_grad():
        net.extra.copy_(torch.from_numpy(z['emb'][27:]));net.gain.copy_(torch.from_numpy(z['gain']));net.shift.copy_(torch.from_numpy(z['shift']))
    return net.eval()

def fit(data,out,weight_path=None):
    torch.set_num_threads(1);torch.manual_seed(4217);device='mps' if torch.backends.mps.is_available() else 'cpu';rng=np.random.default_rng(4217)
    split=json.loads((ROOT/'analysis/v7/master-multi-fit.json').read_text());rows=json.loads((data/'master-worker/index.json').read_text())
    md=master_domain(take_data(data,[r for r in rows if r['fold']=='train'],True));archive=elite_data(split['old_train'])
    # Train the extra domain conservatively on a small sample of ordinary legal inputs; original rows are mathematically independent.
    anchors=rng.choice(len(archive['x']),min(1030,len(archive['x'])),replace=False)
    archive={k:v[anchors] for k,v in archive.items()}
    gap=-np.sign(np.choose(archive['sid'],[0]*23+[1,-1,1,-1])).astype(int)
    archive['expert']=np.where(gap==0,27,28+(gap>0))
    d={k:np.concatenate((md[k],archive[k])) for k in archive}
    base=original().to(device).eval();full=original(data/'master-full.py').to(device).eval();net=Modulated(base).to(device)
    optimizer=torch.optim.AdamW([{'params':[net.extra],'lr':.009},{'params':[net.gain,net.shift],'lr':.0042}],weight_decay=.0007)
    for ep in range(5):
        total=0.
        for ii in np.array_split(rng.permutation(len(d['x'])),max(1,len(d['x'])//40)):
            get=lambda k:torch.tensor(d[k][ii],device=device)
            x=get('x').float()*.01;sc=get('s');sid=get('sid');y=get('y').long();m=get('mask').long();dom=get('domain')[:,None]
            z,v=net(x,get('expert'),sc)
            with torch.no_grad():
                old,v0=base(x,sid,sc);teacher,_=full(x,sid,sc)
            loss=0
            for j,(lo,hi) in enumerate(((0,5),(5,8))):
                lp=z[:,:,lo:hi].log_softmax(-1);lq=teacher[:,:,lo:hi].log_softmax(-1);baseq=old[:,:,lo:hi].log_softmax(-1)
                ok=y[:,j]>=0;ce=-lp.gather(2,y[:,j].clamp_min(0)[:,:,None]).squeeze(2)
                w=ok*dom*(1+.4*((m&2)>0)+.45*((m&4)>0) if j==0 else 1.)
                targ=torch.where(dom[:,:,None],lq,baseq)
                kl=(targ.exp()*(targ-lp)).sum(-1);wk=ok.float()+.025
                loss+=(1. if j==0 else .53)*((ce*w).sum()/w.sum().clamp_min(1)+1.2*(wk*kl).sum()/wk.sum().clamp_min(1))
            loss+=1.8*F.smooth_l1_loss(v,v0,beta=.07)+.015*(net.gain.square().mean()+net.shift.square().mean())
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_([net.gain,net.shift,net.extra],1.);optimizer.step();total+=float(loss.detach().cpu())
        print('film epoch',ep+1,round(total,3),flush=True)
    p=weight_path or HERE/'master_multi_film_weights.npz';original_emb=base.style.weight.detach().cpu().numpy()
    package=io.BytesIO()
    np.savez_compressed(package,emb=np.concatenate((original_emb,net.extra.detach().cpu().numpy())),gain=net.gain.detach().cpu().numpy(),shift=net.shift.detach().cpu().numpy())
    payload=package.getvalue()
    if p.exists() and p.read_bytes()!=payload:raise ValueError('Choose a new --weights path for a new fit')
    if not p.exists():p.write_bytes(payload)
    src=build(PARENT.read_text(),p);compile(src,str(out),'exec')
    if out.exists() and out.read_text()!=src:raise ValueError('Choose a new candidate path for a different model')
    if not out.exists():out.write_text(src)
    import hashlib
    desc=dict(source=split['source'],candidate=hashlib.sha256(src.encode()).hexdigest(),weight=hashlib.sha256(p.read_bytes()).hexdigest(),
       master_training=split['master_train'],old_training=split['old_train'],old_held=split['old_hold'],anchors=len(anchors),
       master_samples=len(md['x']),trainable_parameters=sum(a.numel() for a in (net.gain,net.shift,net.extra)),epochs=5,
       design='Original 27 style rows, convolutions, action and value heads are immutable. Three extra shipment-conditional embeddings and seven channel affine layers for Master rows only are learned end-to-end from 42 public Master layouts. 1030 original-style legal boards from the same disjoint 54 original training maps regularize the specialist when a blockade gate occurs in an ordinary match. Ordinary rows follow the inherited network exactly.')
    (ROOT/'analysis/v7/master-multi-film-fit.json').write_text(json.dumps(desc,indent=2)+'\n');print('published',desc['candidate'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('--weights',type=Path);a=p.parse_args();fit(a.data,a.out,a.weights)
