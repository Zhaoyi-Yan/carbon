"""Check conditionally modulated PyTorch and Winograd inference against the training network and frozen original."""
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
from dual_weights import arrays,encoded
from master_full import ROOT,PARENT,HERE,take_data
from master_multi_film import build,load_film

def module(source):
    m=types.ModuleType('standalone_native');text=source.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)');exec(compile(text,str(source),'exec'),m.__dict__);return m
def main(data,file):
    torch.set_num_threads(1);weights=HERE/'master_multi_film_weights.npz';src=build(PARENT.read_text(),weights)
    compile(src,str(file),'exec');sha=hashlib.sha256(src.encode()).hexdigest()
    if file.exists():
        if file.read_text()!=src:raise ValueError('Existing candidate differs. Choose another path for a new build.')
    else:file.write_text(src)
    fit=ROOT/'analysis/v7/master-multi-film-fit.json';j=json.loads(fit.read_text());j['candidate']=sha;fit.write_text(json.dumps(j,indent=2)+'\n')
    models=[module(p) for p in (PARENT,file)];pure=load_film(weights,'cpu');dataset=json.loads((data/'master-worker/index.json').read_text())
    rows=take_data(data,[r for r in dataset if r['fold']=='test'],False)
    ids=np.random.RandomState(211).choice(len(rows['x']),160,replace=False);X=rows['x'][ids].astype(np.float32)*.01;S=rows['s'][ids];active=(np.arange(len(ids))%2)==0
    base_ids=rows['sid'][ids];sid=base_ids.copy();sid[active]=27+np.arange(active.sum())%3
    raw=[m._TorchNet(m._POLICY,arrays(encoded(src if i else PARENT,'_VALUE'))) for i,m in enumerate(models)]
    out={}
    for tag,backends in [('torch',raw),('winograd',[m._WinoNet(t) for m,t in zip(models,raw)])]:
        errors=[[],[],[]];old_errors=[[],[],[]];equal=[0,0];count=[0,0]
        for ix in np.array_split(np.arange(len(X)),20):
            pred=backends[1](X[ix],sid[ix],S[ix]);ref=backends[0](X[ix],base_ids[ix],S[ix])
            with torch.inference_mode():o,v=pure(torch.from_numpy(X[ix]),torch.from_numpy(sid[ix]),torch.from_numpy(S[ix]))
            truth=(o[:,:,:5].numpy(),o[:,:,5:].numpy(),v.numpy())
            for k in range(3):
                keep=active[ix];errors[k].append(np.max(np.abs(pred[k][keep]-truth[k][keep])))
                old_errors[k].append(np.max(np.abs(pred[k][~keep]-ref[k][~keep])))
                if k<2:
                    which=(rows['y'][ids[ix],k]>=0)&keep[:,None]
                    equal[k]+=int((pred[k][which].argmax(1)==truth[k][which].argmax(1)).sum());count[k]+=int(which.sum())
        assert all(max(v)==0. for v in old_errors)
        assert all(max(v)<.004 for v in errors)
        out[tag]=dict(original_max_abs=[float(max(v)) for v in old_errors],master_max_abs=[float(max(v)) for v in errors],worker_base_equal=equal,worker_base_total=count)
    base=arrays(encoded(PARENT));packed=arrays(encoded(src));assert all(np.array_equal(v,packed[k] if k!='emb' else packed[k][:27]) for k,v in base.items())
    out.update(candidate=sha,source_states=160,layout_count=len(set(rows['map'][ids])),original_parameter_arrays_bit_exact=True)
    (ROOT/'analysis/v7/master-multi-film-native.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('file',type=Path);a=p.parse_args();main(a.data,a.file)
