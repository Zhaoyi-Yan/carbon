"""Check packaged inference and run paired full games under the ordinary time budget."""
import gzip
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from collect import write
from network import ROOT,PARENT,make
from dataset import V7
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/v8'),str(ROOT/'experiments/decision_audit')]
from scripts.selfplay import source_for
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game
from arena import Strategy,run_game
from report import audit
from rank_batch import stats


def native(path,out):
    configure_engine();limit_math_threads()
    module=load_strategy('elite_check_'+path.stem,path)
    assert module._SEARCH_OK and module._RX
    row=next(r for r in json.loads((out/'split.json').read_text()) if r['fold']=='validation')
    with np.load(out/'encoded'/(row['game']+'.npz')) as z:
        x=z['X'][:20].astype('f')*.01;s=z['S'][:20]
    ids=np.array([27+i%10 for i in range(len(x))])
    report={}
    for small,backend in ((False,module._net()),(True,module._V6R_TINY)):
        # Rebuild the training reference from its original exported tensors, separately
        # from the submission's encoded blobs and its runtime backend.
        model=make(PARENT.read_text(),small)
        suffix='small' if small else 'main'
        with np.load(out/(path.stem+'-'+suffix+'.npz')) as z:
            params=dict(model.named_parameters())
            for k in params:
                if k=='style.weight':key='emb'
                else:
                    prefix,field=k.rsplit('.',1)
                    if prefix.startswith('res.'):
                        _,idx,layer=prefix.split('.');prefix='b'+idx+'.'+('c1','g','c2')[int(layer)]
                    prefix={'workers':'hw','bases':'hb','pool':'v1'}.get(prefix,prefix)
                    key=prefix+'.'+(field if prefix in ('v1','v2','v3') else field[0])
                with torch.no_grad():params[k].copy_(torch.from_numpy(z[key]))
        with torch.no_grad():z,v=model(torch.from_numpy(x),torch.from_numpy(ids),torch.from_numpy(s))
        w,b,val=backend(x,ids,s)
        err=[float(np.max(np.abs(i-j))) for i,j in zip((w,b,val),(z[:,:,:5].numpy(),z[:,:,5:].numpy(),v.numpy()))]
        assert max(err)<.015,err
        report[suffix]=err
    return report


def panel(out):
    configure_engine();limit_math_threads()
    seeds=[int(hashlib.sha256(f'elite-refit/1124/holdout/{i}'.encode()).hexdigest()[:8],16)%10**9 for i in range(10)]
    files={}
    for name in ('yan-v6','yan-v7'):
        _,sha,src=source_for(name,ROOT)
        p=out/(name+'.py');p.write_bytes(src);files[name]=Strategy(name,p,sha)
    assert files['yan-v7'].sha256==V7
    for arm in ('top6','top6-plus'):
        path=out/(arm+'.py');files[arm]=Strategy(arm,path,hashlib.sha256(path.read_bytes()).hexdigest())
    matchups=[(a,'yan-v7',seeds[:4]) for a in ('top6','top6-plus')]
    matchups.extend((a,'yan-v6',seeds[4:6]) for a in ('yan-v7','top6','top6-plus'))
    protocol=dict(screening=seeds[:6],confirmation=seeds[6:],sha256={k:v.sha256 for k,v in files.items()},
                  gate='Screen: at least 6/8 vs v7, at least 3 per seat and positive mean; then 4 independent paired maps, same gate.',
                  planned=matchups)
    write(out/'protocol.json',protocol)
    archives=ROOT/'analysis'
    previous=[archives/'v8'/p for p in ('team/freshness.json','opening-and-grove/freshness.json','payback-layouts.json','option-layouts.json')]
    previous.append(archives/'decision-audit/rank-cycle-1/freshness.json')
    subprocess.run([sys.executable,str(ROOT/'experiments/v8/freshness.py'),str(out),'--previous',*map(str,previous)],check=True)
    layouts={r['layout'] for r in json.loads((out/'split.json').read_text())}
    for seed in seeds:
        state,_=new_game(seed,300);board=state[0].observation
        key=hashlib.sha256(json.dumps([board.full_carbon,[sorted(p[1].values()) for p in board.players]],separators=(',',':')).encode()).hexdigest()
        assert key not in layouts
    results={}
    with ProcessPoolExecutor(3) as pool,gzip.open(out/'arena.jsonl.gz','wt') as log:
        def play(matchups,label):
            jobs=[]
            for a,b,maps in matchups:
                for seed in maps:
                    for side in (0,1):
                        seats=[files[a],files[b]]
                        jobs.append((a,b,side,(seats[::-1] if side else seats,seed,300,True)))
            for (a,b,side,_),row in zip(jobs,pool.map(run_game,[j[-1] for j in jobs])):
                log.write(json.dumps(dict(stage=label,**row))+'\n')
                parsed=audit(row,protocol,candidate=a)
                assert parsed['seat']==side
                key=f'{label}/{a}/{b}'
                results.setdefault(key,[]).append(parsed)
        play(matchups,'screen')
        confirm=[]
        for a in ('top6','top6-plus'):
            summary=stats(results[f'screen/{a}/yan-v7'])
            if summary['valid']==8 and min(summary['seat_wins'])>=3 and summary['average']>0:
                confirm.append((a,'yan-v7',seeds[6:]))
        play(confirm,'confirmation')
    write(out/'arena.json',{k:dict(summary=stats([r for r in v if r['turns']==299]),games=v) for k,v in results.items()})


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    write(out/'parity.json',{arm:native(out/(arm+'.py'),out) for arm in ('top6','top6-plus')})
    panel(out)
