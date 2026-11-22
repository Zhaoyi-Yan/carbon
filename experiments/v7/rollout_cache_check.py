"""Check that the transport proposals and simulated team moves preserve the original choices."""
import argparse,collections,gzip,json,sys,time
from pathlib import Path
import numpy as np
from branch_replay import OUT,configure_engine
from benchmark import limit_math_threads,load_strategy
from rollout_cache import build
from receipt_data import layout

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('public',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();old=load_strategy('cache_before',a.parent)
    import types
    m=types.ModuleType('cache_after')
    exec(compile(build(a.parent.read_text()),'<cached simulation>','exec'),m.__dict__);m._torch.set_num_threads(1)
    samples=collections.defaultdict(lambda:[[],[]]);maps=[];sampled=0
    for r in json.loads((a.public/'index.json').read_text())[:15]:
        g=json.load(gzip.open(a.public/r['file']));frames=g['steps'];cfg=g['configuration'];side=r['seat'];maps.append(layout(g))
        tr=m.BeliefTracker(m._Bel,side);prev=None
        for t,f in enumerate(frames[:-1]):
            o=dict(f[0]['observation']);o.update(f[side]['observation']);o.pop('full_carbon',None);o['player']=side
            pl=o['players'];e,k=tr.step(pl,t,o['carbon']);cmds=frames[t+1][side]['action'] or {}
            if t%13==4:
                for mode in ('tree','bank','intercept'):
                    choices=[];tim=samples[mode]
                    for i,mod in enumerate((old,m)):
                        start=time.perf_counter_ns();choices.append(mod._v7_plan(o,cfg,e,cmds,mode));tim[i].append((time.perf_counter_ns()-start)/1e6)
                    if choices[0]!=choices[1]:raise AssertionError((r,t,mode,choices))
                rng=np.random.RandomState(37+t);w=rng.normal(size=(225,5));b=rng.normal(size=(225,3))
                for seed in (317+t,918+t):
                    for s in (0,1):
                        before=old._sample(pl,s,w,b,np.random.RandomState(seed),t)
                        after=m._sample(pl,s,w,b,m._V7RandomTape(seed),t)
                        if before!=after:raise AssertionError((r,t,s,before,after))
                        sampled+=1
            tr.set_actions(pl,cmds);prev=pl
    times={}
    for k,items in samples.items():
        times[k]={'cases':len(items[0]),'mean_ms':[round(float(np.mean(v)),4) for v in items],'p95_ms':[round(float(np.percentile(v,95)),4) for v in items]}
    begin=time.perf_counter()
    for j in range(10000):np.random.RandomState((j//8)%128).random_sample(12)
    elapsed=time.perf_counter()-begin
    begin=time.perf_counter()
    for j in range(10000):m._V7RandomTape((j//8)%128).random_sample(12)
    other=time.perf_counter()-begin
    data=dict(maps=maps,identical_sampled_teams=sampled,identical_proposals=times,draw_microseconds=[round(v*100,2) for v in (elapsed,other)])
    (OUT/'rollout-cache.json').write_text(json.dumps(data,indent=2)+'\n');print(data)
if __name__=='__main__':main()
