"""Check embedded heads, bounded antisymmetry and a real call from the frozen search."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import numpy as np
from netcash_data import HERE,ROOT,OUT,DATA
from netcash import build
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,load_strategy,validate_actions
from netcash_train import REPORT


def main():
    configure_engine();os.environ['CARBON_DEBUG']='1'
    with np.load(OUT) as z:
        games=json.loads(str(z['games']));gidx=next(i for i,r in enumerate(games) if r['fold']=='test')
        ixs=np.flatnonzero((z['G']==gidx)&(z['T']>=100))
        sample=ixs[0];t=int(z['T'][sample]);x=z['X'][sample];asset=float(z['B'][sample,2])
    r=games[gidx]
    with gzip.open(ROOT/r['source']/'replays'/r['file']) as f:game=json.load(f)
    source=ROOT/'experiments/v6r/checkpoints/estuary.py'
    code=build(source.read_text())
    extra=[]
    for name in ('quay','breakwater','tidepool','sound','distill-fast-pilot','distill-class-v6'):
        other=DATA/(name+'.py')
        if other.exists():compile(build(other.read_text()),str(other),'exec');extra.append(name)
    m=load_strategy('netcash_integration',source,code.replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)'))
    m._torch.set_num_threads(1)
    frames=game['steps'];obs=frames[t][0]['observation'];pl=obs['players']
    board=dict(step=t,players=pl,trees=obs['trees'],carbon=obs['carbon'])
    m._FC_HISTORY={k:m._fc_summary(frames[k][0]['observation']['players'],frames[k][0]['observation']['trees']) for k in range(max(0,t-36),t+1)}
    public=m._fc_predict(board,298,m._FC_HISTORY);flow=m._v7_receipts(pl,t,298)
    actual=m._v7_nc_inputs(pl,board['trees'],t,298,flow,public,asset)
    x=np.r_[x,asset*np.array([1,max(0,1-t/90.),t/298.,max(0,1-(299-t)/100.)])]
    assert np.allclose(actual,x,rtol=1e-5,atol=.002)
    reverse=m._v7_nc_inputs(pl[::-1],board['trees'],t,298,flow[::-1],-public,-asset)
    assert np.max(np.abs(actual+reverse))<1e-9
    old=m._r6_asset_score(board,0);v=m._v7_nc_leaf(board,0,old)
    assert abs(v-old)<=.045001 and abs(v+m._v7_nc_leaf(board,1,-old))<1e-9
    assert abs(asset-(board['_v7_nc_assets'][0]-board['_v7_nc_assets'][1]))<.01
    assert m._v7_nc_leaf(dict(step=270),0,old)==old
    # Benchmark different fleets and phases, outside any full-game run.
    testmaps=[v for v in games if v['fold']=='test']
    timed=[testmaps[k*len(testmaps)//6] for k in range(6)]
    timers={k:[] for k in ('cached_flow_and_public','cached_public_only')}
    incremental={k:[] for k in timers}
    published=set()
    for path in (ROOT/'analysis/v7').glob('*.jsonl'):
        if 'qualifying' not in path.name and 'confirm' not in path.name:continue
        for line in path.read_text().splitlines():
            published.add(json.loads(line)['layout_sha256'])
    source_layouts=set()
    for game_meta in games:
        with gzip.open(ROOT/game_meta['source']/'replays'/game_meta['file']) as f:initial=json.load(f)['steps'][0][0]['observation']
        canonical=[15,initial['full_carbon'],[sorted(p[1].values()) for p in initial['players']]]
        source_layouts.add(hashlib.sha256(json.dumps(canonical,separators=(',',':')).encode()).hexdigest())
    assert len(source_layouts)==len(games) and not (source_layouts&published)

    for data in timed:
        with gzip.open(ROOT/data['source']/'replays'/data['file']) as f:sm=json.load(f)['steps']
        for tt in (40,100,205,250):
            if tt+35>=len(sm):continue
            so=sm[tt][0]['observation']
            hh={k:m._fc_summary(sm[k][0]['observation']['players'],sm[k][0]['observation']['trees']) for k in range(tt-30,tt+1)}
            bb=dict(step=tt,players=so['players'],trees=so['trees'],carbon=so['carbon'])
            bb['_v7_nc_public']=m._fc_predict(bb,298,hh)
            frozen=m._v7_receipts(bb['players'],tt,298)
            for mode,elapsed in timers.items():
                for _ in range(15):
                    incumbent=dict(bb);test=dict(bb)
                    if mode=='cached_flow_and_public':test['_v7_flow']=frozen
                    start=time.perf_counter_ns();m._r6_asset_score(incumbent,0)
                    old_ms=(time.perf_counter_ns()-start)/1e6
                    start=time.perf_counter_ns();m._v7_nc_leaf(test,0)
                    full_ms=(time.perf_counter_ns()-start)/1e6
                    elapsed.append(full_ms);incremental[mode].append(full_ms-old_ms)
    costs={k:np.percentile(v,[50,90,99]).round(4).tolist() for k,v in timers.items()}
    extra_cost={k:np.percentile(v,[50,90,99]).round(4).tolist() for k,v in incremental.items()}
    # One ordinary legal action, with a small hardware-independent search cap.
    legal={k:obs[k] for k in ('step','carbon','trees','players','player','remainingOverageTime')}
    cfg={k:v for k,v in game['configuration'].items() if k!='randomSeed'}
    m._time=SimpleNamespace(perf_counter=lambda:0.)
    m._S_K=4;m._S_D2=2;m._S_MMAX=1;m._S_M=1
    original=m._v7_nc_leaf;count=0
    def checked(*args):
        nonlocal count
        count+=1
        return original(*args)
    m._v7_nc_leaf=checked
    action=m.agent(legal,cfg);validate_actions(action,pl[0]);assert count>0
    return dict(layout=r['layout'],step=t,compatible_artifacts_compiled=extra,actual_calls_from_search=count,
                timing_layouts=len(timed),timing_calls={k:len(v) for k,v in timers.items()},full_asset_plus_extension_ms_p50_p90_p99=costs,incremental_ms_p50_p90_p99=extra_cost,published_evaluation_layouts_checked=len(published),overlap_with_all_source_maps=0,mirror_checked=True,raw_head_parity_verified=True,
                correction_cap_money_before_asset_mix=135,one_legal_agent_decision=True)


if __name__=='__main__':
    result=main();report=json.loads(REPORT.read_text());report['integration']=result
    REPORT.write_text(json.dumps(report,separators=(',',':'))+'\n');print(result)
