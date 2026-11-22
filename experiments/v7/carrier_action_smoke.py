"""Compile the optional exporter and check actual Harbormaster root invocation on a public holdout."""
import gzip,json,os
from types import SimpleNamespace
import master_e_branch as arena
import carrier_action as carrier


def main():
    arena.limit_math_threads();arena.configure_engine();os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    path=arena.HERE/'checkpoints/harbormaster.py'
    base=path.read_text();code=carrier.build(base)
    if arena.shot(path)!=arena.FROZEN['harbormaster']:raise ValueError('frozen candidate changed')
    start=code.index('    if _S_COST[0] > 0.0:',code.index('def _search('));end=code.index('    stages =',start)
    code=code[:start]+'    most,D2,M=4,min(D2,2),1\n'+code[end:]
    for a,b in [('_torch.set_num_threads(2)','_torch.set_num_threads(1)'),('for tick in range(1,15):','for tick in range(1,4):'),
                ('return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT','return False')]:
        assert code.count(a)==1;code=code.replace(a,b)
    m=arena.load_strategy('readonly_carrier_action_smoke',path,code);m._time=SimpleNamespace(perf_counter=lambda:0.);m._S_MMAX=1
    assert m._SEARCH_OK
    assert len(m._CA_LAND)==225 and all(a==b for a,b in zip(m._CA_LAND,carrier.LAND))
    orig=m._ca_proposal;trace=[]
    def record(obs,cmd,locks,prev,model):
        out,meta=orig(obs,cmd,locks,prev,model)
        assert (out is None)==(meta is None)
        assert (out,meta)==carrier._ca_proposal(obs,cmd,locks,prev,model)
        if meta:assert meta['load']>=60 and meta['uid'] not in locks and sum(cmd.get(k)!=out.get(k) for k in cmd.keys()|out.keys())==1
        trace.append([obs['step'],meta]);return out,meta
    m._ca_proposal=record
    g=json.loads(gzip.decompress((arena.OUT/'branch-breakout-replays.json.gz').read_bytes()))['loss']
    state,env=arena.restore(g['initial'],g['configuration']);cfg={k:v for k,v in g['configuration'].items() if k!='randomSeed'}
    real=m._search,m._reply_choice
    targets={55,95,140,164}
    for t in range(max(targets)+1):
        assert arena.board_digest(state)==g['expected'][t]
        if t in targets:m._search,m._reply_choice=real
        else:m._search=m._reply_choice=lambda obs,cfg,est,cmd,*rest:cmd
        obs=arena.observe(state,1,t);result=m.agent(obs,cfg);arena.validate_actions(result,obs['players'][1])
        if t==max(targets):break
        g0=g['actions'][t]
        m._ST['tr'].set_actions(obs['players'],g0[1]);m._ST['liquidity_action']=dict(g0[1])
        arena.step(state,env,g0)
    assert {row[0] for row in trace}==targets
    report=dict(model_sha=arena.shot(arena.HERE/'carrier-action-model.npz'),exported_size=len(carrier.build(base).encode()),root=trace)
    (arena.OUT/'carrier-action-smoke.json').write_text(json.dumps(report,indent=1)+'\n')
    print('Embedded model matched standalone checks on all four actual root entries:',trace)

if __name__=='__main__':main()
