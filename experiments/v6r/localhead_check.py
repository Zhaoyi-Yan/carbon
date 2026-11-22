"""Verify rotation indexing, side isolation and CPU root overhead on an actual held-out board."""
import argparse,gzip,json,time
from pathlib import Path
import numpy as np
from localhead_data import load
from localhead import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path)
    p.add_argument('replays',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    m=load(a.candidate)
    game=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())['validation'][0]
    name=game.rsplit('-',1)[0]
    seat=next(json.loads(l)['seat'] for l in (ROOT/'analysis/v6r-artemis-games.jsonl').read_text().splitlines()
              if json.loads(l)['match']==name and json.loads(l)['name'] in ('Artemis','Apollo'))
    with gzip.open(a.replays/f'{game}.json.gz') as f:r=json.load(f)
    t=107;frame=r['steps'][t];pl=frame[0]['observation']['players'];prev=r['steps'][t-1][0]['observation']['players']
    obs={**frame[0]['observation'],'player':seat,'step':t,'carbon':frame[seat]['observation']['carbon']}
    obs.pop('full_carbon',None)
    tracker=m.BeliefTracker(m._Bel,seat)
    for k,fr in enumerate(r['steps'][:t+1]):
        pp=fr[0]['observation']['players'];cc=fr[seat]['observation']['carbon']
        est,known=tracker.step(pp,k,cc)
        if k<t:tracker.set_actions(pp,r['steps'][k+1][seat]['action'] or {})
    x=m._quant(m.encode(pl,t,seat,est,known,prev))
    other=m._quant(m.encode(pl,t,1-seat,est,[1.]*m.N,prev))
    m._v5_root(obs,est,x,prev)
    net=m._net()
    recorded={}
    def capture(X,ids,hidden=False):
        h=net(X,ids,hidden=hidden);recorded.update(X=X,h=h);return h
    m._net=lambda:capture
    get=lambda:m._v5_rotated_heads(x,other,pl[seat],pl[1-seat])
    augmented=get()
    m._net=lambda:lambda *args,**kwargs:recorded['h']
    compute=m._localhead_delta
    def no_delta(x,h,l):return np.zeros_like(l)
    m._localhead_delta=no_delta
    raw=get()
    # A float32 zero should leave every rival head, teacher offset, and recruitment logit alone.
    np.testing.assert_array_equal(raw[1][0],augmented[1][0])
    np.testing.assert_array_equal(raw[1][1],augmented[1][1])
    np.testing.assert_array_equal(raw[0][1],augmented[0][1])
    for side in (0,1):
        for u in raw[side][2]:
            np.testing.assert_array_equal(raw[side][2][u][1],augmented[side][2][u][1])
    prior=m._POLICY.p;ww=prior['hw.w'].reshape(5,-1).T
    manual=np.zeros((225,5),np.float32)
    for i,(cells,inv,acts,_) in enumerate(m._V5_ROTATIONS):
        native=recorded['h'][2*i]
        delta=compute(recorded['X'][2*i],native,native@ww+prior['hw.b'])
        manual+=delta[cells][:,acts]*.25
    err=float(abs(manual-(augmented[0][0]-raw[0][0])).max())
    assert err<3e-5 and abs(manual).max()>.005
    times=[]
    for _ in range(80):
        t0=time.perf_counter()
        for i in (0,2,4,6):
            h=recorded['h'][i]
            compute(recorded['X'][i],h,h@ww+prior['hw.b'])
        times.append(1000*(time.perf_counter()-t0))
    m._net=lambda:net;m._localhead_delta=compute
    m._ST.update(tr=tracker,prev=prev,rng=np.random.RandomState(m._SEED))
    action=m.agent(obs,r['configuration'])
    assert isinstance(action,dict) and set(action)<=set(pl[seat][1])|set(pl[seat][2])
    result={'map':game,'turn':t,'seat':seat,'max_rotation_reconstruction_error':err,
            'max_change':float(abs(manual).max()),'rival_and_recruit_unchanged':True,
            'four_heads_cpu_ms_median_p95':np.percentile(times,[50,95]).tolist(),
            'legal_agent_action_count':len(action),'cpu_threads':m._torch.get_num_threads()}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(result)


if __name__=='__main__':main()
