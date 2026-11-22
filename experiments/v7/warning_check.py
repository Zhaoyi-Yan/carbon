"""Replay legal resource beliefs and check the frozen shipment alarm, root alternatives and inference parity."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from receipt_check import real_target,targets
from benchmark import load_strategy,limit_math_threads,configure_engine

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('replay',type=Path);p.add_argument('output',type=Path);p.add_argument('--alerts-only',action='store_true');a=p.parse_args()
    limit_math_threads();configure_engine();start=time.perf_counter();m=load_strategy('shipping_audit',a.candidate);init=time.perf_counter()-start
    m._torch.set_num_threads(1)
    game=real_target(a.replay);s=0;cfg=game['configuration'];frames=game['steps'];tr=m.BeliefTracker(m._Bel,s);prev=older=None
    rates=[];values={};tests=[]
    if not a.alerts_only:
        refnet=m._TorchNet(m._POLICY,None);wide=m._WinoNet(refnet);small=m._V7ChannelsLast(refnet)
    select=getattr(m,'_v7_late_alarm',getattr(m,'_v7_shipping_alert',m._v7_root_shipping))
    for t,f in enumerate(frames[:-1]):
        ob=dict(f[0]['observation']);ob.pop('full_carbon',None);pl=ob['players']
        est,known=tr.step(pl,t,ob['carbon']);est,known=m._sense_tree_carbon(tr.bel,ob,cfg,prev,est,known)
        if t in (30,48,75,90,120,164,190,208,245,287):
            b=dict(players=pl,carbon=list(est),trees=ob['trees'],step=t,player=s,_sap_previous=prev,_v7_motion_older=older)
            f0=m._v7_flows(b);out,limit=select(b,f0,s,1,5)
            vals=m._v7_receipt_tail(b);actual,_=targets(game)
            rates.append(dict(t=t,width=5,usual_proposals=1,proposed=limit,p500=np.round(vals[:,0],5).tolist(),cash_estimate=np.round(out,2).tolist(),actual_10=np.round(actual[t:t+10].sum(0),2).tolist()))
            x=m._quant(m.encode(pl,t,s,est,known,prev));tests.append(x)
        tr.set_actions(pl,frames[t+1][s]['action'] or {});older,prev=prev,pl
    for n in (() if a.alerts_only else (1,2,4)):
        # Fill each batch from different real observations. Compare live-unit argmax decisions
        # and full logits before measuring warmed passes with one CPU worker in this audit.
        xs=np.stack(tests[-n:]);ids=np.full(n,22);w,b,_=wide(xs,ids);c,d,_=small(xs,ids)
        workers=xs[:,2].reshape(n,225)+xs[:,4].reshape(n,225)>0;bases=xs[:,12].reshape(n,225)>0
        elapsed=[[],[]]
        for idx in (0,1,1,0,1,0,0,1):
            start=time.perf_counter();(wide,small)[idx](xs,ids);elapsed[idx].append(time.perf_counter()-start)
        values[n]=dict(worker_rows=int(workers.sum()),worker_argmax_diff=int((w[workers].argmax(1)!=c[workers].argmax(1)).sum()),
                       base_rows=int(bases.sum()),base_argmax_diff=int((b[bases].argmax(1)!=d[bases].argmax(1)).sum()),
                       max_worker_logit_delta=round(float(np.abs(w-c).max()),5),max_base_logit_delta=round(float(np.abs(b-d).max()),5),
                       median_ms=[np.median(z)*1000 for z in elapsed])
    q=dict(candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),initialization_seconds=round(init,3),alarms=rates,small_batch_comparison=values)
    a.output.write_text(json.dumps(q,indent=2)+'\n');print('190',next(d for d in rates if d['t']==190));print(values)
if __name__=='__main__':main()
