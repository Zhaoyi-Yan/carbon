"""Check two standalone exports and actual root-candidate insertion on the held-out Master7sE replay."""
import argparse,ast,copy,hashlib,json,os,time,types
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from master_worker_data import ROOT
import master_worker as mw
from starvation_replay import read_replay
from benchmark import validate_actions


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--target',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
    args=a.parse_args();g=read_replay(args.target);cfg=g['configuration'];outputs={}
    for name in ('soundings','fairlead'):
        file=ROOT/'experiments/v7/checkpoints'/(name+'.py');text=file.read_text();src=mw.build(text)
        compile(src,'worker-'+name,'exec')
        weights=next(a.value.value for a in ast.parse(text).body if isinstance(a,ast.Assign) and isinstance(a.targets[0],ast.Name) and a.targets[0].id=='_WEIGHTS')
        outputs[name]=dict(export_sha256=hashlib.sha256(src.encode()).hexdigest(),size=len(src.encode()),shared_weights_literal_sha256=hashlib.sha256(weights.encode()).hexdigest())
        if name=='fairlead':
            # Remove timing variability solely for the smoke test: the original search still generates and evaluates the candidates.
            start=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));end=src.index('    stages =',start)
            fixed=src[:start]+'    most,D2,M=5,min(D2,2),1\n'+src[end:]
            fixed=fixed.replace('keys.add(_mw_key);cands.append(_mw_action)',"keys.add(_mw_key);cands.append(_mw_action);_MW_ADMITTED.append(step)")
            m=types.ModuleType('master_worker_export_smoke');m._MW_ADMITTED=[]
            exec(compile(fixed.replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)'),'<bounded-root>','exec'),m.__dict__)
    assert outputs['fairlead']['shared_weights_literal_sha256']==outputs['soundings']['shared_weights_literal_sha256']
    m._torch.set_num_threads(1);m.LAST_TURN=298
    joint=m._mw_joint;trace=[];always=m._search;reply=m._reply_choice
    def capture(obs,base,heads,locked,seed):
        start=time.perf_counter();p=joint(obs,base,heads,locked,seed)
        s=obs['player'];pl=obs['players'][s];validate_actions(p,pl)
        assert {k:v for k,v in p.items() if k in pl[1]}=={k:v for k,v in base.items() if k in pl[1]}
        assert all(p.get(k)==base.get(k) for k in (locked or {}))
        trace.append(dict(t=obs['step'],changed_workers=sum(p.get(k)!=base.get(k) for k in pl[2]),protected=len(locked or {}),
                          joint_millis=round(1000*(time.perf_counter()-start),3),proposal=p))
        return p
    m._mw_joint=capture
    infer=m._mw_propose;timings=[]
    def timed(*args):
        start=time.perf_counter();out=infer(*args);timings.append((time.perf_counter()-start)*1000);return out
    m._mw_propose=timed
    for t,fr in enumerate(g['steps'][:165]):
        s=0;board=fr[0]['observation'];pl=board['players']
        obs=dict(players=pl,player=s,step=t,trees=board['trees'],carbon=fr[s]['observation']['carbon'],remainingOverageTime=5.)
        m._search=always if t in (90,164) else lambda obs,cfg,est,act,*rest:act
        m._reply_choice=reply if t in (90,164) else lambda obs,cfg,est,act,*rest:act
        executed=m.agent(obs,cfg)
        if trace and trace[-1]['t']==t:
            trace[-1]['actually_admitted']=t in m._MW_ADMITTED
            trace[-1]['final_command_equals_proposal']=(executed==trace[-1]['proposal'])
        act=g['steps'][t+1][s]['action'] or {};m._ST['tr'].set_actions(pl,act)
    assert trace and len({row['t'] for row in trace})==2,trace
    report=dict(standalone=outputs,source='4063a34afedb6d4767a9118e1dee9e5d',seat=0,
                test='Existing root search, fixed width five and depth two at turns 90 and 164; full fog history, source actions only before each decision. Proposed alternatives keep current recruits and locked workers.',
                observed=trace,inference_ms_p50_p95=np.percentile(timings,[50,95]).round(4).tolist())
    args.output.write_text(json.dumps(report,indent=1)+'\n');print(report)

if __name__=='__main__':main()
