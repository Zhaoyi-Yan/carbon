"""Add calibrated frozen and collision-pressure controls, whole-layout intervals and an identity audit."""
import argparse,json,os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score
import master_worker as worker
import master_worker_fit as fit


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--data',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--model',type=Path,required=True);a.add_argument('--report',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();torch.set_num_threads(1);model=dict(np.load(args.model));r=json.loads(args.report.read_text())
    rows=json.loads((args.data/'index.json').read_text());train=[v for v in rows if v['fold']=='train']
    by=[(z['base'].astype('f'),z['r'],z['y']) for v in train for z in [dict(np.load(args.data/v['cache']))]]
    Z,R,Y=(np.concatenate([v[i] for v in by]) for i in range(3));del by
    temps=[]
    for role in (0,1):
        b,y=Z[R==role],Y[R==role]
        temps.append(minimize_scalar(lambda v:-log_softmax(b*v,1)[np.arange(len(y)),y].mean(),bounds=(.4,1.8),method='bounded').x)
    test=[v for v in rows if v['fold']=='test'];fits=[];dt=[];tr=[];pr=[];raw=np.zeros(3);old=np.zeros(3);new=np.zeros(3)
    exported_max=0.
    previous=dict(np.load(Path(__file__).with_name('master-model.npz')))
    for meta in test:
        with np.load(args.data/meta['cache']) as v, np.load(args.cache/meta['cache']) as oldfile:
            t=v['t'];roles=v['r'];y=v['y'];base=v['base'].astype('f')
            p=worker._mw_router(v['gs'].astype('f'),model);idx=np.searchsorted(v['tgs'],t);p0=p[idx]
            style=np.stack((1-p0,np.zeros_like(p0),p0),axis=-1).astype('f')
            h=v['h'].astype('f');local=v['l'].astype('f');k=v['k'].astype('f')
            z=worker._mw_logits(h,local,k,roles,base,style,model)
            if not exported_max:
                net=fit.Head({key:model[key] for key in ['cmean','cscale','dscale','lmean','lscale']})
                net.load_state_dict({key:torch.as_tensor(model[key]) for key in net.state_dict()})
                with torch.no_grad():zz=net(*(torch.from_numpy(vv[:18]) for vv in (h,local,k,roles,base,style))).numpy()
                exported_max=float(np.max(np.abs(z[:18]-zz)));assert exported_max<2.e-5
            def loss(vv):return -log_softmax(vv,1)[np.arange(len(y)),y]
            calibrated=base*np.array(temps)[roles,None]
            if meta['teacher']!='Master7sB2':
                tr.extend([meta['teacher']=='Master7sE']*len(p));pr.extend((p>=.5).tolist())
                fits.append([float(np.mean(loss(z)-loss(calibrated))),float(np.mean(np.argmax(z,1)==y)-np.mean(np.argmax(base,1)==y))])
                blocked=(roles==0)&(t>75)&oldfile['blocked'];sel=blocked.sum()
                oracle=fit.old_predict((previous['w0'],previous['s0']),oldfile['wx'][blocked].astype('f'),np.full(sel,fit.NAMES.index(meta['teacher'])))
                for vals,trace in ((raw,calibrated[blocked]),(new,z[blocked]),(old,oracle)):
                    yy=y[blocked];vals+=np.array([sel,-log_softmax(trace,1)[np.arange(sel),yy].sum(),(trace.argmax(1)==yy).sum()])
                vrow=next(rr for rr in r['validation'] if rr['match']==meta['match'])
                dt.extend(vrow['turns'])
    rng=np.random.default_rng(735);d=np.asarray(fits);bootstrap=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
    turns=np.asarray(dt)
    report=dict(torch_numpy_forward_max_error=exported_max,scalar_calibration=temps,D_E_only_maps=len(d),
                per_map_delta_NLL_hit=d.round(5).tolist(),mean_delta_vs_calibrated=[round(x,5) for x in d.mean(0)],
                map_paired_bootstrap_95_NLL_hit=np.percentile(bootstrap,[2.5,97.5],axis=0).round(5).tolist(),
                constrained_heavy=dict(definition='Original prior-worker metric: t>75, load>=65, and an enemy collector no heavier than this worker within two cells.',
                    counts=int(raw[0]),scores={nm:dict(nll=round(c[1]/c[0],5),hit=round(c[2]/c[0],5)) for nm,c in [('frozen_calibrated',raw),('prior_linear_oracle',old),('worker_routed',new)]}),
                source_id_accuracy=round(float(np.mean(np.asarray(tr)==np.asarray(pr))),5),source_id_balanced_accuracy=round(balanced_accuracy_score(tr,pr),5),
                source_id_constant_D_accuracy=round(1.-np.mean(tr),5),
                D_E_local_argmax_team_turns=dict(all=len(turns),different=int((turns[:,1]>0).sum()),more_teacher_moves=int((turns[:,2]>0).sum()),
                                fewer_teacher_moves=int((turns[:,2]<0).sum()),equal_teacher_moves=int((turns[:,2]==0).sum())))
    args.output.write_text(json.dumps(report,indent=1)+'\n');print(report)
if __name__=='__main__':main()
