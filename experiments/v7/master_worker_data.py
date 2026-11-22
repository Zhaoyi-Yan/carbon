"""Cache directional hidden cells from the frozen Fairlead root on official Master worker demonstrations."""
import argparse,gzip,json,os,types
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import master_worker as w
from master_data import SAMPLES
from starvation_replay import read_replay
ROOT=Path(__file__).resolve().parents[2]


def load(path):
    src=path.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    anchor='    sides = []\n    for side, player in enumerate((mine, rival)):'
    src=src.replace(anchor,'    _DIRECTIONAL[0]=latent[::2]\n'+anchor)
    m=types.ModuleType('directional_fairlead');m._DIRECTIONAL={};exec(compile(src,str(path),'exec'),m.__dict__);m._torch.set_num_threads(1)
    for k in ('NPL','N','_V5_REACH','_V5_GEOMETRY','_V5_ROTATIONS'):setattr(w,k,getattr(m,k))
    return m


def extract(m,row,path,output,dense=False):
    g=read_replay(path);f=g['steps'];s=row['side'];m._ST.clear();tr=m.BeliefTracker(m._Bel,s);prev=None
    data={k:[] for k in ('h','l','k','r','y','base','t','heavy','pressure','rich','home','press','gs','tgs')}
    for t,fr in enumerate(f[:-1]):
        shared=fr[0]['observation'];pl=shared['players']
        ob=dict(players=pl,player=s,step=t,trees=shared['trees'],carbon=fr[s]['observation']['carbon'])
        cmd=f[t+1][s]['action'] or {}
        est,known=tr.step(pl,t,ob['carbon'])
        est,known=m._sense_tree_carbon(tr.bel,ob,g['configuration'],prev,est,known)
        if t in SAMPLES or (dense and t in (90,164,190)):
            x=m._quant(m.encode(pl,t,s,est,known,prev))
            main=m._v5_root(ob,est,x,prev);ids,r,h,l,k,scene=w._mw_inputs(ob,x,prev,m._DIRECTIONAL[0])
            if ids:
                own=pl[s];pos=[own[2][u][0] for u in ids];y=np.array([m.WIDX.get(cmd.get(u),0) for u in ids])
                heavy=(r==0)&(l[:,1]*170>=65)&(l[:,2]*7<=4.01)
                pressure=(r==0)&(l[:,1]*170<=15)&((l[:,3]*7<=4.01)|(l[:,5]*5<=3.01))
                terrain=set(q for p in pl for q,a in p[3].values())|{q for p in pl for q in p[1].values()}
                rich=(r==0)&(l[:,6]>=.45)&np.asarray([q not in terrain for q in pos])
                press=(k[:,:,3]>0)|(k[:,:,4]>0)
                vals=[h,l,k,r,y,main[0][pos],np.full(len(y),t),heavy,pressure,rich,k[:,:,2]>0,press,scene[None],np.array([t])]
                for key,v in zip(data,vals):data[key].append(v)
        tr.set_actions(pl,cmd);prev=pl;m._ST['prev']=pl
    out={key:np.concatenate(v) for key,v in data.items()}
    for key in ('h','l','k','base','gs'):out[key]=out[key].astype(np.float16)
    temporary=output.with_suffix('.writing')
    with temporary.open('wb') as f:np.savez(f,**out)
    temporary.replace(output)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--cache',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--screenshot',type=Path,required=True)
    args=a.parse_args();args.out.mkdir(exist_ok=True);m=load(ROOT/'experiments/v7/checkpoints/fairlead.py')
    rows=json.loads((args.cache/'index.json').read_text())
    for i,r in enumerate(rows):
        out=args.out/r['cache']
        if not out.exists():extract(m,r,Path(r['file']),out)
        if i%5==0:print(i,r['fold'],flush=True)
    extract(m,dict(side=0),args.screenshot,args.out/'target.npz',dense=True)
    (args.out/'index.json').write_text(json.dumps(rows))

if __name__=='__main__':main()
