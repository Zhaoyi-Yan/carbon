"""Measure fast two-sided policy approximation on alternative first moves from held-out games."""
import argparse,gzip,json,time,os
from pathlib import Path
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import numpy as np
from branch_replay import ROOT,OUT
from benchmark import load_strategy,configure_engine
from postcargo_check import select
from two_sided_rollout import build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('public',type=Path);a=p.parse_args()
    configure_engine();m=load_strategy('two_sided',a.parent,build(a.parent.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')))
    m._torch.set_num_threads(1)
    design=json.loads((OUT/'postcargo-validation.json').read_text())['maps']
    idx={r['match']:r for r in json.loads((a.public/'index.json').read_text())}
    rows=[]
    for d in design:
        s=d['side'];opp=1-s;f=json.load(gzip.open(a.public/idx[d['match']]['file']));cfg=f['configuration'];fs=f['steps']
        wanted=set(select(fs,s,d['map'],12));tr=m.BeliefTracker(m._Bel,s);prev=None
        for t,fr in enumerate(fs[:max(wanted,default=0)+1]):
            o=dict(fr[0]['observation']);o.update(fr[s]['observation']);o.pop('full_carbon',None);pl=o['players']
            e,k=tr.step(pl,t,o['carbon']);e,k=m._sense_tree_carbon(tr.bel,o,cfg,prev,e,k)
            real_moves=[z['action'] or {} for z in fs[t+1]]
            if t in wanted:
                x=m._quant(m.encode(pl,t,s,e,k,prev));w,b=m._v5_root(o,e,x,prev);root=dict(step=t,players=pl,trees=o['trees'],carbon=list(e),player=s)
                act0=real_moves[s];rng=np.random.RandomState(12943+t);options=[act0]
                for temp in (.7,1.,1.4,1.8,2.2,2.6,3.,3.4,3.8,4.,4.4,4.8):
                    c=m._sample(pl,s,w/temp,b/temp,rng,t)
                    if c not in options:options.append(c)
                    if len(options)==8:break
                options=options[1:]+options[:1]
                parts=[];boards=[]
                for c in options:
                    pair=list(real_moves);pair[s]=c;nxt=m._turn(root,cfg,pair)
                    boards.append(nxt);parts.append([len(parts),0,nxt,pl])
                n=len(parts)
                if n>=4:
                    X=np.concatenate([m._v5_pair(z['players'],t+1,z['carbon'],pl)[[s,opp]] for z in boards]);S=np.stack([m._scalars(z['players'],t+1,i) for z in boards for i in (s,opp)])
                    sides=[s,opp]*n
                    tw,tb,tv=m._net()(X,m._v5_ids(sides),S,play_sides=sides)
                    refs,ar=m._v7_actor_references(X,parts)
                    row=dict(map=d['map'],fold=d['fold'],seat=s,t=t,width=n,references=len(refs));truth=.5*(tv[::2]-tv[1::2])
                    for tag,fn in [('previous',m._v6r_counterfactuals),('paired',lambda *args:m._v7_joint_interim(*args,safeguard=False)),('guarded',m._v7_joint_interim)]:
                        start=time.perf_counter();aw,ab,v=fn(m._net(),X,S,opp,parts,s);elapsed=time.perf_counter()-start
                        rates=[];errors=[]
                        for i in (0,1):
                            mask=(X[i::2,2]+X[i::2,4]).reshape(n,225)>0
                            z=aw[i::2][mask];gt=tw[i::2][mask];rates.append(float((z.argmax(1)==gt.argmax(1)).mean()) if len(z) else 1.)
                            errors.append(float(abs((z-z.mean(1,keepdims=True))-(gt-gt.mean(1,keepdims=True))).mean()) if len(z) else 0.)
                        truth_likelihood=[]
                        commands=[z['action'] or {} for z in fs[t+2]]
                        for j,who in enumerate((s,opp)):
                            own=boards[-1]['players'][who];worker=aw[2*(n-1)+j]
                            known=fs[t+1][0]['observation']['players'][who][2]
                            keys=[u for u in own[2] if u in known and own[2][u][:2]==known[u][:2]]
                            if not keys:
                                truth_likelihood.append(None);continue
                            probs=worker[[own[2][u][0] for u in keys]];probs=probs-probs.max(1,keepdims=True)
                            y=[m.WIDX.get(commands[who].get(u),0) for u in keys]
                            loss=np.log(np.exp(probs).sum(1))-probs[np.arange(len(y)),y]
                            truth_likelihood.append(float(loss.mean()))
                        row[tag]=dict(actual_nll=truth_likelihood,ms=round(1000*elapsed,4),argmax=rates,logit_mae=errors,
                            centered_value_mae=float(np.abs((v-v[0])-(truth-truth[0])).mean()),best_agrees=bool(v.argmax()==truth.argmax()))
                    rows.append(row)
            tr.set_actions(pl,real_moves[s]);prev=pl
        print(d['match'],len(rows),flush=True)
    summary={}
    for fold in ('dev','confirm'):
        q=[r for r in rows if r['fold']==fold];summary[fold]={}
        for method in ('previous','paired','guarded'):
            summary[fold][method]={k:np.mean([r[method][k] for r in q],axis=0).round(4).tolist() for k in ('argmax','logit_mae','centered_value_mae','best_agrees','ms')}
            summary[fold][method]['actual_nll']=[round(float(np.mean([r[method]['actual_nll'][i] for r in q if r[method]['actual_nll'][i] is not None])),4) for i in (0,1)]
    (OUT/'two-sided-rollout.json').write_text(json.dumps(dict(results=summary,rows=rows),indent=2)+'\n');print(summary)
if __name__=='__main__':main()
