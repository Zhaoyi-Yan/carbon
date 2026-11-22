"""Compare branch cash predictions under sequential and identity-coupled random sampling."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
from paired_noise import build
from branch_replay import ROOT,OUT,configure_engine
from benchmark import limit_math_threads,load_strategy


def changed(case,mode):
    act=dict(case['searched_action'])
    for x in case['changes'][mode]:
        if x['to_action']=='NONE':act.pop(x['uid'],None)
        else:act[x['uid']]=x['to_action']
    return act


def run(m,roots,cfg,methods):
    # Separate each saved root into pairs, so every simulator receives the same legal
    # root resources and the same real first rival command. Later both sides respond.
    sums=defaultdict(list)
    for seed in (1729,3170,4987):
        for method in methods:
            if method=='modal' and seed!=1729:continue
            for batch in range(0,len(roots),10):
                simulations=[]
                for i,row in enumerate(roots[batch:batch+10],batch):
                    for action in row['moves']:
                        commands=[None,None];commands[row['seat']]=action;commands[1-row['seat']]=row['rival']
                        board=m._turn(row['start'],cfg,commands)
                        simulations.append([i,board,row['start']['players']])
                for n in range(2,11):
                    views=[];ids=[]
                    for i,b,prev in simulations:
                        views.extend(m._v5_pair(b['players'],b['step'],b['carbon'],prev))
                        styles=[20,20];styles[roots[i]['seat']]=22;ids.extend(styles)
                    w,base,_=m._V6R_TINY(np.asarray(views),np.asarray(ids))
                    for j,sim in enumerate(simulations):
                        i,b,prev=sim;pl=b['players'];t=b['step'];act=[]
                        for s in (0,1):
                            k=seed+373*t+17*s
                            if method=='sequential':
                                a=m._sample(pl,s,w[2*j+s],base[2*j+s],np.random.RandomState(k),t)
                            elif method=='worker_id':
                                a=m._sample(pl,s,w[2*j+s],base[2*j+s],None,t,paired_seed=k)
                            else:
                                rng=None if method=='modal' or (method=='alternating' and n%2) else np.random.RandomState(k)
                                scale=1.85 if method=='low_temperature' else 1.
                                a=m._sample(pl,s,scale*w[2*j+s],scale*base[2*j+s],rng,t)
                            act.append(a)
                        sim[1:]=[m._turn(b,cfg,act),pl]
                    if n in (5,10):
                        for j in range(0,len(simulations),2):
                            i,b,_=simulations[j];alt=simulations[j+1][1];s=roots[i]['seat']
                            get=lambda state:state['players'][s][0]-state['players'][1-s][0]
                            sums[i,method,n].append(get(alt)-get(b))
    for i,row in enumerate(roots):
        row['forecast']={method:{str(n):np.round(sums[i,method,n],2).tolist() for n in (5,10)} for method in methods}


def ranks(x):
    _,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    return (counts.cumsum()-.5*counts)[inverse]


def summary(rows,methods):
    result={}
    for method in methods:
        result[method]={}
        for n in ('5','10'):
            v=np.array([r['forecast'][method][n] for r in rows]);y=np.array([r['actual'][n] for r in rows])
            pred=v.mean(1)
            result[method][n]=dict(mean_seed_sd=round(float(v.std(1).mean()),3),mae=round(float(np.abs(pred-y).mean()),3),
                correlation=round(float(np.corrcoef(ranks(pred),ranks(y))[0,1]),3),signed_agreement_beyond_15=round(float((np.sign(pred[np.abs(y)>15])==np.sign(y[np.abs(y)>15])).mean()),3))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('public',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--modal',action='store_true');args=p.parse_args()
    methods=('sequential','modal','low_temperature','alternating') if args.modal else ('sequential','worker_id')
    configure_engine();limit_math_threads();m=load_strategy('paired_cash',args.candidate);m._torch.set_num_threads(1)
    archives=json.load(gzip.open(OUT/'postcargo-actions.json.gz'));bygame=defaultdict(list)
    for case in archives:
        if case['changes']['legacy']!=case['changes']['corrected']:
            bygame[case['match']].append(case)
    idx={r['match']:r for r in json.loads((args.public/'index.json').read_text())};rows=[]
    for match,samples in sorted(bygame.items()):
        g=json.load(gzip.open(args.public/idx[match]['file']));cfg=g['configuration'];s=samples[0]['seat']
        group=[];byturn={c['t']:c for c in samples}
        tr=m.BeliefTracker(m._Bel,s);prev=None
        for t,frame in enumerate(g['steps'][:max(byturn)+1]):
            ob=dict(frame[0]['observation']);ob.update(frame[s]['observation']);ob.pop('full_carbon',None);pl=ob['players']
            est,known=tr.step(pl,t,ob['carbon']);est,known=m._sense_tree_carbon(tr.bel,ob,cfg,prev,est,known)
            cmds=g['steps'][t+1]
            if t in byturn:
                c=byturn[t];gaps={k:round(c['outcomes']['corrected'][k][0]-c['outcomes']['legacy'][k][0],2) for k in ('5','10')}
                group.append(dict(match=match,fold=c['fold'],seat=s,t=t,actual=gaps,
                    start=dict(players=pl,step=t,player=s,carbon=list(est),trees=ob.get('trees',{})),
                    rival=cmds[1-s]['action'] or {},moves=[changed(c,'legacy'),changed(c,'corrected')]))
            tr.set_actions(pl,cmds[s]['action'] or {});prev=pl
        run(m,group,cfg,methods)
        for item in group:
            for key in ('start','rival','moves'):item.pop(key)
        rows+=group
        print(match,len(rows),flush=True)
    metrics={fold:summary([r for r in rows if r['fold']==fold],methods) for fold in ('dev','confirm')}
    out=dict(design='Paired first real rival order, changing only one own root action between two frozen final choosers. Three rollout seeds, separate persistent worker uniforms vs existing sequential sampler. Realized difference uses nine subsequent pairs of recorded commands; simulated successors react using the small policy.',results=metrics,rows=rows)
    (OUT/('tail-consensus.json' if args.modal else 'paired-noise.json')).write_text(json.dumps(out,indent=2)+'\n');print(metrics,flush=True)


if __name__=='__main__':main()
