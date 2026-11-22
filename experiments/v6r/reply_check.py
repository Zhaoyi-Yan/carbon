"""One-ply unit and paired public-replay counterfactual checks; no game is played."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def synthetic(m):
    for me in (0,1):
        ours=[80.,{'my-base':0},{'valuable':[112,150.,'COLLECTOR']},{}]
        enemy=[80.,{'op-base':204},{'hunter':[113,0.,'COLLECTOR']},{}]
        players=[ours,enemy] if me==0 else [enemy,ours]
        w=np.zeros((225,5),np.float32);w[113,4]=10.
        own=np.zeros((225,5),np.float32);own[112,0]=4.
        base=np.zeros((225,3),np.float32);base[:,0]=9.
        m._ST.clear();m._ST['v5_history']={'opponent':(w,base)}
        obs=dict(players=players,step=70,player=me,trees={})
        cf=dict(m.DEFAULT_CONFIG)
        root={**obs,'carbon':np.zeros(225)}
        trap={'hunter':'LEFT'}
        def advance(action):
            return m._turn(root,cf,[action,trap] if me==0 else [trap,action])
        rescued=m._reply_choice(obs,cf,root['carbon'],{},own,set(),time.perf_counter())
        assert 'valuable' not in advance({})['players'][me][2]
        assert 'valuable' in advance(rescued)['players'][me][2]
        assert m._reply_choice(obs,cf,root['carbon'],{},own,{'valuable'},time.perf_counter())=={}
        enemy[2]['hunter'][0]=37
        assert m._reply_choice(obs,cf,root['carbon'],{},own,set(),time.perf_counter())=={}


def economy(board,me):
    def own(p):
        return p[0]+sum(row[1] for row in p[2].values())
    a,b=board['players'][me],board['players'][1-me]
    return [a[0]-b[0],own(a)-own(b),len(a[2]),len(b[2])]


def paired(m,r,turns):
    frames=r['steps'];cf=r['configuration']
    for me in (0,1):
        m._ST.clear();prev=None
        belief=m.BeliefTracker(m._Bel,me)
        for t,(before,after) in enumerate(zip(frames,frames[1:])):
            full=before[0]['observation'];pl=full['players'];carbon=before[me]['observation']['carbon']
            est,known=belief.step(pl,t,carbon)
            original=after[me]['action'] or {}
            if t in turns or (t,me) in turns:
                obs=dict(players=pl,trees=full['trees'],step=t,player=me,carbon=carbon)
                x=m._quant(m.encode(pl,t,me,est,known,prev))
                ours,_=m._v5_root(obs,est,x,prev)
                m._ST['prev']=prev
                homes=set(pl[me][1].values());enemyhomes=list(pl[1-me][1].values())
                locked={uid for uid,(pos,cargo,role) in pl[me][2].items()
                        if role=='COLLECTOR' and m.go_home(pos,cargo,m._probs(ours[pos]),x,homes,enemyhomes,t,m._HOME) is not None}
                start=time.perf_counter()
                revised=m._reply_choice(obs,cf,est,original,ours,locked,start)
                elapsed=time.perf_counter()-start
                rival=after[1-me]['action'] or {}
                true_root={**obs,'carbon':full['full_carbon']}
                def apply(action):
                    return m._turn(true_root,cf,[action,rival] if me==0 else [rival,action])
                baseline=apply(original)
                updated=apply(revised) if revised!=original else baseline
                actual=after[0]['observation']['players']
                error=max(abs(baseline['players'][i][0]-actual[i][0]) for i in range(2))
                counts=all(len(baseline['players'][i][2])==len(actual[i][2]) for i in range(2))
                yield dict(step=t,seat=me,seed=cf['randomSeed'],changed=revised!=original,
                           move_change={k:[original.get(k),revised.get(k)] for k in set(original)|set(revised) if original.get(k)!=revised.get(k)},
                           seconds=round(elapsed,5),baseline_cash_error=round(error,3),baseline_unit_match=counts,
                           saved_carriers=[uid for uid,(p,v,k) in pl[me][2].items() if k=='COLLECTOR' and v>=50
                                           and uid not in baseline['players'][me][2] and uid in updated['players'][me][2]],
                           delta=(np.array(economy(updated,me))-economy(baseline,me)).round(3).tolist(),
                           tactical_proxy_delta=round(m._reply_assets(updated,me,cf)-m._reply_assets(baseline,me,cf),3))
            belief.set_actions(pl,original);prev=pl


def historical_losses(m,replay):
    # Evaluation-only stress stratum: first three historical laden-collector losses per seat.
    turns=set()
    for me in (0,1):
        count=0
        for t,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
            pl=before[0]['observation']['players']
            future=after[0]['observation']['players'][me][2]
            if any(k=='COLLECTOR' and v>=50 and uid not in future and
                   any(ek=='COLLECTOR' and ev<=v and m.FEAT_DIST[p,q]<=2 for q,ev,ek in pl[1-me][2].values())
                   for uid,(p,v,k) in pl[me][2].items()):
                turns.add((t,me));count+=1
                if count==3:break
    return turns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('model',type=Path);p.add_argument('sources',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--turns',default='35,75,110,145,190,225,260')
    p.add_argument('--stress',action='store_true',help='Evaluate recorded losses of contested collectors carrying at least 50')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('reply_check_candidate',a.candidate)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._torch.set_num_threads(1)
    synthetic(m)
    maps=json.loads(a.model.read_text())['maps']
    heldout=[g for g in maps if g['split']=='test']
    with a.output.open('x') as out:
        for g in heldout:
            path=next(s/'replays'/g['file'] for s in a.sources if (s/'replays'/g['file']).is_file())
            with gzip.open(path) as f:r=json.load(f)
            turns=historical_losses(m,r) if a.stress else set(map(int,a.turns.split(',')))
            for row in paired(m,r,turns):
                out.write(json.dumps(dict(file=g['file'],**row))+'\n')
            out.flush()


if __name__=='__main__':main()
