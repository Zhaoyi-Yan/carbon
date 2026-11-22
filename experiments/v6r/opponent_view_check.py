"""Compare reconstructed rival information with that seat's actual information on replay."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from localhead_data import load


def play(model, paths):
    out=[]
    for path in paths:
        replay=json.load(gzip.open(path))
        for side in (0,1):
            foe=1-side;history={};track=[model.BeliefTracker(model._Bel,s) for s in (0,1)]
            sums=np.zeros((2,5),dtype='f8');prev=None
            for t in range(min(150,len(replay['steps'])-1)):
                frame=replay['steps'][t];pl=frame[0]['observation']['players']
                # Only the evaluator sees the rival's own input and future actions.
                own,_=track[side].step(pl,t,frame[side]['observation']['carbon'])
                view=model._r6_opponent_view(history,pl,t,foe,own,prev)
                flat=model._quant(model.encode(pl,t,foe,own,[1.]*225,prev))
                true,mask=track[foe].step(pl,t,frame[foe]['observation']['carbon'])
                target=model._quant(model.encode(pl,t,foe,true,mask,prev))
                for k,x in enumerate((flat,view)):
                    sums[k,0]+=np.abs(x[0]-target[0]).sum()
                    sums[k,1]+=np.abs(x[1]-target[1]).sum()
                    sums[k,2]+=225
                act=replay['steps'][t+1][foe]['action'] or {}
                if t%8==3 and pl[foe][2]:
                    wb,_,_=model._R6_OLD_NET(np.stack([flat,view,target]),0)
                    for uid,(pos,cargo,kind) in pl[foe][2].items():
                        choice=model.WIDX.get(act.get(uid),0)
                        for k in (0,1):
                            prob=model._probs(wb[k,pos],1.)
                            sums[k,3]-=np.log(max(prob[choice],1.e-9))
                            sums[k,4]+=1
                for s,tr in enumerate(track):
                    tr.set_actions(pl,replay['steps'][t+1][s]['action'] or {})
                prev=pl
            out.append(dict(file=path.name,seat=side,totals=sums.tolist()))
    v=np.sum([s['totals'] for s in out],axis=0)
    stats=[dict(resource_plane_mae=row[0]/row[2],mask_error_rate=row[1]/row[2],
                worker_action_nll=row[3]/row[4],actions=int(row[4])) for row in v]
    return dict(source='Only input of the simulated observing seat is available to the estimator. Actual rival input, actions and weights are used only for scoring. Both predictors use the same original v5 policy and style zero.',by_model=dict(omniscient_mask=stats[0],reconstructed=stats[1]),by_seat=out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();data=play(load(a.source),sorted((a.data/'replays').glob('*.gz'))[:10])
    with a.output.open('x') as f:json.dump(data,f,indent=2)
    print(data['by_model'])
