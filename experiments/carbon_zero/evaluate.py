"""Fixed-map paired evaluation of the temperature and trained actor."""
import hashlib
import time

import numpy as np
import torch

from env import Game, utility
from policy import Encoder, commands, logits, sample, slots, coordinated_sample


def seeds(panel,count):
    return [int.from_bytes(hashlib.sha256(f'carbon-zero/20261124/{panel}/{i}'.encode()).digest()[:4],'big')
            for i in range(count)]


@torch.inference_mode()
def games(teacher, learner, opponent, maps, temperature, opposition_temp, device, raw=None, coordinated=False, opponent_coordinated=None):
    started=time.perf_counter()
    rng=np.random.default_rng(41124)
    rows=[]
    if raw is None:
        specs=[(m,s) for m in maps for s in (0,1)]
        active=[(Game(seed),seat,seed,tuple(Encoder(teacher,i) for i in (0,1))) for seed,seat in specs]
    else:
        assert len(maps)==1
        active=[(Game(maps[0]),raw[1],maps[0],tuple(Encoder(teacher,i) for i in (0,1)))]
        raw[0]._ST={}
    if opponent_coordinated is None:
        opponent_coordinated=coordinated
    while active:
        own=[];other=[];contexts=[]
        for game,seat,seed,enc in active:
            obs=game.observations()
            for side,dest in ((seat,own),(1-seat,other)):
                names,cells,mask=slots(obs[side],game.config)
                dest.append((enc[side].encode(obs[side]),cells,mask,names))
            contexts.append(obs)
        def choose(model,group,temp,avoid):
            x,c,m,names=zip(*group)
            arrays=[torch.as_tensor(np.stack(v),device=device) for v in (x,c,m)]
            z,_=logits(model,*arrays,temp)
            picked=coordinated_sample(z,rng,np.stack(c),np.stack(m))[0] if avoid else sample(z,rng)
            return [commands(n,y) for n,y in zip(names,picked,strict=True)]
        actions=choose(learner,own,temperature,coordinated)
        replies=choose(opponent,other,opposition_temp,opponent_coordinated) if raw is None else [
            raw[0].agent(contexts[0][1-active[0][1]],active[0][0].config)]
        pending=[]
        for p,obs,a,b in zip(active,contexts,actions,replies,strict=True):
            g,seat,seed,enc=p
            enc[seat].commit(obs[seat],a);enc[1-seat].commit(obs[1-seat],b)
            eliminated,scores=g.step((a,b) if seat==0 else (b,a))
            if g.done:
                margin=scores[seat]-scores[1-seat]
                rows.append(dict(seed=seed,seat=seat,turns=g.turn,scores=scores,margin=round(margin,2),
                                 utility=utility(margin,eliminated),elimination=eliminated))
            else:
                pending.append(p)
        active=pending
    return dict(games=rows,seconds=time.perf_counter()-started)
