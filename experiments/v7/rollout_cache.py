"""Reuse a rollout's random tape and route geometry across alternatives at a given turn."""
from functools import lru_cache
import inspect
import numpy as np


@lru_cache(maxsize=256)
def _v7_random_tape(seed):
    return np.random.RandomState(seed).random_sample(32)


class _V7RandomTape:
    def __init__(self,seed):
        self.seed=seed
        self.values=_v7_random_tape(seed)
        self.index=0

    def random_sample(self,count):
        begin=self.index
        self.index+=count
        if self.index>len(self.values):
            # Standard maps need at most two bases and ten workers per side.
            self.values=np.random.RandomState(self.seed).random_sample(self.index)
        return self.values[begin:self.index]


@lru_cache(maxsize=3072)
def _v7_approaches(homes,start,cutoff):
    home_dist=_v7_nearest(homes)
    frontier={start:1.}
    result=[]
    for t in range(1,cutoff+1):
        nxt={}
        for q,prob in frontier.items():
            qs=[n for n in _V7_L[q,1:] if home_dist[n]==home_dist[q]-1]
            for n in qs:nxt[n]=nxt.get(n,0.)+prob/len(qs)
        frontier=nxt
        result.extend((t,q,prob) for q,prob in frontier.items() if home_dist[q]>=1)
    return result


def build(code):
    anchor='def _v7_plan('
    if code.count(anchor)!=1:raise ValueError(anchor)
    helpers='from functools import lru_cache\n'+'\n\n'.join(inspect.getsource(x) for x in (_v7_random_tape,_V7RandomTape,_v7_approaches))
    code=code.replace(anchor,helpers+'\n\n'+anchor)
    start=code.index('                frontier={start:1.}',code.index(anchor))
    stop=code.index('        else:',start)
    old=code[start:stop]
    body=old[old.index('                        travel='):]
    body='\n'.join(line[4:] for line in body.split('\n'))
    body=body.replace('if eh[q]<1 or travel>t or q in enemy_homes:continue','if travel>t:continue')
    needle='                    action=_v7_safe_route(pos,q,original[uid],blocked,danger,tax*.4)'
    if body.count(needle)!=1:raise ValueError(needle)
    body=body.replace(needle,'''                    if q not in route_choice:
                        route_choice[q]=_v7_safe_route(pos,q,original[uid],blocked,danger,tax*.4)
                    action=route_choice[q]''')
    code=code[:start]+'                for t,q,prob in _v7_approaches(tuple(enemy_homes),start,cutoff):\n'+body+code[stop:]
    anchor="        elif mode=='intercept':\n            for start,prize in rival:"
    if code.count(anchor)!=1:raise ValueError(anchor)
    code=code.replace(anchor,"        elif mode=='intercept':\n            route_choice={}\n            for start,prize in rival:")
    for term in ('seed+373*t+17*s','seed0 + 7919 * t + 31 * m + 1','seed0 + 7919 * t + 31 * m + 2'):
        old='np.random.RandomState('+term+')'
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,'_V7RandomTape('+term+')')
    return code
