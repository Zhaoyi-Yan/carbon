"""Replay-warmable, reactive approximations of the two CarbonEdge policies."""
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STYLE = {
    'artemis': ('226891089ba7f54b198c8f8dc771dfc2', 22),
    'apollo': ('dd1f117a64c676225f0c6333b1cdd1e7', 21),
}


@lru_cache(maxsize=1)
def _engine():
    # Grove embeds the v6r shared convolutional stack with both dedicated teacher rows.
    path=ROOT/'experiments/v6r/checkpoints/grove.py'
    spec=importlib.util.spec_from_file_location('v7_surrogate_policy',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._POLICY.p['emb'].shape[0]==23
    return module


def _joint(players, side, worker_logits, base_logits, m, config):
    from scipy.optimize import linear_sum_assignment
    own=players[side]
    workers=list(own[2].items());bases=list(own[1].items())
    costs={role:config.get('rec'+role+'Cost',30)+config.get('rec'+role+'IncreaseCost',0)*
            sum(w[2]==role.upper() for p in players for w in p[2].values()) for role in ('Collector','Planter')}
    free=config.get('workerLimit',10)-len(workers)
    tasks=[]; choices=[]
    for uid,(pos,_,_) in workers:
        land=[pos]+m.NB[pos]
        tasks.append([(q,float(v),m.WACT[i] if i else None) for i,(q,v) in enumerate(zip(land,worker_logits[pos]))])
        choices.append(uid)
    for number,(uid,pos) in enumerate(bases):
        logits=base_logits[pos]
        options=[(225+number,float(logits[0]),None)]
        if free>0:
            roles=[(float(logits[j]),m.BACT[j]) for j,k in enumerate(('','Collector','Planter')) if j and own[0]>=costs[k]]
            if roles:
                value,act=max(roles)
                options.append((pos,value,act))
        tasks.append(options);choices.append(uid)
    if not tasks:return {}
    columns={q:i for i,q in enumerate(sorted({q for opts in tasks for q,_,_ in opts}))}
    values=np.full((len(tasks),len(columns)),-1.e6)
    for i,options in enumerate(tasks):
        for q,score,_ in options:values[i,columns[q]]=score
    assigned=linear_sum_assignment(values,maximize=True)[1]
    actions={};hires=[]
    for uid,options,col in zip(choices,tasks,assigned):
        q,score,action=next(v for v in options if columns[v[0]]==col)
        if action:
            if action.startswith('REC'):
                hires.append((score-options[0][1],uid,action))
            else:actions[uid]=action
    money=own[0]
    for _,uid,act in sorted(hires,reverse=True):
        cost=costs['Planter' if act=='RECPLANTER' else 'Collector']
        if free>0 and money>=cost:
            free-=1;money-=cost;actions[uid]=act
    return actions


class Surrogate:
    """observe consumes the recorded turn; act consumes a turn using predicted actions."""

    def __init__(self,name,seed,config):
        if name not in STYLE:raise ValueError(f'Unknown surrogate: {name}')
        self.name=name
        self.style=STYLE[name][1]
        self.seed=seed
        self.config=dict(config)
        self.tracker=None
        self.previous=None
        self.last_step=None
        self.ever_hired=False

    def _step(self,obs):
        side=obs['player'];t=obs['step']
        if self.tracker is None or t==0:
            m=_engine();self.tracker=m.BeliefTracker(m._Bel,side)
            self.previous=None;self.last_step=None;self.ever_hired=False
        if self.tracker.me!=side or (self.last_step is not None and t!=self.last_step+1):
            raise ValueError(f'Expected next observation after {self.last_step}, got {t} at seat {side}')
        if obs['players'][side][2]:self.ever_hired=True
        return self.tracker.step(obs['players'],t,obs['carbon'])

    def _finish(self,obs,action):
        self.tracker.set_actions(obs['players'],action)
        self.previous=obs['players'];self.last_step=obs['step']

    def observe(self,obs,action):
        self._step(obs);self._finish(obs,action or {})

    def logits(self,obs,estimate,known):
        m=_engine();x=m.encode(obs['players'],obs['step'],obs['player'],estimate,known,self.previous)
        x=np.rint(x*100).clip(0,255).astype(np.float32)*.01
        return m._POLICY.forward(x,self.style)

    def _opening(self,obs):
        side=obs['player'];t=obs['step'];mine,foe=obs['players'][side],obs['players'][1-side]
        homes=sorted(mine[1],key=mine[1].get)
        if self.ever_hired or len(homes)!=2 or mine[0]<60 or t>50:return None
        planter=0
        if self.name=='artemis':
            if t==0 or (not foe[2] and t<46):return {}
            opposing_planters=[w[0] for w in foe[2].values() if w[2]=='PLANTER']
            if not opposing_planters:planter=int(bool(foe[2]))
            elif side==0:
                planter=min(range(2),key=lambda k:abs(mine[1][homes[k]]//15-opposing_planters[0]//15))
        return {homes[planter]:'RECPLANTER',homes[1-planter]:'RECCOLLECTOR'}

    def predict(self,obs):
        estimate,known=self._step(obs)
        opening=self._opening(obs)
        if opening is not None:return opening
        logits,bases=self.logits(obs,estimate,known)
        return _joint(obs['players'],obs['player'],logits,bases,_engine(),self.config)

    def act(self,obs):
        action=self.predict(obs)
        self._finish(obs,action)
        return action


def make(name,seed,config):
    return Surrogate(name,seed,config)
