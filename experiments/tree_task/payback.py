"""Estimate the chance a newly planted tree earns its cost during first ownership."""
from dataclasses import dataclass
import math

import numpy as np

from starvation_replay import LAND, distance

FEATURES = ('stock_return', 'light_collector_pressure', 'planter_pressure', 'guard_edge',
            'active_tax', 'price', 'remaining_life', 'planter_capacity', 'forest_competition')


def features(players, stock, side, cell, turn, cost, cfg):
    """Only the acting team's belief about soil and public state enter the model."""
    own, enemy = players[side], players[1-side]
    occupied = {p for pl in players for p in pl[1].values()}
    trees = {p for pl in players for p,age in pl[3].values() if age < cfg['treeLifespan']}
    blocked = occupied | trees
    rival_collectors = [(p,c) for p,c,k in enemy[2].values() if k=='COLLECTOR']
    rivals = min((distance(cell,q) for q,_,k in enemy[2].values() if k=='PLANTER'), default=10)
    guard = min((distance(cell,q) for q,_,k in own[2].values() if k=='COLLECTOR'), default=10)
    nearest = min((distance(cell,q) for q,_ in rival_collectors), default=10)
    horizon = min(cfg['treeLifespan'],max(0,cfg['episodeSteps']-2-turn))
    own_planters = sum(k=='PLANTER' for _,_,k in own[2].values())
    projected = 0.
    competition = 0.
    rate = cfg['cellAbsorptionRate']
    for move,tile in LAND[cell].items():
        if move=='STAY' or tile in blocked:
            continue
        shared = sum(p in trees for p in LAND[tile].values())
        retention = max(0.,1-rate*(1+shared))
        projected += float(stock[tile])*rate*(1-retention**horizon)/(1-retention)
        competition += shared
    # Existing enemy cargo is transient. Treat its current value as a feature.
    tax = sum(min(c,150.) for q,c in rival_collectors if distance(q,cell)<=2)*cfg['collectorAbsorptionRate']
    return np.asarray((math.log((projected+1)/(cost+1)), max(0,4-nearest),
                       max(0,6-rivals), np.clip(rivals-guard,-5,5),
                       math.log1p(tax),math.log1p(cost),horizon/50.,own_planters,
                       competition), dtype=np.float64)


@dataclass
class Payback:
    center: np.ndarray
    scale: np.ndarray
    odds: np.ndarray
    log_return: np.ndarray

    def forecast(self, x, cost):
        slope = (x-self.center)/self.scale
        z = np.r_[1.,np.clip(slope,-4,4)]
        p = 1./(1.+float(np.exp(-np.clip(z @ self.odds,-20,20))))
        # Median first-spell gross on the transformed scale.
        gross = max(0.,float(np.exp(np.clip(z @ self.log_return,-4,4)))*(cost+1)-1.)
        return p,gross

    def json(self):
        return {k:v.tolist() for k,v in vars(self).items()}


def fit(examples):
    x=np.stack([e['features'] for e in examples])
    cost=np.array([e['cost'] for e in examples])
    observed=np.array([e['income'] for e in examples])
    log_return=np.log((observed+1.)/(cost+1.))
    full=observed>=cost
    center=x.mean(0)
    scale=np.maximum(x.std(0),.10)
    z=np.column_stack((np.ones(len(x)),np.clip((x-center)/scale,-4,4)))
    unique,counts=np.unique([e['match'] for e in examples],return_counts=True)
    per_map = dict(zip(unique,counts))
    weights=np.array([len(x)/(len(unique)*per_map[e['match']]) for e in examples])
    penalty=np.diag([1e-5]+[12.]*x.shape[1])
    coef=np.linalg.solve(z.T@(weights[:,None]*z)+penalty,z.T@(weights*log_return))
    beta=np.zeros(z.shape[1])
    for _ in range(15):
        p=1/(1+np.exp(-np.clip(z@beta,-20,20)))
        hessian=z.T@((weights*p*(1-p))[:,None]*z)+penalty
        beta+=np.linalg.solve(hessian,z.T@(weights*(full-p))-penalty@beta)
    return Payback(center,scale,beta,coef)
