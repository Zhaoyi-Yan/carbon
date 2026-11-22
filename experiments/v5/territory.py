"""Estimate economic control of visible or inferred carbon, including disputed cells."""
import numpy as np
from features import _V5_DIST, _v5_observe, _v5_context

_V5_GRID = np.arange(225)
_V5_GEOMETRY = np.asarray(_V5_DIST)
_V5_ADJ = np.array([[(p//15+dr)%15*15+(p%15+dc)%15 for dr,dc in ((-1,0),(0,1),(1,0),(0,-1))] for p in _V5_GRID])


def _v5_territory(players, carbon, step, last):
    stock = np.asarray(carbon, dtype=float)
    remaining = max(0, last+1-step)
    tree_owner = np.full(225, -1, dtype=int)
    for owner, pl in enumerate(players):
        for pos, age in pl[3].values():
            tree_owner[pos] = owner
    tree_adj = tree_owner[_V5_ADJ]
    multiplicity = np.maximum(1, (tree_adj >= 0).sum(1))
    economies = []
    for side, pl in enumerate(players):
        unit_list = list(pl[2].values())
        gather = [(pos, cargo) for pos, cargo, role in unit_list if role=='COLLECTOR']
        planters = [pos for pos, cargo, role in unit_list if role=='PLANTER']
        def nearest(positions):
            return _V5_GEOMETRY[positions].min(0) if positions else np.full(225, 16)
        light = nearest([p for p,v in gather if v < 30])
        distance = nearest([p for p,v in gather])
        planting = nearest(planters)
        homes = nearest(list(pl[1].values()))
        economies.append((light, distance, planting, homes))
    features = []
    for side, pl in enumerate(players):
        light, distance, planting, homes = economies[side]
        enemy_light, enemy_distance, enemy_planting, enemy_homes = economies[1-side]
        deposits = [w for w in pl[2].values() if w[2]=='COLLECTOR']
        free = tree_owner < 0
        control = np.clip(.5+.20*(enemy_distance-distance), 0., 1.)
        collectable = stock * free * control / (1.+distance*.45+homes*.12)
        nearby_rich = float(collectable[distance<=2].sum()/100.)
        nearest_rich = float(collectable.sum()/100.)
        rich_plant = stock * free * np.clip(.5+.23*(enemy_planting-planting),0.,1.) / (1.+planting*.7)
        tree_life = np.zeros(225)
        for p,age in pl[3].values():
            tree_life[p] = min(remaining, max(0, 50-age))/50.
        # Project returns from each of the four adjacent deposits; overlapping trees share stock.
        tree_share = (tree_life[_V5_ADJ]/multiplicity[:,None]).sum(1)
        yield_potential = np.sum(stock * tree_share)/100.
        exposed = np.sum(stock * (tree_life*np.exp(-enemy_planting*.65))[_V5_ADJ].sum(1)/multiplicity)/100.
        support = np.sum(stock * (tree_life*np.exp(-light*.7))[_V5_ADJ].sum(1)/multiplicity)/100.
        cargo = pressure = harvest = 0.
        for p, load, _ in deposits:
            margin = min(enemy_light[p], 5)-1.
            cargo += load * np.clip(.65+.12*margin, .1, 1.) * (homes[p]<=remaining)/(1.+.08*homes[p])
            pressure += max(0., 3.-enemy_distance[p]) * (load<15) * (enemy_homes[p]<homes[p])
            harvest += min(stock[p],100.) * (tree_owner[p]<0) * (enemy_light[p]>1)
        features.append(np.array([nearby_rich, nearest_rich, rich_plant.sum()/100., yield_potential,
                                  exposed, support, cargo/100., pressure, harvest/100.]))
    delta = features[0]-features[1]
    phase = step/max(1,last)
    return np.r_[delta, delta*phase, delta*max(0.,1.-remaining/48.)]


def _v5_territory_value(board, side):
    st = board.get('_st')
    detail = st.trees_meta if st is not None else board.get('trees', {})
    pl, step = board['players'], board['step']
    econ, styles = _v5_observe(pl,detail,step,LAST_TURN)
    val, _ = _v5_context(econ,styles,pl,step,LAST_TURN,side)
    landscape = _v5_territory(pl,board['carbon'],step,LAST_TURN) * (1-2*side)
    cash = pl[side][0]-pl[1-side][0]
    return (cash/1000.+np.r_[val,landscape] @ _V5['territory'])/3.
