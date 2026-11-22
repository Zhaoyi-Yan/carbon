"""Cheap public-board features shared by replay fitting and the live planner."""
from functools import lru_cache
import numpy as np


_V5_XY = np.indices((15, 15)).reshape(2, -1)
_V5_DELTA = np.abs(_V5_XY[:, :, None] - _V5_XY[:, None, :])
_V5_DIST = np.minimum(_V5_DELTA, 15 - _V5_DELTA).sum(axis=0).tolist()


@lru_cache(maxsize=64)
def _v5_home(bases):
    return [min((_V5_DIST[pos][b] for b in bases), default=15) for pos in range(225)]


def _v5_observe(players, trees, step, last):
    remaining = max(0, last + 1 - step)
    homes = [_v5_home(tuple(p[1].values())) for p in players]
    collectors = [[(pos, cargo) for pos, cargo, kind in p[2].values() if kind == 'COLLECTOR'] for p in players]
    planters = [[pos for pos, _, kind in p[2].values() if kind == 'PLANTER'] for p in players]
    economic, styles = [], []
    for side, player in enumerate(players):
        other = 1 - side
        cs, ps = collectors[side], planters[side]
        count = max(1, len(cs))
        pressure = cargo_bank = cargo_risk = advance = light = 0.
        for pos, load in cs:
            near = homes[side][pos]
            threatened = any(v < load and _V5_DIST[pos][q] <= 2 for q, v in collectors[other])
            cargo_bank += load * max(0., 1. - .045 * near) * (near <= remaining)
            cargo_risk += load * threatened
            light += load < 30
            advance += homes[other][pos] < near
            pressure += load < 30 and any(_V5_DIST[pos][q] <= 2 for q, v in collectors[other] if v >= 30)
        life = income = seize_risk = planting_front = 0.
        for key, (pos, age) in player[3].items():
            turns = min(remaining, max(0, 50 - age))
            rate = float(trees.get(key, (0, 0))[1])
            life += turns / 50.
            income += rate * min(24, turns) / 24.
            if any(_V5_DIST[pos][q] <= 2 for q in planters[other]):
                seize_risk += min(24, turns) * (.1 + rate) / 24.
        for pos in ps:
            planting_front += homes[other][pos] <= homes[side][pos]
        tree_count = len(player[3])
        cargo = sum(v for _, v in cs)
        economic.append(np.array([len(cs), len(ps), cargo / 100., cargo_bank / 100.,
                                  cargo_risk / 100., tree_count, life, income / 10., seize_risk / 10.,
                                  advance, pressure, planting_front, min(player[0], 150.) / 100.]))
        styles.append(np.array([len(ps) / max(1, len(cs) + len(ps)), advance / count,
                                light / count, pressure / count, planting_front / max(1, len(ps)),
                                min(3., cargo / (100. * count)), min(2., tree_count / 10.),
                                min(2., income / 50.)]))
    return economic, styles


def _v5_context(economic, styles, players, step, last, me):
    remaining = max(0, last + 1 - step)
    delta = economic[me] - economic[1-me]
    phase = step / max(1, last)
    late = max(0., 1. - remaining / 50.)
    cash = np.clip((players[me][0] - players[1-me][0]) / 1000., -4., 4.)
    value = np.concatenate((delta, delta * phase, delta * late, [cash, cash * phase]))
    route = np.concatenate(([1., phase, late, cash], styles[1-me], delta[[0, 1, 3, 5, 7]] / 5.))
    return value, route
