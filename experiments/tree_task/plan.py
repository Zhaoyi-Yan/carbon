"""Keep a planter near a new tree until its planting cost is recovered or defense is unsafe."""
from dataclasses import dataclass
from starvation_replay import LAND, distance


def planting_cost(players, side, cfg):
    cost = cfg['plantCost']
    if len(players[side][3]) <= cfg['playerTreeProtectiveNumber']:
        return cost
    live = sum(age < cfg['treeLifespan'] for p in players for _, age in p[3].values())
    return cost + cfg['plantCostInflationRatio'] * cfg['plantCostInflationBase']**live


def arrivals(player, action, except_worker):
    destinations = {LAND[p][action.get(uid,'STAY')] for uid,(p,_,_) in player[2].items()
                    if uid != except_worker}
    return destinations | {p for uid,p in player[1].items() if action.get(uid)}


@dataclass
class TreeTask:
    worker: str
    cell: int
    start: int
    cost: float
    seize_cost: float = 20.
    cap: int = 18
    tree: str | None = None
    paid_back: float = 0.
    stop: str | None = None

    def follow(self, obs, action):
        side = obs['player']
        own, enemy = obs['players'][side], obs['players'][1-side]
        if self.stop is not None:
            return action
        if obs['step'] >= self.start + self.cap:
            self.stop = 'time_cap'
        elif self.worker not in own[2]:
            self.stop = 'planter_lost'
        elif self.tree not in own[3]:
            self.stop = 'tree_lost'
        if self.stop is not None:
            return action

        pos = own[2][self.worker][0]
        foes = [(q,kind) for q,_,kind in enemy[2].values()]
        danger = {q for p,k in foes if k=='COLLECTOR' for q in LAND[p].values()}
        occupied = arrivals(own,action,self.worker)
        bases = set(own[1].values()) | set(enemy[1].values())
        threat = enemy[0] >= self.seize_cost and any(k=='PLANTER' and distance(q,self.cell)<=2 for q,k in foes)
        options = {}
        for move, tile in LAND[pos].items():
            if tile in danger or tile in occupied or tile in bases or distance(tile,self.cell)>1:
                continue
            # Standing on a bare tile could spend cash on a second tree.
            if move=='STAY' and tile!=self.cell:
                continue
            options[move] = tile
        if not options:
            self.stop = 'no_safe_defense'
            return action

        proposed = action.get(self.worker,'STAY')
        chosen = min(options, key=lambda m: (threat and options[m]!=self.cell,
                                              m!=proposed, m=='STAY', m))
        if chosen==proposed:
            return action
        output = dict(action)
        if chosen=='STAY':
            output.pop(self.worker,None)
        else:
            output[self.worker] = chosen
        return output

    def settle(self, obs, side):
        owned = obs['players'][side][3]
        if self.tree is None and self.stop is None:
            self.tree = next((uid for uid,(p,_) in owned.items() if p==self.cell),None)
            if self.tree is None:
                self.stop = 'plant_failed'
        if self.tree in owned and self.stop is None:
            self.paid_back += obs['trees'][self.tree][1]
            if self.paid_back >= self.cost:
                self.stop = 'repaid'
