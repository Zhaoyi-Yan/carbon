"""Audit successful planting, capture, and moves away from an affordable opportunity."""
import numpy as np

PLANTED = 8
SEIZED = 16
LEFT_PLANT = 32
LEFT_SEIZE = 64


def tree_flags(players, after, side, actions, cfg):
    mask = np.zeros(225, dtype=np.uint8)
    own = players[side]
    # Use identity across frames to distinguish capture from creating a new tree.
    for uid, (cell, _) in after[side][3].items():
        if uid not in own[3]:
            mask[cell] |= SEIZED if uid in players[1-side][3] else PLANTED
    live = {cell:i for i,p in enumerate(players) for cell,age in p[3].values()
            if age < cfg['treeLifespan']}
    quote = cfg['plantCost']
    if len(own[3]) > cfg['playerTreeProtectiveNumber']:
        quote += cfg['plantCostInflationRatio'] * cfg['plantCostInflationBase']**len(live)
    bases = {cell for p in players for cell in p[1].values()}
    for uid, (cell, _, kind) in own[2].items():
        if kind != 'PLANTER' or not actions.get(uid) or cell in bases:
            continue
        if cell not in live and own[0] >= quote:
            mask[cell] |= LEFT_PLANT
        elif live.get(cell) == 1-side and own[0] >= cfg['seizeCost']:
            mask[cell] |= LEFT_SEIZE
    return mask
