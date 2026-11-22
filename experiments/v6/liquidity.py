"""Score economic survival and the official early-elimination rule in simulated positions."""
def _v6_eliminated(player, hire_floor):
    cash, bases, workers, trees = player
    return not workers and not trees and (not bases or cash < hire_floor)


def _v6_reserve(player, trees, turns, hire_floor):
    cash, bases, workers, owned = player
    if cash >= 2*hire_floor or not bases or turns < 5:
        return 0.
    home = _v5_home(tuple(bases.values()))
    deposits = []
    for pos, load, role in workers.values():
        if role == 'COLLECTOR' and load > 0. and home[pos] <= min(turns, 7):
            deposits.append(load / (1. + .55 * home[pos]))
    income = sum(min(7., max(0., 50-age))*min(10., trees.get(uid,(None,0.))[1])
                 for uid, (_,age) in owned.items())
    cover = max(0., 1. - (cash + .68*income + .66*max(deposits,default=0.)) / (1.8*hire_floor))
    if cover <= 0.:
        return 0.
    collectors = sum(role=='COLLECTOR' for _,_,role in workers.values())
    # No tree, no readily bankable cargo, and too little to fund a replacement.
    exposure = (72./(1.+.5*collectors) + 64./(1.+len(owned)) +
                145.*(collectors<=1 and not owned) + 210.*(not collectors and not owned))
    return cover*exposure*min(1.,turns/35.)


def _v6_liquidity(board, side, floor):
    state = board.get('_st')
    trees = state.trees_meta if state is not None else board.get('trees', {})
    players = board['players']
    turns = max(0, LAST_TURN + 1 - board['step'])
    return (_v6_reserve(players[1-side], trees, turns, floor) -
            _v6_reserve(players[side], trees, turns, floor)) / 3000.
