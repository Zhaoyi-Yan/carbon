"""Carbon League strategy. Uses only public per-turn observations and the standard library."""

from functools import lru_cache
from math import inf

MOVES = ("UP", "RIGHT", "DOWN", "LEFT")
OPENING = "mixed"
MAX_PLANTERS = 2
RETURN_RATE = 1.3
STATE = {}


@lru_cache(maxsize=4)
def geometry(size):
    def index(row, col):
        return (row % size) * size + col % size

    points = [divmod(i, size) for i in range(size * size)]
    neighbors = []
    vicinity = []
    mirrors = []
    distances = []
    for row, col in points:
        neighbors.append(tuple(index(row + dr, col + dc)
                               for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1))))
        vicinity.append(tuple(index(row + dr, col + dc)
                              for dr in (-1, 0, 1) for dc in (-1, 0, 1)))
        mirrors.append(tuple({index(r, c) for r in (row, size - row - 1)
                              for c in (col, size - col - 1)}))
        distances.append(tuple(
            min(abs(row - r), size - abs(row - r)) + min(abs(col - c), size - abs(col - c))
            for r, c in points
        ))
    return neighbors, vicinity, mirrors, distances


def update_map(obs, config, workers, enemies, bases, trees, state, neighbors, mirrors):
    step = obs["step"]
    count = len(neighbors)
    maximum = config.get("maxCellCarbon", 100)
    growth = config.get("regenRate", 0.03)
    absorb = config.get("cellAbsorptionRate", 0.0375)
    harvest = config.get("initialCollectRate", 0.3)
    initial = config.get("startingCarbon", 2500) / count
    carbon = state.get("carbon", [initial] * count)
    disturbed = state.get("disturbed", set())
    observed = state.get("observed", [-1] * count)
    positions = {key: pos for key, (pos, _, _) in (workers | enemies).items()}
    previous = state.get("positions", {})
    coverage = [0] * count
    for pos in trees:
        for adj in neighbors[pos]:
            coverage[adj] += 1

    if step:
        stationary = {pos for key, pos in positions.items() if previous.get(key) == pos}
        collecting = {pos for key, (pos, _, role) in (workers | enemies).items()
                      if role == "COLLECTOR" and previous.get(key) == pos}
        for pos in range(count):
            if pos in trees or pos in bases:
                carbon[pos] = 0.0
                disturbed.add(pos)
            elif pos in state.get("trees", ()):
                carbon[pos] = config.get("co2FrmWithered", 30)
                disturbed.add(pos)
                observed[pos] = step
            elif pos in collecting:
                carbon[pos] *= 1 - harvest
                disturbed.add(pos)
            elif coverage[pos]:
                carbon[pos] *= max(0.0, 1 - absorb * coverage[pos])
                disturbed.add(pos)
            elif pos in stationary:
                disturbed.add(pos)
            else:
                # Estimate pristine unknown cells by averaging empty and occupied cells.
                # The positive half saturates before the unconditional mean reaches 100.
                unknown_cap = maximum * config.get("carbonCoverage", 45) / 100
                cap = maximum if observed[pos] >= 0 else unknown_cap
                carbon[pos] = min(cap, carbon[pos] * (1 + growth))
    else:
        for pos in bases:
            carbon[pos] = 0.0
            disturbed.add(pos)

    visible = set()
    for pos, _, role in workers.values():
        if role == "COLLECTOR":
            visible.add(pos)
        else:
            visible.update(neighbors[pos])
    for pos in visible:
        value = obs["carbon"][pos]
        carbon[pos] = value
        observed[pos] = step
        if pos not in disturbed:
            for mirror in mirrors[pos]:
                if mirror not in disturbed and mirror not in visible:
                    carbon[mirror] = value
                    observed[mirror] = step

    state.update(carbon=carbon, disturbed=disturbed, observed=observed,
                 positions=positions, trees=set(trees))
    return carbon, coverage


def assign_moves(rows):
    """Find the maximum-weight set of distinct destinations (rectangular Hungarian)."""
    if not rows:
        return []
    cells = sorted({pos for row in rows for pos in row})
    costs = [[-row[pos][0] if pos in row else 1_000_000.0 for pos in cells] for row in rows]
    n, m = len(rows), len(cells)
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    matching, links = [0] * (m + 1), [0] * (m + 1)
    for entrant in range(1, n + 1):
        matching[0] = entrant
        col = 0
        slack, used = [inf] * (m + 1), [False] * (m + 1)
        while True:
            used[col] = True
            row = matching[col]
            delta, next_col = inf, 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                reduced = costs[row - 1][candidate - 1] - u[row] - v[candidate]
                if reduced < slack[candidate]:
                    slack[candidate], links[candidate] = reduced, col
                if slack[candidate] < delta:
                    delta, next_col = slack[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[matching[candidate]] += delta
                    v[candidate] -= delta
                else:
                    slack[candidate] -= delta
            col = next_col
            if matching[col] == 0:
                break
        while col:
            previous = links[col]
            matching[col] = matching[previous]
            col = previous

    result = [None] * n
    for col, row in enumerate(matching[1:]):
        if row:
            result[row - 1] = cells[col]
    return result


def enemy_moves(enemies, enemy_bases, own_bases, trees, carbon, neighbors, distances):
    """Estimate one-turn collision risks, including loaded collectors heading home."""
    threat = [[] for _ in neighbors]
    for pos, load, role in enemies.values():
        options = (pos, *neighbors[pos])
        home = min(distances[base][pos] for base in enemy_bases)
        weights = []
        for dest in options:
            distance = min(distances[base][dest] for base in enemy_bases)
            if role == "COLLECTOR":
                weight = 1.0
                if load >= 25 and distance < home:
                    weight += min(6.0, load / 11)
                if dest == pos and pos not in trees and pos not in enemy_bases:
                    weight += carbon[pos] / 22
                if dest in own_bases or (dest in enemy_bases and load == 0):
                    weight *= 0.18
            else:
                weight = 1.0
                if dest == pos and pos not in enemy_bases:
                    weight += 1.5 if pos not in trees else 0.4
                if dest in trees and dest != pos:
                    weight += 0.25
            weights.append(weight)
        total = sum(weights)
        for dest, weight in zip(options, weights):
            threat[dest].append((weight / total, load, role))
    return threat


def navigate(start, options, target, score, rows, distances, penalty=1.0):
    distance = distances[start][target]
    for dest in options:
        if dest == start and target != start:
            continue
        extra = (dest != start) + distances[dest][target] - distance
        if extra > 0 and target == start:
            continue
        value = score - penalty * extra
        if value > rows[dest][0]:
            rows[dest] = (value, target)


def collector_moves(start, load, options, distances, home, remaining, carbon, trespass,
                    trees, bases, drain, coverage, config, old_target, claimed, threat, neighbors):
    rows = {dest: (-30.0, dest) for dest in options}
    rate = config.get("initialCollectRate", 0.3)
    tax = config.get("smeltCost", 0)
    short_horizon = remaining <= home[start] + 9
    for target in bases:
        walk = distances[start][target]
        if walk > remaining:
            continue
        value = (max(0, load - tax) / max(1, walk)) if short_horizon else (
            RETURN_RATE + 0.14 * load / max(1, walk) - (1.6 if load < 14 else 0.0)
        )
        if not load:
            value = -2.0
        navigate(start, options, target, value, rows, distances, 2.5)

    # Plan a return via one or two deposits. A neighbor on the way home can
    # add a harvest without requiring any additional movement.
    for target, stock in enumerate(carbon):
        walk = distances[start][target]
        if (stock < 3 or walk > 8 or trespass[target] + 2 < home[target]
                or target in trees or home[target] == 0 or trespass[target] == 0
                or walk + home[target] + 1 > remaining):
            continue
        stock *= max(0.45, 1 - 0.045 * coverage[target] * walk)
        stock *= 0.80 if target in claimed else 1.0
        territory = max(0, home[target] - trespass[target])
        survival = max(0.35, 1 - 0.10 * drain[target] - 0.07 * territory)
        best = -inf
        for turns in (1, 2, 3, 4):
            time = walk + turns + home[target]
            if time > remaining:
                break
            gain = stock * (1 - (1 - rate) ** turns)
            if short_horizon:
                value = max(0, (load + gain) * survival - tax) / time
            else:
                value = gain * survival / (time + min(2, home[target]) * 0.65)
                value += 0.14 * load / max(1, home[target])
                value -= 0.035 * load * (walk + turns) / max(1, home[start])
            best = max(best, value)
        if not short_horizon and stock >= 12 and walk <= 5:
            for second in neighbors[target]:
                if (second in trees or second in claimed or not home[second]
                        or not trespass[second] or carbon[second] < 15
                        or trespass[second] + 2 < home[second]):
                    continue
                # A discounted follow-up accounts for other workers harvesting first.
                second_stock = carbon[second] * max(0.45, 0.83 - 0.12 * drain[second])
                for first_turns, second_turns in ((1, 2), (2, 2), (2, 3)):
                    time = walk + first_turns + 1 + second_turns + home[second]
                    if time > remaining:
                        continue
                    gain = stock * (1 - (1 - rate) ** first_turns)
                    gain += second_stock * (1 - (1 - rate) ** second_turns)
                    value = gain * survival / (time + min(2, home[second]) * 0.65)
                    value += 0.14 * load / max(1, home[second])
                    value -= 0.035 * load * (walk + first_turns + second_turns) / max(1, home[start])
                    best = max(best, value - 0.35)
        best += 0.24 * (target == old_target) - 0.55 * (target in claimed)
        navigate(start, options, target, best, rows, distances, 1.65)

    for dest, (score, target) in rows.items():
        risk = 0.0
        for probability, enemy_load, role in threat[dest]:
            if role == "PLANTER":
                risk -= probability * (4.0 if remaining >= 12 else 0.0)
            elif enemy_load + 0.01 < load:
                risk += probability * (23.0 + 0.28 * load)
            elif enemy_load <= load + 0.01:
                risk += probability * (12.0 + 0.18 * load)
            else:
                risk -= probability * (4.5 + 0.17 * enemy_load)
        if dest in bases and load:
            risk -= min(6.0, load * (0.23 if short_horizon else 0.075))
        else:
            risk += drain[dest] * load * 0.031
        if trespass[dest] == 0:
            risk += 200
        rows[dest] = (score - risk, target)

    if not load and remaining <= 4:
        return {dest: (score + (0.8 if dest == start else 0.0), target)
                for dest, (score, target) in rows.items()}
    return rows


def planter_moves(start, options, distances, home, trespass, remaining, trees, carbon,
                  tree_value, price, seize, cash, threat, claimed, old_target):
    rows = {dest: (-10.0, dest) for dest in options}
    for target, yield_value in enumerate(tree_value):
        distance = distances[start][target]
        if distance > 8 or distance + 2 >= remaining or target in claimed:
            continue
        if target in trees and trees[target][0]:
            continue
        cost = seize if target in trees else price
        if cash < cost:
            if distance == 0 or cash < seize * 0.60:
                continue
            yield_value *= 0.66
        if trespass[target] == 0 or home[target] == 0:
            continue
        profit = yield_value - cost
        if profit < 4 and target == start:
            continue
        value = profit / (7.0 + 2.8 * distance)
        value += 0.32 * (target == old_target)
        navigate(start, options, target, value, rows, distances, 2.8)
    for dest, (score, target) in rows.items():
        risk = sum(probability * (30 if role == "COLLECTOR" else 17)
                   for probability, _, role in threat[dest])
        if dest == start and target != start:
            risk += 6
        if dest == start and target == start:
            cost = seize if dest in trees and not trees[dest][0] else price
            if cash >= cost and (dest not in trees or not trees[dest][0]) and home[dest] and trespass[dest]:
                if tree_value[dest] < cost * 1.04 and len(options) > 1:
                    risk += 12
        rows[dest] = (score - risk, target)
    return rows


def agent(observation, configuration):
    step, player = observation["step"], observation["player"]
    size = configuration.get("size", 15)
    remaining = configuration.get("episodeSteps", 300) - 1 - step
    neighbors, vicinity, mirrors, distances = geometry(size)
    if not step or STATE.get("step", -1) >= step or STATE.get("size") != size:
        STATE.clear()
    STATE.update(step=step, size=size)
    cash, bases, workers, own_trees = observation["players"][player]
    _, enemy_bases, enemies, enemy_trees = observation["players"][1 - player]
    own_base_cells, enemy_base_cells = set(bases.values()), set(enemy_bases.values())
    if not own_base_cells or not enemy_base_cells:
        return {}
    all_bases = own_base_cells | enemy_base_cells
    trees = {pos: (own, age) for own, group in ((True, own_trees), (False, enemy_trees))
             for pos, age in group.values()}
    home = [min(distances[pos][base] for base in own_base_cells) for pos in range(size * size)]
    trespass = [min(distances[pos][base] for base in enemy_base_cells) for pos in range(size * size)]
    carbon, coverage = update_map(observation, configuration, workers, enemies, all_bases,
                                  trees, STATE, neighbors, mirrors)
    threat = enemy_moves(enemies, enemy_base_cells, own_base_cells, trees, carbon, neighbors, distances)

    drain = [0] * (size * size)
    for pos, (owned, _) in trees.items():
        if not owned:
            for adj in vicinity[pos]:
                drain[adj] += 1
    lifespan = configuration.get("treeLifespan", 50)
    absorption = configuration.get("cellAbsorptionRate", 0.0375)
    plant = configuration.get("plantCost", 20)
    protected = configuration.get("playerTreeProtectiveNumber", 0)
    price = plant if len(own_trees) <= protected else (
        plant + configuration.get("plantCostInflationRatio", 5)
        * configuration.get("plantCostInflationBase", 1.235) ** sum(age < lifespan for _, age in trees.values())
    )
    seize = configuration.get("seizeCost", 20)
    tree_value = []
    for pos, adjacent in enumerate(neighbors):
        owned, age = trees.get(pos, (False, 0))
        horizon = max(0, min(lifespan - age, remaining - 2))
        income = 0.0
        for cell in adjacent:
            if cell in all_bases or cell in trees:
                continue
            competition = max(1, coverage[cell] + (pos not in trees))
            depletion = min(0.95, absorption * competition)
            # Near our bases, available deposits are worth more to collectors.
            # Across the center line, planting also denies deposits to the opponent.
            territory_weight = 1.0
            if pos not in trees:
                territory_weight += 0.055 * max(-5, min(5, home[cell] - trespass[cell]))
                if home[cell] <= 2:
                    territory_weight *= 0.76
            income += carbon[cell] * territory_weight * (1 - (1 - depletion) ** horizon) / competition
        if pos in trees:
            income *= 1.85 if not owned else 0.0
        # Opposing collectors pay additional carbon in busy routes near their bases.
        if not owned:
            traffic = sum(1 for cell, _, role in enemies.values()
                          if role == "COLLECTOR" and distances[pos][cell] <= 2)
            income += min(1.0, horizon / 16) * (4.5 * traffic + max(0, 4 - trespass[pos]) * 1.8)
        tree_value.append(income)

    jobs = sorted(workers, key=lambda key: (workers[key][2] == "PLANTER", -workers[key][1], key))
    rows, claimed = [], set()
    old_targets = STATE.get("targets", {})
    for key in jobs:
        start, load, role = workers[key]
        options = (start, *neighbors[start])
        if role == "COLLECTOR":
            scores = collector_moves(start, load, options, distances, home, remaining, carbon,
                                     trespass, trees, own_base_cells, drain, coverage,
                                     configuration, old_targets.get(key), claimed, threat, neighbors)
        else:
            scores = planter_moves(start, options, distances, home, trespass, remaining,
                                   trees, carbon, tree_value, price, seize, cash, threat,
                                   claimed, old_targets.get(key))
        rows.append(scores)
        best = max(scores.values())[1]
        if best not in all_bases:
            claimed.add(best)

    moves = assign_moves(rows)
    actions, targets = {}, {}
    future = set(moves)
    tree_budget = 0.0
    for key, dest, row in zip(jobs, moves, rows):
        start, _, role = workers[key]
        targets[key] = row[dest][1]
        if dest != start:
            actions[key] = MOVES[neighbors[start].index(dest)]
        elif role == "PLANTER" and dest not in all_bases:
            if dest not in trees:
                tree_budget += price
            elif not trees[dest][0]:
                tree_budget += seize
    STATE["targets"] = targets

    collectors = sum(role == "COLLECTOR" for _, _, role in workers.values())
    planters = len(workers) - collectors
    enemy_collectors = sum(role == "COLLECTOR" for _, _, role in enemies.values())
    collector_cost = configuration.get("recCollectorCost", 30) + (
        collectors + enemy_collectors) * configuration.get("recCollectorIncreaseCost", 0)
    planter_cost = configuration.get("recPlanterCost", 30) + (
        planters + len(enemies) - enemy_collectors) * configuration.get("recPlanterIncreaseCost", 0)
    worker_limit = configuration.get("workerLimit", 10)
    if remaining < 12:
        return actions

    def base_quality(item):
        _, base = item
        return -sum(carbon[pos] for pos in neighbors[base])

    for key, base in sorted(bases.items(), key=base_quality):
        if collectors + planters >= worker_limit or base in future:
            continue
        if any(role == "COLLECTOR" and load < 0.01 for _, load, role in threat[base]):
            continue
        want_planter = planters < MAX_PLANTERS and remaining > 24 and (
            (step == 0 and (OPENING == "trees" or (OPENING == "mixed" and not planters)))
            or (not planters and (step >= 12 or OPENING != "collectors")
                and (collectors >= 1 or cash >= planter_cost + plant))
            or (collectors >= 4 and cash >= 105 + 25 * planters and price <= 105)
        )
        if want_planter and cash - tree_budget >= planter_cost + plant:
            actions[key] = "RECPLANTER"
            cash -= planter_cost
            planters += 1
        elif cash - tree_budget >= collector_cost and (collectors == 0 or remaining > 19):
            if step < 8 and collectors >= 2 and cash < 70:
                continue
            actions[key] = "RECCOLLECTOR"
            cash -= collector_cost
            collectors += 1
    return actions
