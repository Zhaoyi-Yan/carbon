"""A local v5 stress opponent that detaches one empty collector to contest return routes."""
from hashlib import sha256
from pathlib import Path
import subprocess
from types import ModuleType


BASE_SHA256 = '0483eca2a5e8031270dfb89b87cd5d901ee81f823383f6094f8ed839dec6a7dd'
ROOT = Path(__file__).resolve().parents[1]


def load_v5():
    path = ROOT / 'opponents' / 'yan-v2.py'
    source = path.read_bytes()
    if sha256(source).hexdigest() != BASE_SHA256:
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'v5-tree-value:submission.py'])
        if sha256(source).hexdigest() != BASE_SHA256:
            raise RuntimeError('Pressure opponent requires the original v5-tree-value weights')
    module = ModuleType('pressure_v5')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


v5 = load_v5()
_SEARCH_OK = v5._SEARCH_OK
_S_STATS = v5._S_STATS
scout = None


def harass(obs, actions, candidate):
    own, enemy = obs['players'][obs['player']], obs['players'][1-obs['player']]
    homes = set(enemy[1].values())
    collectors = {key: pos for key, (pos, load, role) in own[2].items() if role == 'COLLECTOR'}
    empty = {key: pos for key, pos in collectors.items() if own[2][key][1] < 1}
    if len(collectors) < 3 or not empty or not homes:
        return None
    distance = v5.DIST

    def home_distance(cell):
        return min(distance[cell][home] for home in homes)

    if candidate not in empty:
        candidate = min(empty, key=lambda key: (home_distance(empty[key]), key))
    start = empty[candidate]
    reserved = {v5.move(w[0], v5.WIDX.get(actions.get(key), 0))
                for key, w in own[2].items() if key != candidate}
    reserved.update(pos for key, pos in own[1].items() if key in actions)
    options = [(k, v5.move(start, k)) for k in range(5)]
    options = [(k, pos) for k, pos in options if pos not in reserved and pos not in homes]
    if not options:
        return candidate

    # Intercept the next inward step of a visible laden enemy. Among equivalent
    # routes favour v5's original move and avoid a tie with an opposing empty collector.
    inward = {}
    guards = []
    for pos, load, kind in enemy[2].values():
        if kind != 'COLLECTOR':
            continue
        if load < 1:
            guards.append(pos)
        if load >= 40 and home_distance(pos) <= 4:
            for cell in v5.NB[pos]:
                if cell not in homes and home_distance(cell) < home_distance(pos):
                    inward[cell] = max(inward.get(cell, 0.), min(load, 300.)/100)
    prior = v5.WIDX.get(actions.get(candidate), 0)

    def score(option):
        k, pos = option
        d = home_distance(pos)
        danger = any(distance[pos][guard] <= 1 for guard in guards)
        # Staying on a rich tile fills the scout and ends its interception role.
        stay_cost = 2.2 if k == 0 and obs['carbon'][pos] > 0 else 0.
        return (-2.5*abs(d-1) + inward.get(pos, 0.) - 2.8*danger - stay_cost
                + .3*(k == prior))

    k, _ = max(options, key=score)
    if k:
        actions[candidate] = v5.WACT[k]
    else:
        actions.pop(candidate, None)
    return candidate


def agent(observation, configuration):
    global scout
    step = observation['step']
    if step == 0:
        scout = None
    actions = v5.agent(observation, configuration)
    if configuration.get('size', 15) == 15 and 12 <= step < configuration.get('episodeSteps', 300) - 25:
        scout = harass(observation, actions, scout)
        # Keep v5's partial-observation attribution aligned with the emitted action.
        tracker = v5._ST.get('tr')
        if tracker is not None:
            tracker.set_actions(observation['players'], actions)
    return actions
