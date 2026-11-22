"""Export the tree-guard challenger with a batched move sampler for simulated boards."""
import argparse
import hashlib
import inspect
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GUARD_COMMIT = '9e99a57fb2d16783c53ab351c197f82b877877ff'
GUARD_SHA = '9b4fe64e37e5158b1cc04e09972cb54d5613740f8678eb57dac3d257a158f15f'


def _draw_rows(logits, cells, rng):
    """One RandomState uniform per row, in original unit order."""
    rows = logits[cells]
    weights = np.exp((rows - rows.max(axis=1, keepdims=True)) / _TEMP)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    cumulative = probabilities.cumsum(axis=1, dtype=np.float64)
    cumulative /= cumulative[:, -1:]
    choices = (rng.random_sample(len(cells))[:, None] >= cumulative).sum(axis=1)
    return probabilities, choices


def _sample(players, me, lw, lb, rng, step, fixed=None):
    cash, bases, units, _ = players[me]
    actions, occupied = {}, {}
    population = len(units)
    if bases:
        _, hires = _draw_rows(lb, list(bases.values()), rng)
        for (key, pos), hire in zip(bases.items(), hires):
            if hire and cash >= 30 and population < 10:
                actions[key] = BACT[hire]
                cash -= 30
                population += 1
                occupied[pos] = 'X'

    for key, action in (fixed or {}).items():
        if key not in units:
            continue
        pos, _, role = units[key]
        target = _ROLLOUT_LANDING[pos][WIDX.get(action or 'NONE', 0)]
        role = 'C' if role == 'COLLECTOR' else 'P'
        occupied[target] = role if occupied.get(target) in (None, role) else 'X'
        if action:
            actions[key] = action

    free = [(key, unit) for key, unit in units.items() if not fixed or key not in fixed]
    if not free:
        return actions
    probabilities, sampled = _draw_rows(lw, [unit[0] for _, unit in free], rng)
    # Python's stable sort preserves the original unit order for ties.
    priority = sorted(range(len(free)), key=lambda i: -probabilities[i, sampled[i]])
    home = set(bases.values())
    last_turns = LAST_TURN - step < _BANK
    for i in priority:
        key, (pos, _, role) = free[i]
        role = 'C' if role == 'COLLECTOR' else 'P'
        choice = int(sampled[i])
        destinations = _ROLLOUT_LANDING[pos]
        target = destinations[choice]
        if target in occupied:
            may_bank = last_turns and role == occupied[target] == 'C' and target in home
            if not may_bank:
                # Only conflicted units need to rank their remaining four actions.
                for alternative in np.argsort(-probabilities[i]):
                    target = destinations[alternative]
                    if target not in occupied:
                        choice = int(alternative)
                        break
                else:
                    continue
        occupied[target] = role
        if choice:
            actions[key] = WACT[choice]
    return actions


def export(source, output):
    if hashlib.sha256(source).hexdigest() != GUARD_SHA:
        raise ValueError('The source must match the archived tree-guard challenger')
    text = source.decode()
    start = text.index('def _sample(')
    end = text.index('def _deviations(', start)
    table = '_ROLLOUT_LANDING = tuple(tuple(move(pos, a) for a in range(5)) for pos in range(N))'
    text = text[:start] + '\n\n'.join((table, inspect.getsource(_draw_rows),
                                      inspect.getsource(_sample))) + '\n\n' + text[end:]
    compile(text, str(output), 'exec')
    with output.open('x', encoding='utf-8') as stream:
        stream.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', GUARD_COMMIT + ':submission.py'])
    export(source, args.output)
