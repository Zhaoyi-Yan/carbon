"""Distill our own searched worker moves from mirrored self-play into an extra root proposal."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np


def _v7_own_features(board, players, side, prior, previous):
    ids, roles, logits, features = _ds_features(board, players, side, previous, prior, None)
    if not ids:
        return None
    # Only legal per-seat planes and current root logits are used to forecast our searched action.
    return [players[side][2][uid][0] for uid in ids], roles, logits, features


def _v7_search_teacher(board, players, side, prior, previous):
    rows = _v7_own_features(board, players, side, prior, previous)
    if rows is None:
        return prior
    pos, roles, logits, features = rows
    corrected = np.einsum('naf,nf->na', features, _V7_OWN_COEF[roles])
    result = prior.copy()
    result[pos] = logits * _V7_OWN_SCALE[roles, None] + corrected
    return result


def build(source, fit_path):
    model = json.loads(Path(fit_path).read_text())
    anchor = 'def _act(obs, configuration=None, t0=None):'
    observe = '        lw=_selfcredit_adjust(x,lw)'
    root = '    if most>=3:\n        tree_action=_r6_first_tree'
    if any(source.count(s) != 1 for s in (anchor, observe, root)):
        raise ValueError("Unrecognized economic search")
    constants = (f"_V7_OWN_COEF=np.asarray({model['coefficients']!r},np.float32)\n"
                 f"_V7_OWN_SCALE=np.asarray({model['scale']!r},np.float32)\n")
    text = source.replace(anchor, constants + inspect.getsource(_v7_own_features) + '\n' +
                          inspect.getsource(_v7_search_teacher) + '\n' + anchor)
    text = text.replace(observe, observe + "\n        st['own_search_prior']=_v7_search_teacher(x,players,me,lw,st['prev'])")
    insert = """    if most >= 4 and len(cands) < most:
        teacher = _ST.get('own_search_prior')
        if teacher is not None:
            # Use a separate RNG to keep the ordinary root proposal stream reproducible.
            bases = lb0.copy()
            for uid, pos in players[me][1].items():
                original = act0.get(uid)
                action = BACT.index(original) if original else 0
                bases[pos] = -100.
                bases[pos, action] = 0.
            option = _sample(players, me, teacher * _TEMP, bases, np.random.RandomState(seed0 + 45289), step, fixed)
            marker = tuple(sorted(option.items()))
            if marker not in keys:
                cands.append(option)
                keys.add(marker)
"""
    return text.replace(root, insert + root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    code = build(args.parent.read_text(), Path(__file__).with_name('own-search-model.json'))
    compile(code, str(args.output), 'exec')
    if len(code.encode()) >= 10_000_000:
        raise ValueError("Tournament file exceeds size limit")
    with args.output.open('x') as dest:
        dest.write(code)
