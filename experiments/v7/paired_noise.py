"""Couple future actions by persistent worker ID when evaluating alternative moves."""
import inspect
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=128)
def _paired_random_block(seed, group, block):
    # Base and worker suffixes live in separate namespaces. Reusing a seed gives every
    # survivor the same variate even when another worker dies on just one branch.
    stream=(int(seed)+2654435761*group+2246822519*block)&0xffffffff
    return np.random.RandomState(stream).random_sample(256)


def _paired_uniforms(seed, keys, group):
    values=[]
    for uid in keys:
        index=int(uid.rsplit('-',1)[1])
        values.append(_paired_random_block(seed,group,index//256)[index%256])
    return np.asarray(values)


def build(code):
    changes={
        'def _draw_rows(logits, cells, rng):':
            'def _draw_rows(logits, cells, rng, draws=None):',
        '    choices = (rng.random_sample(len(cells))[:, None] >= cumulative).sum(axis=1)':
            '    random_rows = rng.random_sample(len(cells)) if draws is None else draws\n    choices = (random_rows[:, None] >= cumulative).sum(axis=1)',
        'def _sample(players, me, lw, lb, rng, step, fixed=None):':
            'def _sample(players, me, lw, lb, rng, step, fixed=None, paired_seed=None):',
        '        _, hires = _draw_rows(lb, list(bases.values()), rng)':
            '        draws = _paired_uniforms(paired_seed,bases,1) if paired_seed is not None else None\n        _, hires = _draw_rows(lb, list(bases.values()), rng, draws)',
        '    probabilities, sampled = _draw_rows(lw, [unit[0] for _, unit in free], rng)':
            '    draws = _paired_uniforms(paired_seed,(u for u,_ in free),2) if paired_seed is not None else None\n    probabilities, sampled = _draw_rows(lw, [unit[0] for _, unit in free], rng, draws)',
        'np.random.RandomState(seed+373*t+17*s),t) for s in (0,1)]':
            'None,t,paired_seed=seed+373*t+17*s) for s in (0,1)]',
        '            r1 = np.random.RandomState(seed0 + 7919 * t + 31 * m + 1); r2 = np.random.RandomState(seed0 + 7919 * t + 31 * m + 2)':
            '            r1 = seed0 + 7919 * t + 31 * m + 1\n            r2 = r1 + 1',
        "lwm, lbm, r1, o['step'])":"lwm, lbm, None, o['step'], paired_seed=r1)",
        "lwp, lbp, r2, o['step'])":"lwp, lbp, None, o['step'], paired_seed=r2)",
    }
    for old,new in changes.items():
        if code.count(old)!=1:raise ValueError('unexpected simulator sampling call: '+old)
        code=code.replace(old,new)
    anchor='def _draw_rows('
    funcs='from functools import lru_cache\n'+inspect.getsource(_paired_random_block)+'\n'+inspect.getsource(_paired_uniforms)+'\n'
    return code.replace(anchor,funcs+anchor)
