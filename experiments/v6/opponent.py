"""Build an opponent-only, online confidence calibration experiment from frozen yan-v5."""
import argparse
import hashlib
import inspect
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA256 = '0a413a11f80f6872a5c8f0d313061d7bf11121ae16fbdad97247ff3cb15242e5'
TEMPERATURES = (.40, .65, 1., 1.6, 2.5, 4., 6.4)


def _v6_opponent_temperature(state, step, side, players):
    """Score the last predicted directions when that opponent worker is observed again."""
    # Fixed-share, exponentially discounted evidence lets the opponent change its policy midgame.
    if 'calibration' not in state:
        state['calibration'] = dict(loss=np.zeros((2, len(_V6_TEMPS))), pending=[], turn=-1)
    history = state['calibration']
    if history['turn'] == step:
        return
    loss = history['loss']
    loss *= .96
    if history['turn'] == step - 1:
        for uid, start, role, logp in history['pending']:
            unit = players[side][2].get(uid)
            if unit is None or unit[0] not in _ROLLOUT_LANDING[start]:
                continue
            direction = _ROLLOUT_LANDING[start].index(unit[0])
            # Cap any single censored/collision-affected observation's influence.
            score = np.maximum(logp[:, direction], -9.)
            loss[role] += .35 * (score - score.max())
    # A weak prior close to the incumbent leaves familiar opponents largely unchanged.
    posterior = loss - .40 * (np.log(_V6_TEMPS[None, :] / _TEMP))**2
    posterior -= posterior.max(axis=1, keepdims=True)
    posterior = np.exp(posterior)
    posterior /= posterior.sum(axis=1, keepdims=True)
    # During cold start blend smoothly back towards the incumbent inverse temperature.
    trust = min(1., step / 10.)
    multiplier = np.exp(trust * (posterior @ np.log(_TEMP / _V6_TEMPS)))
    state['opponent_scale'] = (side, multiplier.astype(np.float32))
    workers, _ = state['opponent']
    pending = []
    for uid, (pos, _, kind) in players[side][2].items():
        role = int(kind == 'PLANTER')
        raw = workers[pos].astype(np.float64)
        z = (raw - raw.max())[None, :] / _V6_TEMPS[:, None]
        pending.append((uid, pos, role, z - np.log(np.exp(z).sum(axis=1, keepdims=True))))
        workers[pos] *= multiplier[role]
    history['turn'], history['pending'] = step, pending


def source_for(template):
    if hashlib.sha256(template).hexdigest() != BASE_SHA256:
        raise ValueError('Expected frozen yan-v5')
    text = template.decode()
    helper = '_V6_TEMPS = np.array(' + repr(TEMPERATURES) + ')\n\n' + inspect.getsource(_v6_opponent_temperature)
    replacements = [
        ('def _v5_root(obs,', helper + '\n\ndef _v5_root(obs,'),
        ("    state['opponent'] = _v5_mix(enemy,rival,.8)\n", "    state['opponent'] = _v5_mix(enemy,rival,.8)\n    _v6_opponent_temperature(state, step, op, players)\n"),
        ('            output[rows,cells]+=hidden[rows,cells]@weight[role,:count].T+bias[role,:count]',
         """            output[rows,cells]+=hidden[rows,cells]@weight[role,:count].T+bias[role,:count]
            rival, scale = _ST['v5_history']['opponent_scale']
            if side == rival and role < 2:
                output[rows,cells] *= scale[role]"""),
    ]
    for old,new in replacements:
        if text.count(old) != 1:
            raise ValueError(f'Unexpected template at {old!r}')
        text = text.replace(old,new)
    compile(text, '<v6-opponent>', 'exec')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = (ROOT/'yan-v5.py').read_bytes()
    source = source_for(data)
    with args.output.open('x') as f:
        f.write(source)
    print(hashlib.sha256(source.encode()).hexdigest(),args.output)


if __name__ == '__main__':
    main()
