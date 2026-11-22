"""Carry the identified opponent and the selected policy into simulated future turns."""
def _v5_rollout_set(own_side, own_style, adversary):
    identity = _V5_EYE[own_style]
    policies = [None,None]
    for side, distribution, mix in ((own_side,identity,.68),(1-own_side,adversary,.72)):
        weights = np.tensordot(distribution, _V5['coef'],axes=1).astype(np.float32)
        biases = np.tensordot(distribution, _V5['bias'],axes=1).astype(np.float32)
        policies[side] = (mix*weights,mix*biases)
    _ST['v5_history']['rollout'] = policies


def _v5_rollout_update(x, hidden, workers, bases, sides):
    planes=x.reshape(len(x),NPL,225)
    for side, (weight,bias) in enumerate(_ST['v5_history']['rollout']):
        wanted=np.flatnonzero(np.asarray(sides)==side)
        if not len(wanted):
            continue
        for role,plane in enumerate((2,4,12)):
            local,cells=np.nonzero(planes[wanted,plane]>.5)
            if not len(cells):
                continue
            rows=wanted[local]
            output=bases if role==2 else workers
            count=output.shape[-1]
            output[rows,cells]+=hidden[rows,cells]@weight[role,:count].T+bias[role,:count]
