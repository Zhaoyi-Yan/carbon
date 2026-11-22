"""Ensemble the policy in four orientations while preserving move semantics."""
def _v5_symmetry_maps():
    # A rotation maps a physical cell and every directional action together.
    r, c = np.indices((15, 15)).reshape(2, 225)
    actions = np.arange(5)
    rotate = np.array([0, 2, 3, 4, 1])
    result = []
    for _ in range(4):
        cells = (r % 15) * 15 + c % 15
        channels = np.arange(NPL)
        for start in (25, 30):
            channels[start + actions] = start + np.arange(5)
        result.append((cells, np.argsort(cells), actions, channels))
        r, c = c, -r
        actions = rotate[actions]
    return result


_V5_ROTATIONS = _v5_symmetry_maps()


def _v5_rotated_heads(x, opponent_x, mine, rival):
    frames = np.stack((x, opponent_x)).reshape(2, NPL, N)
    batch = [frames[:, channels][:, :, inverse].reshape(2, NPL, SIZE, SIZE)
             for cells, inverse, actions, channels in _V5_ROTATIONS]
    latent = _net()(np.concatenate(batch).astype(np.float32), _SID, hidden=True)
    prior = _POLICY.p
    ww = prior['hw.w'].reshape(5, -1).T
    bw = prior['hb.w'].reshape(3, -1).T
    sides = []
    for side, player in enumerate((mine, rival)):
        workers = np.zeros((225, 5), dtype=np.float32)
        bases = np.zeros((225, 3), dtype=np.float32)
        offsets = {}
        for i, (cells, _, actions, _) in enumerate(_V5_ROTATIONS):
            hidden = latent[2*i+side, cells]
            workers += (hidden @ ww + prior['hw.b'])[:, actions] * .25
            bases += (hidden @ bw + prior['hb.b']) * .25
            for uid, (pos, _, kind) in player[2].items():
                role = 0 if kind == 'COLLECTOR' else 1
                delta = (_V5['coef'][:, role] @ hidden[pos] + _V5['bias'][:, role])[:, actions]
                if i == 0:
                    offsets[uid] = pos, np.zeros_like(delta), False
                offsets[uid][1][:] += delta * .25
            for uid, pos in player[1].items():
                delta = _V5['coef'][:, 2, :3] @ hidden[pos] + _V5['bias'][:, 2, :3]
                if i == 0:
                    offsets[uid] = pos, np.zeros_like(delta), True
                offsets[uid][1][:] += delta * .25
        sides.append((workers, bases, offsets))
    return sides
