"""Export a frozen v5 candidate that models its own replay-trained policy during rollouts."""
import argparse
from hashlib import sha256
import inspect
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = '0483eca2a5e8031270dfb89b87cd5d901ee81f823383f6094f8ed839dec6a7dd'


def _rollout_replay_head(inputs, hidden, worker, base, stride):
    """Apply the v5 head only to our rows and our occupied cells. Hidden is (batch, C, N)."""
    row_ids, cell_ids, examples = [], [], []
    for row in range(0, len(inputs), stride):
        x = inputs[row].reshape(NPL, N)
        cells = np.flatnonzero(x[2] + x[4] + x[12])
        if not len(cells):
            continue
        dest = _ROLLOUT_DEST[cells]
        neighbors = _ROLLOUT_DEST[dest, 1:]
        supply = x[0, neighbors].mean(axis=2)
        pressure = (x[8] + x[10])[neighbors].mean(axis=2)
        local = x[[10, 9, 11, 39, 40, 43, 47, 0]][:, dest].transpose(1, 0, 2).reshape(len(cells), -1)
        examples.append(np.concatenate((hidden[row][:, cells].T, x[:, cells].T, supply, pressure, local), axis=1))
        row_ids.extend([row] * len(cells))
        cell_ids.extend(cells)
    if not examples:
        return
    weights = _REPLAY_HEAD
    features = np.clip((np.concatenate(examples) - weights['mean']) * weights['scale'], -6, 6)
    activation = np.maximum(features @ weights['layer'] + weights['bias'], 0)
    ix = np.asarray(row_ids), np.asarray(cell_ids)
    worker[ix] = (.62 * worker[ix] + (activation @ weights['worker'] + weights['worker_bias'])) / .46
    base[ix] = (.62 * base[ix] + (activation @ weights['base'] + weights['base_bias'])) / .46


def candidate_source(source):
    if sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError('Expected the published v5-tree-value source')
    text = source.decode()
    def patch(before, after, count=1):
        nonlocal text
        if text.count(before) != count:
            raise ValueError(f'Archive does not contain the expected {before!r}')
        text = text.replace(before, after)

    helper = inspect.getsource(_rollout_replay_head)
    template = '_ROLLOUT_DEST = np.array([[move(pos, a) for a in range(5)] for pos in range(N)])'
    patch('class _TorchNet:', template + '\n\n\n' + helper + '\n\nclass _TorchNet:')
    patch('def __call__(self, X, sid, sc=None):', 'def __call__(self, X, sid, sc=None, actor_stride=0):', 2)
    for end_logits, hidden in [("lb = (_torch.einsum('kc,ncp->npk', self.hb[0], hf) + self.hb[1]).numpy()",
                                'hf.numpy()'),
                               ("lb = (hf @ self.hb[0] + self.hb[1]).permute(1, 0, 2).contiguous().numpy()",
                                'hf.permute(1, 2, 0).numpy()')]:
        patch(end_logits, end_logits + '\n            if actor_stride:\n'
              '                _rollout_replay_head(X, ' + hidden + ', lw, lb, actor_stride)')
    patch('def look(live, both):', 'def look(live, both, continues):')
    patch('lw, lb, vv = net(X, _SID, S)', 'lw, lb, vv = net(X, _SID, S, actor_stride=w if continues else 0)')
    patch('v, pol = look(live, both)', 'v, pol = look(live, both, t < D2)')
    patch('v, pol = look(live, t < D2)', 'v, pol = look(live, t < D2, t < D2)')
    compile(text, '<v5-own-rollout>', 'exec')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = (ROOT / 'submission.py').read_bytes()
    if sha256(source).hexdigest() != BASE_SHA256:
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'v5-tree-value:submission.py'])
    result = candidate_source(source)
    with args.output.open('x') as f:
        f.write(result)
    print(sha256(result.encode()).hexdigest(), args.output)


if __name__ == '__main__':
    main()
