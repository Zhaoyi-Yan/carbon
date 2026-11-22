"""Fit the Carbon League action heads and tree-value head from public match replays.

Example: python train_replays.py --output /tmp/carbon-weights
Dependencies: NumPy and PyTorch. GPU acceleration is optional.
The strategy uses the exported arrays at inference, without access to replays.
"""
import argparse
import ast
import base64
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
# Indices in the published snapshot, sorted by match ID. All states of a game stay together.
VALIDATION_INDICES = {1, 5, 9, 13, 17}


def load_strategy(path):
    spec = importlib.util.spec_from_file_location('carbon_strategy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Backbone:
    def __init__(self, strategy):
        self.net = strategy._TorchNet(strategy._POLICY, None)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.stem = [w.to(self.device) for w in self.net.stem]
        self.embedding = self.net.emb[strategy._SID].to(self.device)[None, :, None, None]
        self.blocks = [[w.to(self.device) for w in block] for block in self.net.blocks]

    @torch.inference_mode()
    def __call__(self, rows):
        conv = self.net._conv
        x = torch.as_tensor(np.stack(rows), device=self.device)
        h = torch.relu(conv(x, *self.stem) + self.embedding)
        for c1, b1, gate, bias, c2, b2 in self.blocks:
            a = torch.relu(conv(h, c1, b1))
            a = a + (a.mean((2, 3)) @ gate.T + bias)[:, :, None, None]
            h = torch.relu(h + conv(a, c2, b2))
        return h.cpu().numpy()


def action_labels(strategy, player, action, future_workers):
    rows = []
    for wid, (pos, _, role) in player[2].items():
        direction = strategy.WIDX[action.get(wid) or 'NONE']
        future = future_workers.get(wid)
        displacement = 9  # No longer on board six steps later.
        if future is not None:
            later = future[0]
            dr = (later // 15 - pos // 15 + 7) % 15 - 7
            dc = (later % 15 - pos % 15 + 7) % 15 - 7
            displacement = (int(np.sign(dr)) + 1) * 3 + int(np.sign(dc)) + 1
        rows.append((pos, direction, int(role == 'PLANTER'), displacement))
    if player[0] >= 30 and len(player[2]) < 10:
        for bid, pos in player[1].items():
            rows.append((pos, strategy.BIDX[action.get(bid) or 'NONE'], 2, -1))
    return rows


def action_examples(strategy, backbone, inputs, units):
    weights = strategy._POLICY.p
    result = []
    for x, h, rows in zip(inputs, backbone(inputs), units):
        if not rows:
            continue
        pos, act, role, offset = np.asarray(rows).T
        f = strategy._replay_features(x, h)[pos]
        latent = h.reshape(h.shape[0], -1).T[pos]
        prior = [latent @ weights[k+'.w'].reshape(n, -1).T + weights[k+'.b']
                 for k, n in [('hw', 5), ('hb', 3)]]
        result.append((f, act, role, offset, np.column_stack(prior)))
    return result


def extract(strategy, backbone, game, replay):
    slot = next(k for k, v in game['strategies'].items() if v['name'] == 'CarbonEdge_Apollo')
    side = game['order'].index(slot)
    steps = replay['steps']
    assert all(abs(replay['rewards'][i] - game['scores'][k]) < .02 for i, k in enumerate(game['order']))
    # Positive income belongs to seat 0. A change of ownership flips the target sign.
    income = []
    for state in steps:
        obs = state[0]['observation']
        income.append({key: (1-2*seat) * obs['trees'][key][1]
                       for seat, p in enumerate(obs['players']) for key in p[3]})
    tracker = strategy.BeliefTracker(strategy._Bel, side)
    prev = None
    inputs, units, actor, trees = [], [], [], []
    for t, (cur, nxt) in enumerate(zip(steps, steps[1:])):
        obs = cur[0]['observation']
        players = obs['players']
        carbon, known = tracker.step(players, t, cur[side]['observation']['carbon'])
        inputs.append(strategy._quant(strategy.encode(players, t, side, carbon, known, prev)))
        action = nxt[side]['action'] or {}
        future = steps[min(t+6, len(steps)-1)][0]['observation']['players'][side][2]
        units.append(action_labels(strategy, players[side], action, future))
        tracker.set_actions(players, action)
        prev = players
        keys, owners, features = strategy._tree_features(players, carbon, obs['trees'], t, replay['configuration'])
        if keys:
            future_income = [sum(.96**(h-1) * income[t+h].get(key, 0)
                                 for h in range(1, min(19, len(steps)-t))) for key in keys]
            trees.append((features, np.asarray(future_income) * (1-2*owners)))
        if len(inputs) == 24 or t == len(steps)-2:
            actor.extend(action_examples(strategy, backbone, inputs, units))
            inputs, units = [], []
    def columns(rows):
        return tuple(np.concatenate(a) for a in zip(*rows))
    f, act, role, offset, prior = columns(actor)
    return (f.astype('float16'), act, role, offset, prior.astype('float16')), columns(trees)


def tensors(games):
    return [torch.from_numpy(np.concatenate(col)) for col in zip(*games)]


def normalized(train, val, min_std, limit):
    mean = train.mean(0)
    scale = 1 / train.std(0).clamp_min(min_std)
    return mean, scale, ((train-mean)*scale).clamp(-limit, limit), ((val-mean)*scale).clamp(-limit, limit)


class Actor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer = nn.Linear(width, 96)
        self.worker = nn.Linear(96, 5)
        self.base = nn.Linear(96, 3)
        self.travel = nn.Linear(96, 10)
        nn.init.zeros_(self.worker.weight)
        nn.init.zeros_(self.base.weight)

    def forward(self, x, prior):
        h = F.dropout(F.relu(self.layer(x)), .13, self.training)
        return .62*prior[:, :5]+self.worker(h), .62*prior[:, 5:]+self.base(h), self.travel(h)


def fit_actor(train, validation, output):
    x, act, role, offset, prior = tensors(train)
    vx, va, vr, _, vp = tensors(validation)
    mean, scale, x, vx = normalized(x.float(), vx.float(), .075, 6)
    prior, vp = prior.float(), vp.float()
    torch.manual_seed(311)
    model = Actor(len(mean))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.012)
    best = float('inf')
    for epoch in range(26):
        model.train()
        for idx in torch.randperm(len(x)).split(512):
            r, a = role[idx], act[idx]
            wk, base = r < 2, r == 2
            w, b, t = model(x[idx], prior[idx])
            entropy = F.cross_entropy(w[wk], a[wk], reduction='none')
            loss = (entropy*(1+.45*(r[wk] == 1))).mean() + .32*F.cross_entropy(b[base], a[base])
            loss += .18*F.cross_entropy(t[wk], offset[idx][wk])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        if epoch == 13:
            optimizer.param_groups[0]['lr'] = .0006
        model.eval()
        with torch.no_grad():
            w, b, _ = model(vx, vp)
            losses = [F.cross_entropy(z[vr == r], va[vr == r]).item() for r, z in enumerate([w, w, b])]
            metric = sum(round(loss, 4)*weight for loss, weight in zip(losses, [1, .45, .32]))
            if metric < best:
                best = metric
                params = {'mean': mean, 'scale': scale}
                for name, bias in [('layer', 'bias'), ('worker', 'worker_bias'), ('base', 'base_bias')]:
                    layer = getattr(model, name)
                    params[name], params[bias] = layer.weight.T, layer.bias
                np.savez_compressed(output/'head.npz', **{k: v.numpy().astype('float16') for k,v in params.items()})
                print('actor', epoch, 'validation cross-entropies', losses, flush=True)


def fit_tree(train, validation, output):
    x, y = (v.float() for v in tensors(train))
    vx, vy = (v.float() for v in tensors(validation))
    mean, scale, x, vx = normalized(x, vx, .05, 7)
    torch.manual_seed(311)
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002)
    best = float('inf')
    for epoch in range(32):
        for idx in torch.randperm(len(y)).split(800):
            loss = F.smooth_l1_loss(net(x[idx])[:, 0], y[idx]/50, beta=1.15)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mae = (net(vx)[:, 0]*50-vy).abs().mean().item()
            if mae < best:
                best = mae
                np.savez_compressed(output/'tree.npz', mean=mean.numpy(), scale=scale.numpy(),
                                    w=net[0].weight.numpy().T, b=net[0].bias.numpy(),
                                    out=net[2].weight.numpy()[0]*50, offset=net[2].bias.numpy()*50)
                print('tree', epoch, 'validation MAE', mae, flush=True)


def export_strategy(source_path, output):
    source = source_path.read_text()
    lines = source.splitlines(keepends=True)
    archives = {'_learned': 'head.npz', '_tree_parameters': 'tree.npz'}
    for block in ast.parse(source).body:
        if not isinstance(block, ast.With) or len(block.items) != 1:
            continue
        item = block.items[0]
        if not isinstance(item.optional_vars, ast.Name) or item.optional_vars.id not in archives:
            continue
        name = archives.pop(item.optional_vars.id)
        blob = base64.b64encode((output/name).read_bytes()).decode('ascii')
        [literal] = [node for node in ast.walk(item.context_expr)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        line = lines[literal.lineno-1]
        lines[literal.lineno-1] = line[:literal.col_offset] + '"' + blob + '"' + line[literal.end_col_offset:]
    if archives:
        raise ValueError(f'The strategy template lacks model archives {list(archives)}')
    candidate = ''.join(lines)
    compile(candidate, 'candidate.py', 'exec')
    (output/'candidate.py').write_text(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path, default=ROOT/'submission.py')
    parser.add_argument('--replays', type=Path, default=ROOT/'replays'/'apollo')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    strategy = load_strategy(args.strategy)
    backbone = Backbone(strategy)
    manifest = json.loads((args.replays/'manifest.json').read_text())
    training, validation = [], []
    for i, g in enumerate(sorted(manifest['games'], key=lambda row: row['id'])):
        with gzip.open(args.replays/Path(g['file']).name) as f:
            pair = extract(strategy, backbone, g, json.load(f))
        (validation if i in VALIDATION_INDICES else training).append(pair)
    fit_actor([a for a,_ in training], [a for a,_ in validation], args.output)
    fit_tree([t for _,t in training], [t for _,t in validation], args.output)
    export_strategy(args.strategy, args.output)


if __name__ == '__main__':
    main()
