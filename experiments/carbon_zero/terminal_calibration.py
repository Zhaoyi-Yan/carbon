"""Measure whether a checkpoint's critic stays calibrated on its own policy.

Plays each checkpoint against the frozen ZW0 reference on a fixed map panel,
labels every visited state with the full-game shaped return U - Phi, and reports
value accuracy plus full-game GAE-vs-MC advantage differences.  The cash
comparator uses the pre-trained B1 baseline coefficients and is never refit on
this panel.  The `ZW0critic` checkpoint is the zero-step baseline: ZW0 actor with
the pre-trained critic attached.
"""
import argparse
import hashlib
import json
from pathlib import Path

from pilot import Actor, FT3_SHA256, STYLE, load_ft3, np, torch, write
from policy import actor_from_checkpoint, public_state_features
from terminal_games import Match, TEMPERATURE, keyed_seed, play

CRITIC_SHA256 = 'fceedd2e7944394f2119377ae30997bc86adb8db00272f0dcd14e3548540ec47'
LAMBDA = .95


def panel(count, namespace='terminal-calibration-v1'):
    matches = []
    for i in range(count):
        layout = keyed_seed(namespace, i, 'layout') & 0xffffffff
        for seat in (0, 1):
            keys = (namespace, i, seat)
            matches.append(Match(layout, seat, keyed_seed(*keys, 'learner'),
                                 keyed_seed(*keys, 'opponent'), i))
    return matches


def predict(actor, features, pub, device):
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        if actor.residual_critic:
            out = actor.predict_value(x, torch.as_tensor(pub, dtype=torch.float32, device=device))
        else:
            out = actor.value(x)
        return out.detach().cpu().numpy().reshape(-1)


def corr(x, y):
    x, y = np.ravel(x), np.ravel(y)
    if len(x) != len(y):
        raise ValueError(f'correlation inputs differ in length: {len(x)} vs {len(y)}')
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def full_game_advantages(rewards, values, lam):
    """Full-game GAE with gamma=1; lam=1 is exactly the MC advantage."""
    rewards, values = np.ravel(rewards), np.ravel(values)
    if rewards.shape != values.shape:
        raise ValueError(f'reward/value shapes differ: {rewards.shape} vs {values.shape}')
    out = np.zeros_like(values)
    acc = 0.
    for t in range(len(values) - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < len(values) else 0.
        acc = rewards[t] + nxt - values[t] + lam * acc
        out[t] = acc
    return out


def metrics(pred, label):
    pred, label = np.ravel(pred), np.ravel(label)
    if pred.shape != label.shape:
        raise ValueError(f'prediction/label shapes differ: {pred.shape} vs {label.shape}')
    residual = label - pred
    var = float(np.var(label))
    return dict(N=len(label), MSE=float(np.mean(residual ** 2)), bias=float(np.mean(pred - label)),
                EV=None if var == 0 else float(1. - np.var(residual) / var),
                corr=corr(pred, label))


def analyse(played, actor, cash_w, device):
    features, pub, label, turns = [], [], [], []
    game_adv, traces = [], []
    for row, trace, _ in played:
        if not trace:
            continue
        feats = np.stack([t[0] for t in trace])
        gaps = np.asarray([t[1] for t in trace], dtype=np.float64)
        steps = np.asarray([t[2] for t in trace], dtype=np.int64)
        phi = 0.5 * np.tanh(gaps / 1000.)
        pub_g = public_state_features(gaps, steps)
        value = predict(actor, feats, pub_g, device)
        rewards = np.empty_like(phi)
        rewards[:-1] = phi[1:] - phi[:-1]
        rewards[-1] = row['utility'] - phi[-1]
        shape = (len(trace),)
        for name, array in (('value', value), ('phi', phi), ('reward', rewards)):
            if array.shape != shape:
                raise ValueError(f'{name} has shape {array.shape}, expected {shape}')
        gae = full_game_advantages(rewards, value, LAMBDA)
        mc = row['utility'] - phi - value
        identity = full_game_advantages(rewards, value, 1.0)
        gap = float(np.max(np.abs(identity - mc)))
        if gap > 1e-5:
            raise AssertionError(f'lambda=1 GAE is not the MC advantage: {gap} '
                                 f'seed={row["seed"]} seat={row["seat"]} T={len(trace)} U={row["utility"]}')
        features.append(feats); pub.append(pub_g); label.append(row['utility'] - phi); turns.append(steps)
        game_adv.append(dict(seed=row['seed'], seat=row['seat'], utility=row['utility'],
                             gae=gae, mc=mc, turn=steps))
        traces.append(dict(seed=row['seed'], seat=row['seat'], turns=int(len(trace)),
                           utility=row['utility'], value=np.round(value, 6).tolist(),
                           phi=np.round(phi, 6).tolist()))
    if not features:
        return None
    features = np.concatenate(features); pub = np.concatenate(pub)
    label = np.concatenate(label); turns = np.concatenate(turns)
    value = predict(actor, features, pub, device)
    cash = (pub.astype(np.float64) @ cash_w[0]).astype(np.float64)
    zero = np.zeros_like(value)
    all_gae = np.concatenate([g['gae'] for g in game_adv])
    all_mc = np.concatenate([g['mc'] for g in game_adv])
    phases = {'all': np.ones(len(turns), bool), 'opening': turns <= 60,
              'middle': (turns > 60) & (turns < 230), 'ending': turns >= 230}
    report = {'states': int(len(label)), 'games': int(len(game_adv)),
              'value': {}, 'cash': {}, 'zero': {}, 'gae_vs_mc': {}, 'traces': traces}
    for name, mask in phases.items():
        report['value'][name] = metrics(value[mask], label[mask])
        report['cash'][name] = metrics(cash[mask], label[mask])
        report['zero'][name] = metrics(zero[mask], label[mask])
        report['gae_vs_mc'][name] = dict(
            N=int(mask.sum()), mean_abs=float(np.mean(np.abs(all_gae[mask] - all_mc[mask]))),
            mean_signed=float(np.mean(all_gae[mask] - all_mc[mask])),
            corr=corr(all_gae[mask], all_mc[mask]),
            mean_gae=float(np.mean(all_gae[mask])), mean_mc=float(np.mean(all_mc[mask])))
    report['per_map'] = {str(seed): float(np.mean([g['utility'] for g in game_adv
                                                   if g['seed'] == seed])) for seed in sorted({g['seed'] for g in game_adv})}
    return report


def main(a):
    torch.set_num_threads(2)
    critic_sha = hashlib.sha256(a.critic.read_bytes()).hexdigest()
    if critic_sha != CRITIC_SHA256:
        raise ValueError(f'critic hash {critic_sha} does not match the frozen digest')
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    reference = Actor(teacher._POLICY, style).to(a.device).eval()
    critic = torch.load(a.critic, map_location='cpu', weights_only=True)
    cash_w = critic['value_baseline']['weight'].double().numpy()
    matches = panel(a.maps)
    results = {}
    for item in a.checkpoint:
        name, _, path = item.partition('=')
        if name == 'ZW0':
            actor = reference
            checkpoint_sha = None
        elif name == 'ZW0critic':
            actor = Actor(teacher._POLICY, style, residual_critic=True).to(a.device).eval()
            actor.value_baseline.load_state_dict(critic['value_baseline'])
            actor.value.load_state_dict(critic['value'])
            actor.requires_grad_(False)
            checkpoint_sha = critic_sha
        else:
            actor, _ = actor_from_checkpoint(teacher, style, Path(path), a.device)
            checkpoint_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        played, timing = play(teacher, actor, reference, matches, a.device, capture='critic')
        results[name] = analyse(played, actor, cash_w, a.device)
        results[name].update(checkpoint_sha256=checkpoint_sha, seconds=timing['elapsed'])
        print('calibrated', name, flush=True)
    write(a.out, dict(panel_namespace='terminal-calibration-v1', maps=a.maps,
                      temperature=TEMPERATURE, lambda_gae=LAMBDA,
                      source_sha256=FT3_SHA256, critic_sha256=critic_sha,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      results=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('critic', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--checkpoint', action='append', required=True,
                   help='name=path; ZW0 and ZW0critic need no path')
    p.add_argument('--maps', type=int, default=8)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
