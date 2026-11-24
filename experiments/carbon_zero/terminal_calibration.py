"""Measure whether a checkpoint's critic stays calibrated on its own policy.

Plays each checkpoint against the frozen ZW0 reference on a fixed map panel,
labels every visited state with the full-game shaped return U - Phi, and reports
value accuracy and GAE-vs-MC advantage differences.  The cash comparator uses the
pre-trained B1 baseline coefficients and is never refit on this panel.
"""
import argparse
import json
from pathlib import Path

from pilot import Actor, STYLE, load_ft3, np, torch, write
from policy import actor_from_checkpoint, public_state_features
from terminal_games import Match, TEMPERATURE, keyed_seed, play


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
            return actor.predict_value(x, torch.as_tensor(pub, dtype=torch.float32, device=device)).cpu().numpy()
        return actor.value(x).cpu().numpy()


def corr(x, y):
    x, y = np.ravel(x), np.ravel(y)
    if len(x) != len(y) or len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(pred, label):
    pred, label = np.ravel(pred), np.ravel(label)
    residual = label - pred
    var = float(np.var(label))
    return dict(N=len(label), MSE=float(np.mean(residual ** 2)), bias=float(np.mean(pred - label)),
                EV=None if var == 0 else float(1. - np.var(residual) / var),
                corr=corr(pred, label))


def phase_of(turn):
    return 'opening' if turn <= 60 else ('middle' if turn < 230 else 'ending')


def analyse(played, actor, cash_w, device):
    features, pub, label, turns, maps = [], [], [], [], []
    game_adv = []
    for row, trace, _ in played:
        if not trace:
            continue
        feats = np.stack([t[0] for t in trace])
        gaps = np.asarray([t[1] for t in trace], dtype=np.float64)
        steps = np.asarray([t[2] for t in trace], dtype=np.int64)
        phi = 0.5 * np.tanh(gaps / 1000.)
        pub_g = public_state_features(gaps, steps)
        value = predict(actor, feats, pub_g, device)
        target = row['utility'] - phi
        features.append(feats); pub.append(pub_g); label.append(target); turns.append(steps)
        # GAE(lambda=.95) over the full game, no bootstrap at the terminal.
        rewards = np.empty_like(phi)
        rewards[:-1] = phi[1:] - phi[:-1]
        rewards[-1] = row['utility'] - phi[-1]
        gae = np.zeros_like(value)
        acc = 0.
        for t in range(len(value) - 1, -1, -1):
            nxt = value[t + 1] if t + 1 < len(value) else 0.
            delta = rewards[t] + nxt - value[t]
            acc = delta + .95 * acc
            gae[t] = acc
        mc = row['utility'] - phi - value
        game_adv.append(dict(seed=row['seed'], seat=row['seat'], utility=row['utility'],
                             gae=gae.tolist(), mc=mc.tolist(), turn=steps.tolist()))
        maps.append(row['seed'])
    if not features:
        return None
    features = np.concatenate(features); pub = np.concatenate(pub)
    label = np.concatenate(label); turns = np.concatenate(turns)
    value = predict(actor, features, pub, device)
    cash = (pub.astype(np.float64) @ cash_w[0]).astype(np.float64)
    zero = np.zeros_like(value)
    phases = {'all': np.ones(len(turns), bool), 'opening': turns <= 60,
              'middle': (turns > 60) & (turns < 230), 'ending': turns >= 230}
    report = {'states': int(len(label)), 'games': int(len(game_adv)),
              'value': {}, 'cash': {}, 'zero': {},
              'gae_vs_mc': {}}
    for name, mask in phases.items():
        report['value'][name] = metrics(value[mask], label[mask])
        report['cash'][name] = metrics(cash[mask], label[mask])
        report['zero'][name] = metrics(zero[mask], label[mask])
    all_gae = np.concatenate([np.asarray(g['gae']) for g in game_adv])
    all_mc = np.concatenate([np.asarray(g['mc']) for g in game_adv])
    all_turns = np.concatenate([np.asarray(g['turn']) for g in game_adv])
    for name, mask in phases.items():
        diff = all_gae[mask] - all_mc[mask]
        report['gae_vs_mc'][name] = dict(
            N=int(mask.sum()), mean_abs=float(np.mean(np.abs(diff))),
            mean_signed=float(np.mean(diff)),
            corr=corr(all_gae[mask], all_mc[mask]),
            mean_gae=float(np.mean(all_gae[mask])), mean_mc=float(np.mean(all_mc[mask])))
    report['per_map'] = {str(seed): float(np.mean([g['utility'] for g in game_adv
                                                   if g['seed'] == seed])) for seed in sorted(set(maps))}
    return report


def main(a):
    torch.set_num_threads(2)
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
        else:
            actor, _ = actor_from_checkpoint(teacher, style, Path(path), a.device)
        played, timing = play(teacher, actor, reference, matches, a.device, capture='critic')
        results[name] = analyse(played, actor, cash_w, a.device)
        results[name]['seconds'] = timing['elapsed']
        print('calibrated', name, flush=True)
    write(a.out, dict(panel_namespace='terminal-calibration-v1', maps=a.maps,
                      temperature=TEMPERATURE, cash_baseline_sha256=a.critic.name,
                      results=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('critic', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--checkpoint', action='append', required=True,
                   help='name=path; use ZW0 for the frozen reference actor')
    p.add_argument('--maps', type=int, default=8)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
