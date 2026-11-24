"""Compare M (MC state baseline) and G (full-game GAE) advantages.

Both arms use one frozen pre-trained B1 critic (head + independent ZW0 trunk),
the same complete-game collector, optimizer and loss scale as C.  Only the
per-step advantage construction differs.  lambda=1 G must equal M elementwise.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import time

from pilot import Actor, FT3_SHA256, STYLE, engine_fingerprint, load_ft3, np, torch, write
from env import carbon
from policy import public_state_features
from terminal_calibration import full_game_advantages
from terminal_games import HORIZON, TEMPERATURE, group_matches, keyed_seed, play
from terminal_rloo import optimize

CRITIC_SHA256 = 'fceedd2e7944394f2119377ae30997bc86adb8db00272f0dcd14e3548540ec47'
SELECTION = dict(
    primary='equal-weight delta-U over ZW0 / ft3-generic / yan-v6 on the shared 12-map dev panel',
    promote_if='point estimate > 0 and no single-opponent 95% map-bootstrap upper bound < 0',
    confirm_seed=24010000,
    note='pre-registered before the first training run; exploratory, not a statistical promotion rule')


def load_critic(teacher, style, path, device):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CRITIC_SHA256:
        raise ValueError(f'critic hash {digest} does not match the frozen digest')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    critic = Actor(teacher._POLICY, style, residual_critic=True).to(device).eval()
    critic.value_baseline.load_state_dict(saved['value_baseline'])
    critic.value.load_state_dict(saved['value'])
    critic.requires_grad_(False)
    return critic, digest


def state_values(critic, trace, gaps, device):
    features = np.stack([step[0] for step in trace])
    steps = np.arange(len(gaps))
    pub = public_state_features(gaps, steps)
    with torch.no_grad():
        value_features = critic.value_input(critic.spatial(
            torch.as_tensor(features, dtype=torch.float32, device=device)))
        b = critic.predict_value(value_features,
                                 torch.as_tensor(pub, dtype=torch.float32, device=device))
    return b.detach().cpu().numpy().reshape(-1)


def game_advantages(row, trace, critic, device):
    gaps = np.asarray(row['gaps'], dtype=np.float64)
    if len(gaps) != len(trace):
        raise ValueError('gap/trace length mismatch')
    phi = 0.5 * np.tanh(gaps / 1000.)
    b = state_values(critic, trace, gaps, device)
    rewards = np.empty_like(phi)
    rewards[:-1] = phi[1:] - phi[:-1]
    rewards[-1] = row['utility'] - phi[-1]
    mc = row['utility'] - phi - b
    gae = full_game_advantages(rewards, b, .95)
    identity = full_game_advantages(rewards, b, 1.0)
    if float(np.max(np.abs(identity - mc))) > 1e-5:
        raise AssertionError('lambda=1 GAE is not the MC advantage')
    return dict(phi=phi, b=b, rewards=rewards, mc=mc, gae=gae)


def build_dataset(played, critic, arm, device):
    sample_rows, weighted, matches, stats = [], [], [], []
    for row, trace, _ in played:
        advantages = game_advantages(row, trace, critic, device)
        selected = advantages['mc'] if arm == 'M' else advantages['gae']
        if selected.shape != (len(trace),):
            raise ValueError('advantage shape mismatch')
        sample_rows.extend(trace)
        weighted.extend(selected.tolist())
        matches.append(row)
        stats.append(dict(seed=row['seed'], seat=row['seat'], turns=len(trace),
                          utility=row['utility'], mc_mean=float(np.mean(advantages['mc'])),
                          gae_mean=float(np.mean(advantages['gae'])),
                          mc_std=float(np.std(advantages['mc'])), gae_std=float(np.std(advantages['gae'])),
                          mean_abs_gap=float(np.mean(np.abs(advantages['gae'] - advantages['mc'])))))
    arrays = tuple(np.stack(v) for v in zip(*sample_rows, strict=True))
    return (*arrays, np.asarray(weighted, dtype=np.float32)), matches, stats


def main(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    assert style == 122
    actor = Actor(teacher._POLICY, style).to(a.device)
    actor.value.requires_grad_(False)
    base = copy.deepcopy(actor).eval().requires_grad_(False)
    critic, critic_sha = load_critic(teacher, style, a.critic, a.device)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=3e-5)
    protocol = dict(arm=a.arm, seed=a.seed, style=style, source_sha256=FT3_SHA256,
                    critic_sha256=critic_sha, engine_sha256=engine_fingerprint(carbon),
                    decoder='sequential', temperature=TEMPERATURE, steps=HORIZON + 1,
                    group_size=4, groups_per_batch=4, frozen_opponent='ZW0', beta=.5, K=1500,
                    lr=3e-5, ref_KL=.1, entropy=.01, PPO_epsilon=.2, minibatch=512,
                    rollout='complete games; one frozen behavior snapshot per batch',
                    advantage='MC state baseline' if a.arm == 'M' else 'full-game GAE lambda=.95',
                    selection=SELECTION, budget=a.budget)
    write(out / 'protocol.json', protocol)
    logs, results, achieved, total_e = [], [], set(), 0
    marks = sorted({mark for mark in (10000, 50000, a.budget) if mark <= a.budget})
    while total_e < a.budget:
        batch_id = len(logs)
        matches = group_matches(('terminal-mg-train', a.seed), batch_id)
        played, timing = play(teacher, actor, base, matches, a.device, capture='policy')
        data, rows, stats = build_dataset(played, critic, a.arm, a.device)
        total_e += len(data[0])
        reached = [mark for mark in marks if total_e >= mark and mark not in achieved]
        row = optimize(actor, base, optimizer, data, len(played),
                       keyed_seed(a.seed, 'mg-optimizer', batch_id), a.device,
                       audit=(batch_id == 0 or bool(reached)))
        logs.append(dict(**row, E=total_e, L=total_e, batch_E=len(data[0]),
                         behavior_version=batch_id, collection=timing,
                         advantages=dict(arm=a.arm,
                                         mean_mc=float(np.mean([s['mc_mean'] for s in stats])),
                                         mean_gae=float(np.mean([s['gae_mean'] for s in stats])),
                                         mean_abs_gap=float(np.mean([s['mean_abs_gap'] for s in stats])),
                                         zero_groups=0)))
        results.extend(rows)
        achieved.update(reached)
        state = dict(parameters=actor.state_dict(), optimizer=optimizer.state_dict(),
                     E=total_e, arm=a.arm, critic_sha256=critic_sha, protocol=protocol)
        torch.save(state, out / 'latest.pt')
        write(out / 'updates.json', logs)
        write(out / 'episodes.json', results)
        for mark in reached:
            torch.save(state, out / f'checkpoint-{mark}.pt')
            print(a.arm, 'checkpoint', mark, 'E', total_e, 'batches', len(logs), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('critic', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--arm', choices=('M', 'G'), required=True)
    p.add_argument('--budget', type=int, default=50000)
    p.add_argument('--seed', type=int, default=24000000)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
