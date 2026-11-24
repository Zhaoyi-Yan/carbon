"""Compare M (MC state baseline) and G (full-game GAE) advantages.

Both arms use one frozen pre-trained B1 critic (head + independent ZW0 trunk),
the same complete-game collector, optimizer and loss scale as C.  Only the
per-step advantage construction differs.  lambda=1 G must equal M elementwise.

`--wide` switches the batch allocation from 2 layouts x 2 seats x 4 replicates
to 8 layouts x 2 seats x 1 replicate, keeping 16 games per batch.
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
from terminal_rloo import archive_trace, optimize

CRITIC_SHA256 = 'fceedd2e7944394f2119377ae30997bc86adb8db00272f0dcd14e3548540ec47'
GATES = dict(
    current=dict(arm='M', seed=24000000, note='seed 1 macro dU > 0 and no opponent upper bound < 0'),
    confirm=dict(seed=24010000, note='two seeds, no best-seed selection'),
    wide=dict(rule=('M-wide - same-seed M macro dU > 0; M-wide - ZW0 macro dU > 0; '
                    'no per-opponent M-wide - M 95% upper bound < 0'),
              seed=24010000, note='only after the seed-1 wide screen passes'),
    selection=dict(primary='equal-weight delta-U over ZW0 / ft3-generic / yan-v6', steps=3))


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


def critic_digest(actor):
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


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
        raise AssertionError(f'lambda=1 GAE is not the MC advantage: '
                             f'{float(np.max(np.abs(identity - mc)))} seed={row["seed"]} seat={row["seat"]}')
    return dict(phi=phi, b=b, rewards=rewards, mc=mc, gae=gae)


def advantage_stats(values, turns):
    values, turns = np.asarray(values), np.asarray(turns)
    phases = {'all': np.ones(len(values), bool), 'opening': turns <= 60,
              'middle': (turns > 60) & (turns < 230), 'ending': turns >= 230}
    return {name: dict(N=int(mask.sum()), mean=float(values[mask].mean()),
                       std=float(values[mask].std()), rms=float(np.sqrt(np.mean(values[mask] ** 2))),
                       mean_abs=float(np.mean(np.abs(values[mask]))),
                       positive=float(np.mean(values[mask] > 0)))
            for name, mask in phases.items() if mask.any()}


def build_dataset(played, critic, arm, device):
    sample_rows, weighted, matches, stats = [], [], [], []
    selected_values, selected_turns = [], []
    mc_values, gae_values = [], []
    baseline_seconds = 0.
    for row, trace, _ in played:
        started = time.perf_counter()
        advantages = game_advantages(row, trace, critic, device)
        baseline_seconds += time.perf_counter() - started
        selected = advantages['mc'] if arm == 'M' else advantages['gae']
        if selected.shape != (len(trace),):
            raise ValueError('advantage shape mismatch')
        sample_rows.extend(trace)
        weighted.extend(selected.tolist())
        matches.append(row)
        turns = np.arange(len(trace))
        selected_values.extend(selected.tolist()); selected_turns.extend(turns.tolist())
        mc_values.extend(advantages['mc'].tolist()); gae_values.extend(advantages['gae'].tolist())
        stats.append(dict(seed=row['seed'], seat=row['seat'], turns=len(trace),
                          utility=row['utility'],
                          mean_abs_gap=float(np.mean(np.abs(advantages['gae'] - advantages['mc'])))))
    arrays = tuple(np.stack(v) for v in zip(*sample_rows, strict=True))
    summary = dict(baseline_seconds=baseline_seconds,
                   selected=advantage_stats(selected_values, selected_turns),
                   mc=advantage_stats(mc_values, selected_turns),
                   gae=advantage_stats(gae_values, selected_turns),
                   mean_abs_gap=float(np.mean([s['mean_abs_gap'] for s in stats])))
    return (*arrays, np.asarray(weighted, dtype=np.float32)), matches, summary


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
    critic_hash = critic_digest(critic)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=3e-5)
    layout = 'wide' if a.wide else 'replicated'
    arm = f'{a.arm}-wide' if a.wide else a.arm
    protocol = dict(arm=arm, layout=layout, seed=a.seed, style=style, source_sha256=FT3_SHA256,
                    critic_sha256=critic_sha, critic_param_sha256=critic_hash,
                    engine_sha256=engine_fingerprint(carbon), decoder='sequential',
                    temperature=TEMPERATURE, steps=HORIZON + 1,
                    group_size=1 if a.wide else 4, groups_per_batch=16,
                    maps_per_batch=8 if a.wide else 2,
                    frozen_opponent='ZW0', beta=.5, K=1500, lr=3e-5, ref_KL=.1,
                    entropy=.01, PPO_epsilon=.2, minibatch=512,
                    rollout='complete games; one frozen behavior snapshot per batch',
                    advantage='MC state baseline' if a.arm == 'M' else 'full-game GAE lambda=.95',
                    gates=GATES, budget=a.budget, batches=a.batches)
    write(out / 'protocol.json', protocol)
    logs, results, achieved, total_e = [], [], set(), 0
    marks = sorted({mark for mark in (10000, 50000, a.budget) if mark <= a.budget})
    while total_e < a.budget and (not a.batches or len(logs) < a.batches):
        batch_id = len(logs)
        matches = group_matches(('terminal-mg-train', a.seed), batch_id, wide=a.wide)
        played, timing = play(teacher, actor, base, matches, a.device, capture='policy')
        data, rows, stats = build_dataset(played, critic, a.arm, a.device)
        total_e += len(data[0])
        reached = [mark for mark in marks if total_e >= mark and mark not in achieved]
        row = optimize(actor, base, optimizer, data, len(played),
                       keyed_seed(a.seed, 'mg-optimizer', batch_id), a.device,
                       audit=(batch_id == 0 or bool(reached)))
        trace_path = out / f'batch-{batch_id:03d}.jsonl.gz'
        trace_digest = archive_trace(trace_path, played)
        np.save(out / f'advantages-{batch_id:03d}.npy', data[-1])
        logs.append(dict(**row, E=total_e, L=total_e, batch_E=len(data[0]),
                         behavior_version=batch_id, collection=timing,
                         baseline_seconds=stats['baseline_seconds'],
                         advantage_stats=dict(selected=stats['selected'], mc=stats['mc'],
                                              gae=stats['gae'], mean_abs_gap=stats['mean_abs_gap']),
                         trace_sha256=trace_digest))
        results.extend(rows)
        achieved.update(reached)
        state = dict(parameters=actor.state_dict(), optimizer=optimizer.state_dict(),
                     E=total_e, arm=arm, layout=layout, critic_sha256=critic_sha,
                     critic_param_sha256=critic_hash, protocol=protocol)
        torch.save(state, out / 'latest.pt')
        write(out / 'updates.json', logs)
        write(out / 'episodes.json', results)
        for mark in reached:
            torch.save(state, out / f'checkpoint-{mark}.pt')
            print(arm, 'checkpoint', mark, 'E', total_e, 'batches', len(logs), flush=True)
    torch.save(state, out / 'final.pt')
    if critic_digest(critic) != critic_hash:
        raise AssertionError('The frozen critic changed during training')
    print(arm, 'done E', total_e, 'batches', len(logs), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('critic', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--arm', choices=('M', 'G'), required=True)
    p.add_argument('--wide', action='store_true', help='8 layouts x 2 seats x 1 replicate')
    p.add_argument('--budget', type=int, default=50000)
    p.add_argument('--batches', type=int, default=0,
                   help='Stop after this many complete batches (0 = use --budget)')
    p.add_argument('--seed', type=int, default=24000000)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
