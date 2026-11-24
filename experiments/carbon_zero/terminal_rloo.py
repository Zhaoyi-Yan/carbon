"""Optimize complete-game RLOO on grouped, frozen-playout trajectories."""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

from pilot import Actor, FT3_SHA256, STYLE, engine_fingerprint, load_ft3, np, torch, write
from env import carbon
from policy import action_logits
from ppo import joint_logprob
from terminal_games import HORIZON, TEMPERATURE, group_matches, keyed_seed, play

ANCHOR = 0.1


def leave_one_out(returns):
    rewards = np.asarray(returns, dtype=np.float64)
    if rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError('RLOO needs at least two independent trajectories per group')
    return (len(rewards) * rewards - rewards.sum()) / (len(rewards) - 1)


def score(model, features, cells, legal):
    spatial = model.spatial(features)
    return action_logits(model.workers(spatial), model.bases(spatial), cells, legal, TEMPERATURE)


def ratio_loss(scores, action, old, adv, rows_scale):
    delta = joint_logprob(scores, action) - old
    ratio = delta.exp()
    objective = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).sum() * rows_scale
    return objective, (ratio - 1. - delta).mean(), ((ratio - 1.).abs() > .2).float().mean()


def penalties(scores, target_logp, legal, rows_scale):
    logp = scores.log_softmax(-1)
    active = legal.sum(-1) > 1
    slot_entropy = -(logp.exp() * logp).sum(-1)
    entropy = ((slot_entropy * active).sum(-1) / active.sum(-1).clamp_min(1)).sum() * rows_scale
    kl = (target_logp.exp() * (target_logp - logp)).sum() * rows_scale
    return entropy, kl


def grad_breakdown(losses, trunk):
    all_grads = [torch.autograd.grad(loss, trunk, retain_graph=True, allow_unused=True)
                 for loss in losses.values()]
    vectors = [torch.cat([g.reshape(-1) for g in grads if g is not None]) for grads in all_grads]
    names = list(losses)
    result = {name: float(v.norm().detach().cpu()) for name, v in zip(names, vectors, strict=True)}
    if 'value' in names:
        a, b = (vectors[names.index(label)] for label in ('policy', 'value'))
        result['policy_value_cosine'] = float(torch.nn.functional.cosine_similarity(a, b, dim=0).detach().cpu())
    return result


def prepare(played):
    sample_rows, weighted, matches, groups = [], [], [], []
    for group, items in itertools.groupby(played, key=lambda item: item[0]['group']):
        items = list(items)
        record = [row for row, _, _ in items]
        advantages = leave_one_out([row['utility'] for row in record])
        for advantage, (row, trace, _) in zip(advantages, items, strict=True):
            row['advantage'] = float(advantage)
            matches.append(row)
            sample_rows.extend(trace)
            weighted.extend([advantage] * len(trace))
        returns = [row['utility'] for row in record]
        groups.append(dict(group=group, layout_seed=record[0]['seed'], seat=record[0]['seat'],
                           game_count=len(record), terminal_std=float(np.std(returns)),
                           zero_advantage=bool(np.all(advantages == 0)),
                           advantages=advantages.tolist()))
    arrays = tuple(np.stack(v) for v in zip(*sample_rows, strict=True))
    return (*arrays, np.asarray(weighted, dtype=np.float32)), matches, groups


def optimize(actor, reference, optimizer, dataset, game_count, seed, device, *, audit=False):
    n = len(dataset[0])
    order = np.random.default_rng(seed).permutation(n)
    groups = [order[start:start + 512] for start in range(0, n, 512)]
    rows_scale = len(groups) / (game_count * HORIZON)

    def batch(ids):
        return tuple(torch.as_tensor(v[ids], device=device) for v in dataset)

    with torch.no_grad():
        x, c, m, a, old, _ = batch(order[:96])
        current = joint_logprob(score(actor, x, c, m), a)
        parity = float((current - old).abs().max().cpu())
    if parity > 2e-4:
        raise ValueError(f'Replayed conditional log-probability mismatch {parity}')

    started = time.perf_counter()
    accepted_updates = []
    diagnostic = None
    trunk = [p for key, p in actor.named_parameters()
             if key.startswith(('stem.', 'blocks.', 'style')) and p.requires_grad]
    for ids in groups:
        optimizer.zero_grad(set_to_none=True)
        measured = []
        for start in range(0, len(ids), 256):
            x, c, m, a, old, adv = batch(ids[start:start + 256])
            with torch.no_grad():
                ref_logp = score(reference, x, c, m).log_softmax(-1)
            z = score(actor, x, c, m)
            policy_loss, old_kl, clipped = ratio_loss(z, a, old, adv, rows_scale)
            entropy, anchor = penalties(z, ref_logp, m, rows_scale)
            if audit and diagnostic is None and trunk:
                diagnostic = grad_breakdown(
                    dict(policy=policy_loss, entropy=-.01 * entropy, reference=ANCHOR * anchor), trunk)
            (policy_loss - .01 * entropy + ANCHOR * anchor).backward()
            measured.append(torch.stack((policy_loss, old_kl, clipped, entropy, anchor)).detach())
        if any(p.grad is not None for p in actor.value.parameters()):
            raise AssertionError('RLOO unexpectedly backpropagated into the critic')
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
        optimizer.step()
        accepted_updates.append([*torch.stack(measured).mean(0).cpu().tolist(), float(norm)])
    averages = np.mean(accepted_updates, axis=0)
    result = dict(zip(('policy', 'old_KL', 'clip_rate', 'entropy_scaled', 'reference_KL_scaled',
                       'gradient_norm'), map(float, averages), strict=True))
    return dict(**result, logprob_error=parity, optimization_updates=len(groups),
                optimization_seconds=time.perf_counter() - started, gradient_diagnostic=diagnostic)


def archive_trace(path, played):
    """Store complete episode commands and the exact conditional probability inputs."""
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=3) as out:
        for row, sampled, commands in played:
            trace = []
            for (_, cells, masks, picks, old), (names, sent) in zip(sampled, commands, strict=True):
                trace.append([names, cells.tolist(), np.packbits(masks).tobytes().hex(),
                              picks.tolist(), float(old), sent])
            out.write(json.dumps(dict(game=row, trace=trace), separators=(',', ':')) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    out = a.out
    out.mkdir(parents=True, exist_ok=a.resume)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    assert style == 122
    actor = Actor(teacher._POLICY, style).to(a.device)
    base = copy.deepcopy(actor).eval().requires_grad_(False)
    if a.heads_only:
        for name, p in actor.named_parameters():
            p.requires_grad_(name.startswith(('workers.', 'bases.')))
    else:
        actor.value.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=3e-5)
    protocol = dict(arm='D' if a.heads_only else 'C', seed=a.seed, style=style,
                    source_sha256=FT3_SHA256, engine_sha256=engine_fingerprint(carbon),
                    decoder='sequential', temperature=TEMPERATURE, steps=HORIZON + 1,
                    group_size=4, groups_per_batch=4, frozen_opponent='ZW0', beta=.5, K=1500,
                    lr=3e-5, ref_KL=ANCHOR, entropy=.01, PPO_epsilon=.2, minibatch=512,
                    reference_prefix='learner sampled prefix', optimizer='AdamW', device=a.device,
                    rollout='complete; every group uses one frozen behavior snapshot')
    if a.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Resume protocol differs from the saved run')
        state = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
        actor.load_state_dict(state['parameters'])
        optimizer.load_state_dict(state['optimizer'])
        logs = state['logs']
        results = state['episodes']
        achieved = set(state['marks'])
        total_e = state['E']
    else:
        write(out / 'protocol.json', protocol)
        logs, results, achieved, total_e = [], [], set(), 0
    marks = sorted({mark for mark in (10000, 50000, 100000, a.budget) if mark <= a.budget})
    while total_e < a.budget:
        batch_id = len(logs)
        matches = group_matches(('terminal-rloo-train', a.seed), batch_id)
        played, timing = play(teacher, actor, base, matches, a.device, capture='policy')
        data, rows, groups = prepare(played)
        total_e += len(data[0])
        reached = [mark for mark in marks if total_e >= mark and mark not in achieved]
        row = optimize(actor, base, optimizer, data, len(played),
                       keyed_seed(a.seed, 'rloo-optimizer', batch_id), a.device,
                       audit=(batch_id == 0 or bool(reached)))
        trace_path = out / f'batch-{batch_id:03d}.jsonl.gz'
        trace_digest = archive_trace(trace_path, played)
        for match in rows:
            match['behavior_version'] = batch_id
        results.extend(rows)
        logs.append(dict(**row, E=total_e, L=total_e, behavior_version=batch_id,
                         groups=groups, collection=timing, trace_sha256=trace_digest,
                         batch_E=len(data[0]), trace_file=trace_path.name))
        achieved.update(reached)
        state = dict(parameters=actor.state_dict(), optimizer=optimizer.state_dict(), E=total_e,
                     next_batch=batch_id + 1, marks=sorted(achieved), logs=logs, episodes=results)
        tmp = out / 'latest.tmp'
        torch.save(state, tmp)
        tmp.replace(out / 'latest.pt')
        write(out / 'updates.json', logs)
        write(out / 'episodes.json', results)
        for mark in reached:
            tmp = out / 'checkpoint.tmp'
            torch.save(state, tmp)
            tmp.replace(out / f'checkpoint-{mark}.pt')
            print('RLOO checkpoint', mark, 'E', total_e, 'batches', len(logs), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--budget', type=int, default=100000)
    p.add_argument('--seed', type=int, default=24000000)
    p.add_argument('--heads-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
