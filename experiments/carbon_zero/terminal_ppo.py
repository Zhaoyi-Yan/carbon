"""Continue the anchored PPO control or start it with a calibrated critic."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from pilot import Actor, FT3_SHA256, STYLE, ROOT, engine_fingerprint, load_ft3, np, torch, write
from env import carbon
from policy import logits
from ppo import Collector, joint_logprob, update
from terminal_games import TEMPERATURE
from terminal_rloo import grad_breakdown

OLD_RUN = ROOT / 'analysis/carbon-zero/anchored-10000'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient_audit(model, base, trajectory, device):
    """Measure loss components on one frozen pre-update minibatch."""
    if len(trajectory) == 8:
        x, c, m, a, old, adv, target, pubs = trajectory
        pub_t = torch.as_tensor(pubs[:32], device=device)
    else:
        x, c, m, a, old, adv, target = trajectory
        pub_t = None
    ia = (adv - adv.mean()) / max(adv.std(), 1e-6)
    args = [torch.as_tensor(v[:32], device=device) for v in (x, c, m)]
    z, value = logits(model, *args, TEMPERATURE, pub=pub_t)
    with torch.no_grad():
        ref, _ = logits(base, *args, TEMPERATURE)
        ref = ref.log_softmax(-1)
    lp = z.log_softmax(-1)
    active = args[2].sum(-1) > 1
    entropy = -(lp.exp() * lp).sum(-1)
    entropy = ((entropy * active).sum(-1) / active.sum(-1).clamp_min(1)).mean()
    actions = torch.as_tensor(a[:32], device=device)
    advantage = torch.as_tensor(ia[:32], device=device)
    ratio = (joint_logprob(z, actions) - torch.as_tensor(old[:32], device=device)).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
    y = torch.as_tensor(target[:32], dtype=torch.float32, device=device)
    value_loss = .5 * (y - value).square().mean()
    anchor = .1 * (ref.exp() * (ref - lp)).sum((-1, -2)).mean()
    trunk = [p for name, p in model.named_parameters()
             if name.startswith(('stem.', 'blocks.', 'style'))]
    return grad_breakdown(dict(policy=policy, value=value_loss, entropy=-.01 * entropy, reference=anchor), trunk)


def main(a):
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    assert style == 122
    residual_critic = False
    calibrated = None
    if a.critic:
        calibrated = json.loads((a.critic / 'critic.json').read_text())
        if not calibrated['qualified'] or calibrated['source_sha256'] != FT3_SHA256:
            raise ValueError('The frozen critic has not passed held-out calibration')
        residual_critic = bool(calibrated.get('residual_critic', False))
    learner = Actor(
        teacher._POLICY, style,
        residual_critic=residual_critic,
        detach_value_trunk=a.detach_value_trunk,
    ).to(a.device)
    anchor = Actor(teacher._POLICY, style).to(a.device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(learner.parameters(), lr=3e-5)
    rng = np.random.default_rng(a.seed + 133)
    collector = Collector(teacher, 16, a.seed, a.device, coordinated=True)
    pretrain_e = 0
    if a.control:
        if a.seed != 24000000:
            raise ValueError('The archived 10k checkpoint used seed 24000000')
        expected = json.loads((OLD_RUN / 'artifact-hashes.json').read_text())['checkpoint-10000.pt']['sha256']
        if digest(a.control) != expected:
            raise ValueError('Anchored PPO checkpoint SHA-256 mismatch')
        saved = torch.load(a.control, weights_only=True, map_location='cpu')
        steps = saved['E']
        learner.load_state_dict(saved['parameters'])
        opt.load_state_dict(saved['optimizer'])
        collector.recover(saved['collector'])
        rng.bit_generator.state = saved['optimizer_rng']
        updates = json.loads((OLD_RUN / 'updates.json').read_text())
        if updates[-1]['E'] != steps:
            raise ValueError('Anchored PPO training logs mismatch')
        initial = dict(experiment='archived-anchored-10000', sha256=expected)
    else:
        weight = a.critic / 'critic.pt'
        loaded = torch.load(weight, weights_only=True, map_location='cpu')
        if residual_critic:
            learner.value_baseline.load_state_dict(loaded['value_baseline'])
            learner.value.load_state_dict(loaded['value'])
        else:
            learner.value.load_state_dict(loaded)
        pretrain_e = calibrated['train_E'] + calibrated['heldout_E']
        steps, updates = 0, []
        initial = dict(
            experiment='frozen-policy-MC-residual-critic' if residual_critic else 'frozen-policy-MC-critic',
            sha256=digest(weight),
        )
    arm = 'A' if a.control else ('B2' if a.detach_value_trunk else 'B1')
    output = a.out
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', dict(
        arm=arm, source_sha256=FT3_SHA256, engine_sha256=engine_fingerprint(carbon),
        seed=a.seed, device=a.device, style=style, temperature=TEMPERATURE,
        opponent='ZW0', decoder='sequential', initialization=initial, pretrain_E=pretrain_e,
        residual_critic=residual_critic, detach_value_trunk=a.detach_value_trunk,
        lr=3e-5, ref_KL=.1, beta=.5, K=1500, rollout=64, lambda_GAE=.95,
        actor_warmup_updates=2, environments=16))
    markers = [n for n in sorted({10000, 50000, 100000, a.budget}) if steps < n <= a.budget]
    for target in markers:
        while steps < target:
            batch_turns = min(64, (target - steps + 15) // 16)
            data = collector.collect(learner, anchor, TEMPERATURE, batch_turns)
            steps += len(data[0])
            audited = len(updates) in (2, 10) or steps >= target
            audit = gradient_audit(learner, anchor, data, a.device) if audited else None
            first = time.perf_counter()
            row = update(learner, opt, data, a.device, TEMPERATURE, rng,
                         warmup=(len(updates) < 2), anchor=anchor, anchor_weight=.1)
            row.update(E=steps, L=steps, total_E=steps + pretrain_e,
                       optimization_seconds=time.perf_counter() - first,
                       cumulative=dict(collector.timing), forwards=collector.calls,
                       finished=len(collector.episodes), gradient_diagnostic=audit)
            updates.append(row)
            write(output / 'updates.json', updates)
        checkpoint = dict(parameters=learner.state_dict(), optimizer=opt.state_dict(),
                          residual_critic=residual_critic, detach_value_trunk=a.detach_value_trunk,
                          E=steps, total_E=steps+pretrain_e,
                          next_seed=collector.next_seed, optimizer_rng=rng.bit_generator.state,
                          collector=collector.snapshot())
        torch.save(checkpoint, output / 'checkpoint.tmp')
        (output / 'checkpoint.tmp').replace(output / f'checkpoint-{target}.pt')
        write(output / 'episodes.json', collector.episodes)
        print(arm, 'checkpoint', target, 'E', steps, 'total E', steps+pretrain_e, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('out', type=Path)
    warm = p.add_mutually_exclusive_group(required=True)
    warm.add_argument('--control', type=Path)
    warm.add_argument('--critic', type=Path)
    p.add_argument('--detach-value-trunk', action='store_true')
    p.add_argument('--budget', type=int, default=50000)
    p.add_argument('--seed', type=int, default=24000000)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
