"""Measure checkpoint drift from the frozen actor on identical held-out histories."""
import argparse
import hashlib
import json
from pathlib import Path

from pilot import Actor, Encoder, Game, STYLE, load_ft3, np, torch
from policy import coordinated_sample, logits, slots

TURNS = {0, 40, 100, 160, 220, 280, 298}


def roots(teacher, replay_file, max_games):
    features, tiles, masks, phases, manifest = [], [], [], [], []
    with replay_file.open() as source:
        for line in source:
            row = json.loads(line)
            if row['status'] != 'completed' or not row.get('actions_and_post_cash'):
                continue
            manifest.append(dict(seed=row['seed'], layout_sha256=row['layout_sha256'],
                                 strategy_sha256=[p['sha256'] for p in row['players']]))
            game = Game(row['seed'], row['episode_steps'])
            trackers = [Encoder(teacher, seat) for seat in (0, 1)]
            for actions, _ in row['actions_and_post_cash']:
                for seat, obs in enumerate(game.observations()):
                    encoded = trackers[seat].encode(obs)
                    if game.turn in TURNS:
                        _, cells, legal = slots(obs, game.config)
                        features.append(encoded)
                        tiles.append(cells)
                        masks.append(legal)
                        phases.append(game.turn)
                    trackers[seat].commit(obs, actions[seat])
                game.step(actions)
            if len(manifest) == max_games:
                break
    if not manifest:
        raise ValueError('No completed, action-recorded arena games in replay file')
    return [np.stack(v) for v in (features, tiles, masks)], np.asarray(phases), manifest


def role_stats(kl, changed, active, slots_for_role):
    valid = active[:, slots_for_role]
    return dict(slot_count=int(valid.sum()),
                KL_per_active_slot=float(kl[:, slots_for_role][valid].mean()) if valid.any() else None,
                greedy_flip_rate=float(changed[:, slots_for_role][valid].mean()) if valid.any() else None)


@torch.inference_mode()
def compare(source, training, replay_file, max_games):
    teacher = load_ft3(source)
    style = teacher._POLICY.sid(STYLE)
    model = Actor(teacher._POLICY, style).eval()
    (x, cells, mask), phase, manifest = roots(teacher, replay_file, max_games)
    observations = [torch.as_tensor(v) for v in (x, cells, mask)]
    temperature = json.loads((training / 'calibration.json').read_text())['temperature']
    fresh, _ = logits(model, *observations, temperature)
    _, conditional = coordinated_sample(fresh, np.random.default_rng(751201), cells, mask)
    conditional = torch.as_tensor(conditional)
    target = fresh.masked_fill(~conditional, -1.e9)
    target_lp = target.log_softmax(-1)
    active = conditional.sum(-1).numpy() > 1
    report = dict(replay_sha256=hashlib.sha256(replay_file.read_bytes()).hexdigest(),
                  games=manifest, roots=len(x), temperature=temperature,
                  conditional_prefix='sampled once from frozen actor on the replay observation', checkpoints={})
    for mark in (10000, 100000, 200000):
        checkpoint = training / f'checkpoint-{mark}.pt'
        if not checkpoint.exists():
            continue
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['parameters'])
        candidate, _ = logits(model, *observations, temperature)
        candidate_lp = candidate.masked_fill(~conditional, -1.e9).log_softmax(-1)
        kl = (target_lp.exp() * (target_lp - candidate_lp)).sum(-1).numpy()
        changed = (target_lp.argmax(-1) != candidate_lp.argmax(-1)).numpy()
        report['checkpoints'][mark] = dict(
            mean_team_KL=float(kl.sum(1).mean()),
            bases=role_stats(kl, changed, active, slice(0, 2)),
            workers=role_stats(kl, changed, active, slice(2, None)),
            team_KL_by_turn={int(turn): float(kl[phase==turn].sum(1).mean()) for turn in np.unique(phase)})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'training', 'replay', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--games', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = compare(args.source, args.training, args.replay, args.games)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
