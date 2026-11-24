"""Compare held-out conditional action distributions using multiple reference prefixes."""
import argparse
import hashlib
from pathlib import Path

from pilot import Actor, STYLE, load_ft3, np, torch, write
from policy import coordinated_sample
from terminal_games import Match, keyed_seed, play
from terminal_rloo import score

TICKS = (20, 60, 100, 150, 200, 250, 298)


def stages(turn):
    return 'opening' if turn <= 60 else 'middle' if turn < 230 else 'ending'


def roots(teacher, reference, device):
    matches = []
    for layout in range(8):
        seed = keyed_seed('terminal-drift-v1', layout) & 0xffffffff
        for seat in (0, 1):
            matches.append(Match(seed, seat, keyed_seed(seed, seat, 'own'),
                                 keyed_seed(seed, seat, 'other'), layout))
    played, costs = play(teacher, reference, reference, matches, device, capture='policy')
    features, cells, legal, manifest = [], [], [], []
    for game, trace, sent in played:
        # Replay the legal mask at the same positions without earlier reservations.
        # The saved effective masks supply slots, but the unmasked moves (all 5
        # for each live worker) are needed when resampling reference prefixes.
        for turn in TICKS:
            if turn >= len(trace):
                continue
            x, c, mask, _, _ = trace[turn]
            legal_mask = mask.copy()
            names = sent[turn][0]
            legal_mask[2:len(names)] = True
            features.append(x)
            cells.append(c)
            legal.append(legal_mask)
            manifest.append(dict(seed=game['seed'], seat=game['seat'], turn=turn))
    return tuple(np.stack(v) for v in (features, cells, legal)), manifest, costs


@torch.inference_mode()
def compare(a):
    torch.set_num_threads(2)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    reference = Actor(teacher._POLICY, style).to(a.device).eval()
    arrays, manifest, costs = roots(teacher, reference, a.device)
    fixed = [torch.as_tensor(v, device=a.device) for v in arrays]
    ref_scores = score(reference, *fixed)
    reference_logp = []
    active = []
    for prefix in range(4):
        rng = np.random.default_rng(keyed_seed('terminal-drift-prefix', prefix))
        _, mask = coordinated_sample(ref_scores, rng, arrays[1], arrays[2])
        mask = torch.as_tensor(mask, device=a.device)
        reference_logp.append((mask, ref_scores.masked_fill(~mask, -1.e9).log_softmax(-1)))
        active.append(mask.sum(-1).cpu().numpy() > 1)
    result = dict(prefixes=4, root_manifest=manifest, collection_costs=costs, candidates={})
    phase = np.asarray([stages(item['turn']) for item in manifest])
    for item in a.candidate:
        name, file = item.split('=', 1)
        model = Actor(teacher._POLICY, style).to(a.device).eval()
        model.load_state_dict(torch.load(file, map_location='cpu', weights_only=True)['parameters'])
        raw = score(model, *fixed)
        team_kl, changes = [], []
        for masks, ref in reference_logp:
            ahead = raw.masked_fill(~masks, -1.e9).log_softmax(-1)
            team_kl.append((ref.exp() * (ref - ahead)).sum(-1).cpu().numpy())
            changes.append((ref.argmax(-1) != ahead.argmax(-1)).cpu().numpy())
        kl = np.stack(team_kl)
        total = kl.sum(-1)
        switches = np.stack(changes)
        valid = np.stack(active)
        result['candidates'][name] = dict(
            sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest(),
            team_KL_mean=float(total.mean()), team_KL_p90=float(np.quantile(total, .9)),
            team_KL_p99=float(np.quantile(total, .99)),
            phase_KL={p: float(total[:, phase == p].mean()) for p in sorted(set(phase))},
            worker_flip=float(switches[:, :, 2:][valid[:, :, 2:]].mean()),
            base_flip=float(switches[:, :, :2][valid[:, :, :2]].mean()))
    write(a.out, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--candidate', action='append', default=[], required=True)
    p.add_argument('--device', default='cpu')
    compare(p.parse_args())
