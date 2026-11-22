"""Fit a worker policy to the final v6 instructions from previously finished tournament games."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from branch_replay import HERE, OUT, configure_engine
from distill_data import local
from distill_fit import fit
from phase_switch import single_thread
from preempt_tree_probe import model

SOURCE = HERE / 'checkpoints/lifeline.py'
FROZEN = 'e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0'


def schedule():
    rows = []
    used = set()
    for round_id in range(1, 9):
        games = [json.loads(line) for line in (OUT / f'{round_id:02d}-qualifying.jsonl').open()]
        for idx, layout in enumerate(sorted({g['layout_sha256'] for g in games})):
            assert layout not in used
            used.add(layout)
            candidates = [g for g in games if g['layout_sha256'] == layout and g['status'] == 'completed']
            if not candidates:
                continue
            row = min(candidates, key=lambda g: (g['players'][(idx + round_id) % 2]['sha256'] != FROZEN, g['seed']))
            seat, = [s for s, p in enumerate(row['players']) if p['sha256'] == FROZEN]
            rows.append((round_id, seat, row))
    return rows


def extract(agent, row, seat, destination):
    agent._ST, agent._FC_HISTORY = {}, {}
    previous, tracker = None, agent.BeliefTracker(agent._Bel, seat)
    groups = [[] for _ in range(4)]
    for t, obs, _, actions, cfg in local(row, seat):
        ps = obs['players']
        est, known = tracker.step(ps, t, obs['carbon'])
        est, known = agent._sense_tree_carbon(tracker.bel, obs, cfg, previous, est, known)
        # Follow the policy router in four full contiguous time windows, including their warmup.
        if any(a <= t < b for a, b in ((0, 59), (85, 122), (162, 201), (245, 284))):
            x = agent._quant(agent.encode(ps, t, seat, est, known, previous))
            lw, lb = agent._v5_root(obs, est, x, previous)
            lw, _ = agent._v6_response(obs, est, cfg, lw, lb)
            lw = agent._selfcredit_adjust(x, lw)
            if t % 2 == 0 and any(a <= t < b for a, b in ((10, 59), (97, 122), (174, 201), (257, 284))):
                ids, roles, z, feat = agent._ds_features(x, ps, seat, previous, lw, None)
                if ids:
                    labels = [agent.WIDX.get(actions[seat].get(uid), 0) for uid in ids]
                    for dst, src in zip(groups, (roles, z, feat, labels)):
                        dst.append(np.asarray(src))
        tracker.set_actions(ps, actions[seat])
        previous = ps
        agent._ST['prev'] = ps
    with destination.open('wb') as output:
        np.savez_compressed(output, **{key: np.concatenate(arr) for key, arr in zip(('role','z','x','y'), groups)})


def measure(records, scale, weights):
    per_map = []
    for meta, d in records:
        roles, z, x, y = (d[k] for k in ('role','z','x','y'))
        naive = z * scale[roles, None]
        adjusted = naive + np.einsum('naf,nf->na', x, weights[roles])
        metrics = []
        for policy in (naive, adjusted):
            logp = policy - np.logaddexp.reduce(policy, axis=1, keepdims=True)
            metrics.append([float(-logp[np.arange(len(y)), y].mean()), float(np.mean(policy.argmax(1) == y))])
        per_map.append(dict(**meta, workers=len(y), calibrated=metrics[0], learned=metrics[1]))
    return per_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    args = parser.parse_args()
    args.cache.mkdir(exist_ok=True)
    single_thread()
    configure_engine()
    m = model(SOURCE, 'own_search_data', False)
    assert m._SEARCH_OK
    records = []
    for round_id, seat, row in schedule():
        layout = row['layout_sha256']
        dest = args.cache / f'{layout}.npz'
        if not dest.exists():
            extract(m, row, seat, dest)
        with np.load(dest, allow_pickle=False) as data:
            records.append((dict(round=round_id, layout=layout, seed=row['seed'], seat=seat,
                                 fold='train' if round_id <= 6 else 'holdout'), dict(data)))
        print(len(records), round_id, row['seed'], seat, flush=True)
    training = [data for meta, data in records if meta['fold'] == 'train']
    roles, z, x, y = [np.concatenate([d[key] for d in training]) for key in ('role','z','x','y')]
    scales, weights = [], []
    for role in (0, 1):
        use = roles == role
        zz = z[use]
        scale = float(max(.05, fit(zz[:,:,None], y[use], np.zeros_like(zz), .0005)[0]))
        scales.append(scale)
        weights.append(fit(x[use], y[use], zz * scale, .014))
    scales, weights = np.asarray(scales), np.asarray(weights)
    results = measure([r for r in records if r[0]['fold'] == 'holdout'], scales, weights)
    model_data = dict(scale=scales.round(7).tolist(), coefficients=weights.round(7).tolist())
    (HERE / 'own-search-model.json').write_text(json.dumps(model_data) + '\n')
    report = dict(teacher=FROZEN, student=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  training=[r for r, _ in records if r['fold'] == 'train'], training_workers=len(y), holdout=results,
                  criterion='Fit on maps from batches 1–6. Compare against a role-calibrated parent policy on disjoint maps from batches 7–8. Each replay contains final post-search instructions for exactly one v6 seat.')
    train_layouts = {r['layout'] for r in report['training']}
    assert train_layouts.isdisjoint({r['layout'] for r in results})
    (OUT / 'own-search-fit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('held', np.mean([r['calibrated'] for r in results], axis=0), np.mean([r['learned'] for r in results], axis=0), flush=True)


if __name__ == '__main__':
    main()
