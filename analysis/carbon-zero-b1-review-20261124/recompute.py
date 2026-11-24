"""Recompute B1 evidence from the afcfe81 snapshot; no policy training.

Usage: python recompute.py /path/to/carbon-b1-checkout [output.json]
Requires NumPy. Input JSONs and heldout.npz are read-only.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

SOURCE_COMMIT = 'afcfe81712676dcc0d680e47881db6ecc71ea55b'
BOOTSTRAP_SEED = 20261124
REPLICATES = 100000
RIVALS = ('ZW0', 'ft3-generic-no-search', 'yan-v6-no-search')
PAIRS = (('B1_10k', 'ZW0'), ('B1_50k', 'ZW0'), ('A10', 'ZW0'),
         ('A50', 'ZW0'), ('C50', 'ZW0'), ('C100', 'ZW0'),
         ('B1_10k', 'A10'), ('B1_50k', 'A50'), ('B1_50k', 'B1_10k'))


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(values), size=(REPLICATES, len(values)))
    distribution = values[draws].mean(axis=1)
    sd = float(values.std(ddof=1))
    return {
        'mean': float(values.mean()),
        'percentile_95': np.quantile(distribution, [.025, .975]).tolist(),
        'cluster_count': len(values), 'cluster_sd': sd,
        'normal_95_halfwidth_if_same_sd': {str(n): 1.96 * sd / np.sqrt(n) for n in (25, 40)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('output', type=Path, nargs='?', default=Path('statistics.json'))
    args = parser.parse_args()
    repo = args.repository.resolve()
    folder = repo / 'analysis/carbon-zero-terminal/b1'
    inputs = sorted((folder / 'evaluation').glob('*.json'))
    inputs.extend([folder / 'critic/critic.json', folder / 'critic/heldout.npz'])
    # Do not silently label changed data as evidence from the pinned snapshot.
    for path in inputs:
        relative = path.relative_to(repo).as_posix()
        pinned = subprocess.check_output(['git', '-C', str(repo), 'show', f'{SOURCE_COMMIT}:{relative}'])
        if path.read_bytes() != pinned:
            raise ValueError(f'Input differs from pinned source {SOURCE_COMMIT}: {relative}')
    panels = {p.stem: json.loads(p.read_text()) for p in inputs if p.parent.name == 'evaluation' and p.stem != 'protocol'}
    result = {
        'source_commit': SOURCE_COMMIT,
        'source_inputs_verified_against_git': True,
        'input_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'method': {
            'numpy_version': np.__version__, 'python_version': platform.python_version(),
            'rng': 'numpy.default_rng / PCG64, reset for each same-sized bootstrap',
            'seed': BOOTSTRAP_SEED, 'replicates': REPLICATES,
            'interval': 'two-sided unadjusted 95% percentile bootstrap',
            'evaluation_unit': 'map; average two seats within each map before resampling',
            'critic_unit': 'map; average eight independent games (two seats x four streams) within each map',
            'precision_projection': '1.96 * observed cluster SD / sqrt(n maps), planning heuristic only; not an attained interval or power guarantee',
            'limitations': ['One trained B1 seed; intervals condition on this checkpoint.', 'Exploratory multiple comparisons; no multiplicity correction.', 'Current opponent panels use different maps, so cross-opponent differences are not within-map paired.', 'No claim of equivalence or non-inferiority from an interval crossing zero.'],
        }, 'panel_geometry': {}, 'comparisons': {}, 'critic': {},
    }
    eval_maps = []
    for rival in RIVALS:
        games = panels['ZW0']['rivals'][rival]['games']
        seeds = sorted({g['seed'] for g in games})
        eval_maps.extend(seeds)
        result['panel_geometry'][rival] = {'games': len(games), 'maps': len(seeds), 'seeds': seeds}
    result['panel_geometry']['total_distinct_maps'] = len(set(eval_maps))
    for a, b in PAIRS:
        pair_results = {}
        for rival in RIVALS:
            ga = {(g['seed'], g['seat']): g for g in panels[a]['rivals'][rival]['games']}
            gb = {(g['seed'], g['seat']): g for g in panels[b]['rivals'][rival]['games']}
            assert ga.keys() == gb.keys()
            for key in ga:
                for field in ('learner_rng', 'opponent_rng', 'layout_sha256'):
                    assert ga[key][field] == gb[key][field], (a, b, rival, key, field)
            seeds = sorted({k[0] for k in ga})
            item = {}
            for metric in ('margin', 'utility', 'win_score'):
                transform = (lambda g: float(g['margin'] > 0) + .5 * float(g['margin'] == 0)) if metric == 'win_score' else lambda g: g[metric]
                values = [float(np.mean([transform(ga[(s,t)]) - transform(gb[(s,t)]) for t in (0,1)])) for s in seeds]
                item[metric] = bootstrap(values)
            pair_results[rival] = item
        result['comparisons'][a + '-' + b] = pair_results
    critic = json.loads((folder / 'critic/critic.json').read_text())
    held = np.load(folder / 'critic/heldout.npz')
    game_rows = critic['heldout_games']
    seeds = sorted({g['seed'] for g in game_rows})
    assert len(game_rows) == len(np.unique(held['group']))
    assert all(g['turns'] == 299 for g in game_rows)
    # All heldout games are full-length, so equal-game weighting equals pooled frame weighting per phase.
    result['critic']['geometry'] = {'training_games': len(critic['training_games']), 'training_maps': len({g['seed'] for g in critic['training_games']}), 'heldout_games': len(game_rows), 'heldout_maps': len(seeds), 'heldout_frames': len(held['turn'])}
    error_delta = (held['cash'] - held['labels']) ** 2 - (held['value'] - held['labels']) ** 2
    phases = {'all': np.ones(len(held['turn']), dtype=bool), 'opening': held['turn'] <= 60, 'middle': (held['turn'] > 60) & (held['turn'] < 230), 'ending': held['turn'] >= 230}
    for phase, mask in phases.items():
        per_game = [float(error_delta[mask & (held['group'] == i)].mean()) for i in range(len(game_rows))]
        per_map = [float(np.mean([per_game[i] for i,g in enumerate(game_rows) if g['seed'] == seed])) for seed in seeds]
        stats = bootstrap(per_map)
        # Provided solely for auditing the original README's interval convention.
        stats['game_bootstrap_for_diagnostic_only'] = bootstrap(per_game)['percentile_95']
        result['critic'][phase] = stats
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Wrote {args.output}; {len(inputs)} hashed inputs; {REPLICATES} bootstrap replicates, seed {BOOTSTRAP_SEED}.')
    for pair in ('B1_10k-ZW0', 'B1_50k-ZW0', 'B1_50k-A50'):
        print(pair)
        for rival, metrics in result['comparisons'][pair].items():
            u = metrics['utility']; margin = metrics['margin']
            print(f'  {rival}: delta_U {u["mean"]:+.6f}, CI {u["percentile_95"]}; delta_margin {margin["mean"]:+.3f}; halfwidth25/40 {u["normal_95_halfwidth_if_same_sd"]}')
    print('critic geometry', result['critic']['geometry'])
    for phase in phases:
        stats=result['critic'][phase]
        print('critic', phase, stats['mean'], stats['percentile_95'])


if __name__ == '__main__':
    main()
