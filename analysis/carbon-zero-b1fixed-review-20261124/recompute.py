"""Read-only recomputation of B1-fixed d6008ed evidence; no training.
Usage: python recompute.py /path/to/carbon-checkout output.json
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np

SOURCE_COMMIT = 'd6008ed2e8bc42df056ab08ac48711d19c4249c8'
RNG_SEED = 20261124
REPLICATES = 100000


def normalise(obj):
    if isinstance(obj, dict):
        return {str(k): normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalise(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return normalise(obj.tolist())
    if isinstance(obj, np.generic):
        return normalise(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        raise ValueError('Nonfinite scalar in output')
    return obj


def ci(values):
    a = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(RNG_SEED).integers(len(a), size=(REPLICATES, len(a)))
    samples = a[draws].mean(axis=1)
    return {'mean': a.mean(), 'percentile95': np.quantile(samples, [.025, .975]),
            'map_count': len(a), 'map_sd': a.std(ddof=1)}


def game_summary(rows):
    gaps = np.asarray([g['scores'][g['seat']] - g['scores'][1 - g['seat']] for g in rows], dtype=float)
    utility = [.5*np.sign(d) + .5*np.tanh(d/1500) if r['kind'] == 'full' else np.sign(d)
               for d, r in zip(gaps, rows)]
    return dict(games=len(rows), wins=np.sum(gaps > 0), draws=np.sum(gaps == 0),
                losses=np.sum(gaps < 0), mean_margin=gaps.mean() if len(rows) else None,
                mean_utility=np.mean(utility) if len(rows) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    repo = args.repository.resolve()
    base = repo / 'analysis/carbon-zero-terminal/b1'
    files = sorted((base/'evaluation-shared').glob('*.json'))
    files += sorted((base/'b1-fixed/s1-50k').glob('*.json'))
    files += sorted((base/'b1-fixed/s2-50k').glob('*.json'))
    files += [base/'calibration/audit-8maps.json', base/'b1-fixed/calibration-s1.json',
              base/'b1-fixed/calibration-s2.json', base/'critic/critic.pt']
    for path in files:
        relative = path.relative_to(repo).as_posix()
        pinned = subprocess.check_output(['git', '-C', str(repo), 'show', f'{SOURCE_COMMIT}:{relative}'])
        if path.read_bytes() != pinned:
            raise ValueError(f'Input differs from pinned source: {relative}')
    load = lambda rel: json.loads((base/rel).read_text())
    result = {'source_commit': SOURCE_COMMIT,
              'source_inputs_verified_against_git': True,
              'input_sha256': {str(f.relative_to(repo)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              'method': {'numpy_version': np.__version__, 'python_version': platform.python_version(),
                         'rng': 'numpy.default_rng PCG64; reset per comparison', 'seed': RNG_SEED,
                         'replicates': REPLICATES, 'bootstrap_unit': 'map, average both seats before resampling',
                         'interval': 'unadjusted exploratory 95% percentile bootstrap',
                         'macro': 'equal weight to the two fixed training seeds within each map and seat; no extra independent maps',
                         'limitations': ['Only ZW0 opponent was evaluated in the shared panel.',
                                         'Intervals condition on the fixed training seeds, not all training randomness.',
                                         'Calibration reports lack per-game or per-map prediction errors; their uncertainty cannot be recomputed.',
                                         'Calibration states and continuation policies differ between actor checkpoints.',
                                         'Non-residual GAE/MC summary has a (T,1)/(T,) broadcasting defect; no use of those metrics as valid evidence.']},
              'evaluation': {}, 'paired': {}, 'training': {}, 'calibration': {}}
    names = ['ZW0', 'A50', 'C50', 'B1fix_s1', 'B1fix_s2']
    data = {n: load(f'evaluation-shared/{n}.json') for n in names}
    protocol = load('evaluation-shared/protocol.json')
    result['evaluation']['protocol_rivals'] = list(protocol['panels'])
    result['evaluation']['shared_maps'] = protocol['shared_maps']
    games = {}
    for name in names:
        assert list(data[name]['rivals']) == ['ZW0']
        row = data[name]['rivals']['ZW0']
        games[name] = {(g['seed'], g['seat']): g for g in row['games']}
        result['evaluation'][name] = dict(checkpoint_sha256=data[name]['checkpoint_sha256'], **row['overall'])
    keys = sorted(games['ZW0'])
    seeds = sorted({k[0] for k in keys})
    result['evaluation']['map_count'] = len(seeds)
    for name in names:
        assert set(games[name]) == set(keys)
        for k in keys:
            for field in ('learner_rng', 'opponent_rng', 'layout_sha256'):
                assert games[name][k][field] == games['ZW0'][k][field]
    pairs = [(n, [n], 'ZW0') for n in names if n != 'ZW0']
    pairs += [(n, [n], ref) for n in ('B1fix_s1', 'B1fix_s2') for ref in ('A50', 'C50')]
    pairs += [('B1fix_s2', ['B1fix_s2'], 'B1fix_s1')]
    pairs += [('B1fix_two_seed_mean', ['B1fix_s1', 'B1fix_s2'], ref) for ref in ('ZW0', 'A50', 'C50')]
    for label, selected, ref in pairs:
        report = {}
        for metric in ('margin', 'utility', 'win_score'):
            value = ((lambda g: float(g['margin'] > 0) + .5*float(g['margin'] == 0))
                     if metric == 'win_score' else lambda g: g[metric])
            values = [np.mean([np.mean([value(games[n][(s, seat)]) for n in selected])
                               - value(games[ref][(s, seat)]) for seat in (0, 1)]) for s in seeds]
            report[metric] = ci(values)
        result['paired'][label+'-'+ref] = report
    for run in ('s1-50k', 's2-50k'):
        ep = load(f'b1-fixed/{run}/episodes.json')
        updates = load(f'b1-fixed/{run}/updates.json')
        protocol = load(f'b1-fixed/{run}/protocol.json')
        assert updates[-1]['finished'] == len(ep)
        result['training'][run] = {
            'all_completed': game_summary(ep),
            'full_only': game_summary([g for g in ep if g['kind'] == 'full']),
            'elimination_only': game_summary([g for g in ep if g['kind'] == 'elimination']),
            'elimination_games': [g for g in ep if g['kind'] == 'elimination'],
            'actor_E': updates[-1]['E'], 'total_E_including_pretrain_and_critic_validation': updates[-1]['total_E'],
            'updates': len(updates), 'baseline_max_abs_diff_max': max(u['baseline_max_abs_diff'] for u in updates),
            'baseline_in_optimizer_any': any(u['baseline_in_optimizer'] for u in updates),
            'seed': protocol['seed'],
        }
    for rel in ('calibration/audit-8maps.json', 'b1-fixed/calibration-s1.json', 'b1-fixed/calibration-s2.json'):
        d = load(rel)
        for name, row in d['results'].items():
            stats = {'states': row['states'], 'games': row['games'], 'per_map_utility': row['per_map'], 'phases': {}}
            for phase in ('all', 'opening', 'middle', 'ending'):
                v, c, z = (row[k][phase] for k in ('value', 'cash', 'zero'))
                label_var = z['MSE'] - z['bias']**2
                for r in (v, c):
                    assert np.isclose((r['MSE']-r['bias']**2)/(1-r['EV']), label_var, rtol=1e-5, atol=1e-9)
                stats['phases'][phase] = {
                    'value': v, 'cash': c, 'label_variance': label_var,
                    'MSE_cash_minus_value': c['MSE']-v['MSE'],
                    'EV_value_minus_cash': v['EV']-c['EV'],
                    'bias_squared_cash_minus_value': c['bias']**2-v['bias']**2,
                }
            if name in result['calibration']:
                assert result['calibration'][name] == stats
            result['calibration'][name] = stats
    result = normalise(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'Wrote {args.output}, {len(files)} hashed inputs.')
    for pair in result['paired']:
        if 'two_seed' in pair or pair.endswith('-ZW0') or pair.endswith('-B1fix_s1'):
            print(pair, result['paired'][pair]['utility'])
    for name, row in result['training'].items():
        print(name, row['all_completed'], 'full', row['full_only'], 'elim', row['elimination_only'])
    for name in ('B1_50k', 'B1fix_50k', 'B1fix_s2_50k'):
        x = result['calibration'][name]['phases']['all']
        print(name, 'var_y', x['label_variance'], 'value', x['value'], 'cash', x['cash'], 'delta_EV',x['EV_value_minus_cash'])


if __name__ == '__main__':
    main()
