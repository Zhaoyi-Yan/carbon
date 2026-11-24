"""Recompute the frozen full-v8 own-prior screen and confirmation evidence.

Usage: python recompute.py /path/to/e97a386-checkout statistics.json
Only reads committed results and regenerates initial boards; no agents are run.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCE = 'e97a386b290a1727f26331470094ab57cd452948'
BASELINE = 'a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396'
CANDIDATE = '83bc00ca52ba42ece876830649c9bc488105e763179b62018558f555bb94fa9c'
SEED = 20261125
REPLICATES = 1000000


def digest(content):
    return hashlib.sha256(content).hexdigest()


def bootstrap(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    samples = np.empty(REPLICATES)
    for start in range(0, REPLICATES, 10000):
        count = min(10000, REPLICATES - start)
        indices = rng.integers(len(values), size=(count, len(values)))
        samples[start:start+count] = values[indices].mean(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def summarize(rows):
    margins = np.asarray([r['margin'] for r in rows])
    utilities = np.asarray([r['utility'] for r in rows])
    seeds = sorted({r['seed'] for r in rows})
    pairs = [[next(r for r in rows if r['seed'] == seed and r['seat'] == seat)
              for seat in (0, 1)] for seed in seeds]
    map_u = np.asarray([[r['utility'] for r in pair] for pair in pairs])
    map_mean = map_u.mean(axis=1)
    wins_per_map = [sum(r['margin'] > 0 for r in pair) for pair in pairs]
    result = dict(games=len(rows), maps=len(seeds), wins=int(sum(margins > 0)),
                  draws=int(sum(margins == 0)), losses=int(sum(margins < 0)),
                  mean_margin=float(margins.mean()), mean_utility=float(utilities.mean()),
                  map_bootstrap95=bootstrap(map_mean), positive_map_means=int(sum(map_mean > 0)),
                  paired_map_results={str(k):wins_per_map.count(k) for k in (0, 1, 2)},
                  seats={}, per_map={}, failures=sum(r['failure_count'] for r in rows),
                  max_act_seconds=max(r['candidate_max_act_seconds'] for r in rows),
                  both_agents_max_act_seconds=max(r['max_act_seconds'] for r in rows),
                  candidate_max_act_seconds=max(r['candidate_max_act_seconds'] for r in rows),
                  all_turns=sorted({r['turns'] for r in rows}))
    for seat in (0, 1):
        ss = [r for r in rows if r['seat'] == seat]
        result['seats'][str(seat)] = dict(wins=sum(r['margin'] > 0 for r in ss),
            losses=sum(r['margin'] < 0 for r in ss), mean_margin=float(np.mean([r['margin'] for r in ss])),
            mean_utility=float(map_u[:, seat].mean()), map_bootstrap95=bootstrap(map_u[:, seat]))
    result['seat0_minus_seat1'] = dict(mean=float((map_u[:, 0] - map_u[:, 1]).mean()),
                                      map_bootstrap95=bootstrap(map_u[:, 0] - map_u[:, 1]))
    loo = (map_mean.sum() - map_mean) / (len(map_mean) - 1)
    result['exploratory_leave_one_map_out_mean_range'] = [float(loo.min()), float(loo.max())]
    for seed, pair in zip(seeds, pairs):
        result['per_map'][str(seed)] = dict(mean_utility=float(np.mean([r['utility'] for r in pair])),
            wins=sum(r['margin'] > 0 for r in pair),
            seat0_margin=pair[0]['margin'], seat1_margin=pair[1]['margin'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.repository.resolve()
    folder = root / 'analysis/v8/own-prior'
    files = ['analysis/v8/own-prior/' + p for p in
             ['README.md', 'screen-24.jsonl', 'confirm-24.jsonl', 'screen-summary.json',
              'confirm-summary.json', 'own-search-fit-v8.json', 'own-search-model-v8.json', 'panels.json']]
    files += ['yan-v8.py', 'yan-v8-own-prior.py', 'yan-v8-own-prior.json',
              'experiments/v7/panels.json', 'experiments/v8/own_prior_fit.py',
              'experiments/v8/own_prior_panel.py', 'experiments/v8/own_prior_build.py',
              'benchmark.py', 'arena.py']
    hashed = {}
    for rel in files:
        content = (root / rel).read_bytes()
        frozen = subprocess.run(['git', '-C', str(root), 'show', f'{SOURCE}:{rel}'],
                                check=True, stdout=subprocess.PIPE).stdout
        assert content == frozen, rel
        hashed[rel] = digest(content)
    assert hashed['yan-v8.py'] == BASELINE
    assert hashed['yan-v8-own-prior.py'] == CANDIDATE
    load = lambda name: json.loads((folder / name).read_text())
    fit = load('own-search-fit-v8.json')
    panel = load('panels.json')
    train = {r['layout'] for r in fit['training']}
    held = {r['layout'] for r in fit['holdout']}
    sources = {r['layout'] for r in fit['sources']}
    assert len(train) == fit['train_layouts'] == 22
    assert len(held) == fit['holdout_layouts'] == 11
    assert not train & held and train | held == sources
    train_workers = sum(r['workers'] for r in fit['sources'] if r['layout'] in train)
    assert train_workers == fit['train_workers'] == 12322
    old = json.loads((root / 'experiments/v7/panels.json').read_text())
    old_seeds = set(old['development'])
    for fold in ('qualifying', 'confirmation'):
        for batch in old[fold]:
            old_seeds.update(batch)
    assert len(old_seeds) == panel['old_panel_seeds'] == 196
    sys.path.insert(0, str(root))
    from benchmark import configure_engine, new_game
    configure_engine()
    from zerosum_env.envs.carbon import carbon
    from arena import engine_fingerprint
    engine_sha = engine_fingerprint(carbon)

    def initial_hashes(seed):
        state, env = new_game(seed, 300)
        obs = state[0].observation
        content = [obs['full_carbon'], [sorted(p[1]) for p in obs['players']]]
        prior = digest(json.dumps(content, sort_keys=True).encode())
        opening = [obs['full_carbon'], [list(p[1].values()) for p in obs['players']]]
        map_sha = digest(json.dumps(opening, separators=(',', ':')).encode())
        layout = [env.configuration.size, obs['full_carbon'], [sorted(p[1].values()) for p in obs['players']]]
        arena_sha = digest(json.dumps(layout, separators=(',', ':')).encode())
        return prior, map_sha, arena_sha

    old_layouts = {initial_hashes(seed)[0] for seed in old_seeds}
    all_panel_layouts = set()
    all_rows = []
    result = dict(source_commit=SOURCE, input_sha256=hashed, panels={},
                  method=dict(rng_seed=SEED, bootstrap_replicates=REPLICATES,
                              numpy=np.__version__, python=platform.python_version(),
                              interval='unadjusted percentile bootstrap; sample whole maps retaining both seats',
                              notes=['Same single candidate throughout both batches.',
                                     'Independent confirmation is the prespecified decision sample; pooled interval is descriptive.',
                                     'New bootstrap uses a documented fixed RNG and one million resamples; small differences from the stored interval are Monte Carlo differences.',
                                     'Seat and leave-one-map-out summaries are exploratory, not additional gates.',
                                     'Recorded replay fingerprints are checked for partition separation, but upstream replay bytes and deployment identity are not revalidated here.']))
    for name in ('screen', 'confirm'):
        definitions = panel['panels'][name]
        expected = {r['seed']: r['layout'] for r in definitions}
        assert len(expected) == 12 and not set(expected) & old_seeds
        assert not set(expected.values()) & (sources | old_layouts | all_panel_layouts)
        all_panel_layouts.update(expected.values())
        generated = {seed: initial_hashes(seed) for seed in expected}
        assert all(generated[seed][0] == value for seed, value in expected.items())
        rows = []
        raw = [json.loads(line) for line in (folder / f'{name}-24.jsonl').read_text().splitlines()]
        for g in raw:
            assert g['engine_sha256'] == engine_sha
            assert g['seed'] in expected
            assert (g['map_sha256'], g['layout_sha256']) == generated[g['seed']][1:]
            assert g['status'] == 'completed' and g['turns'] == 299 and g['episode_steps'] == 300
            assert len(g['actions_and_post_cash']) == 299
            assert g['actions_and_post_cash'][-1][1] == g['cash'] == g['scores']
            assert {p['sha256'] for p in g['players']} == {BASELINE, CANDIDATE}
            seat = next(i for i, p in enumerate(g['players']) if p['sha256'] == CANDIDATE)
            delta = g['scores'][seat] - g['scores'][1-seat]
            assert g['winner'] == (seat if delta > 0 else 1-seat if delta < 0 else None)
            assert all(p['search_enabled'] and p['calls'] == 299 and p['searches'] > 0 for p in g['players'])
            assert all(not p['failure'] and p['slow_steps'] == 0 for p in g['players'])
            rows.append(dict(seed=g['seed'], seat=seat, margin=delta,
                utility=float(.5*np.sign(delta) + .5*np.tanh(delta/1500)), turns=g['turns'],
                failure_count=sum(bool(p['failure']) for p in g['players']),
                max_act_seconds=max(p['max_act_seconds'] for p in g['players']),
                candidate_max_act_seconds=g['players'][seat]['max_act_seconds']))
        assert len(rows) == len({(r['seed'], r['seat']) for r in rows}) == 24
        assert {(r['seed'], r['seat']) for r in rows} == {(seed, seat) for seed in expected for seat in (0, 1)}
        stats = summarize(rows)
        stored = load(f'{name}-summary.json')
        for field in ('games', 'wins', 'losses', 'failures', 'mean_margin', 'mean_utility', 'max_act_seconds'):
            assert abs(stats[field] - stored[field]) < 1e-12, (name, field)
        for seat in ('0', '1'):
            for field in ('wins', 'losses', 'mean_utility'):
                assert abs(stats['seats'][seat][field] - stored['seats'][seat][field]) < 1e-12
        if 'map_bootstrap95' in stored:
            stats['stored_map_bootstrap95'] = stored['map_bootstrap95']
        result['panels'][name] = stats
        all_rows.extend(rows)
    result['panels']['pooled_descriptive'] = summarize(all_rows)
    cal = np.asarray([r['calibrated'] for r in fit['holdout']])
    learned = np.asarray([r['learned'] for r in fit['holdout']])
    result['fit'] = dict(train_layouts=len(train), heldout_layouts=len(held),
                        train_workers=train_workers, heldout_workers=sum(r['workers'] for r in fit['holdout']),
                        map_macro_calibrated_nll=float(cal[:,0].mean()), map_macro_learned_nll=float(learned[:,0].mean()),
                        map_macro_calibrated_accuracy=float(cal[:,1].mean()), map_macro_learned_accuracy=float(learned[:,1].mean()),
                        nll_change=float((learned[:,0] - cal[:,0]).mean()),
                        accuracy_change_percentage_points=float(100*(learned[:,1] - cal[:,1]).mean()),
                        comparison='new temperature-calibrated root logits versus same logits plus newly fitted residual; not old-v8-prior versus new-v8-prior')
    result['gates'] = dict(screen_pass=result['panels']['screen']['mean_utility'] > 0,
                          confirmation_pass=result['panels']['confirm']['mean_utility'] > 0 and result['panels']['confirm']['map_bootstrap95'][0] > 0,
                          no_runtime_failures=True)
    result['audit'] = dict(all_input_bytes_match_frozen_commit=True, all_game_hashes_match=True,
                          initial_boards_regenerated=24+len(old_seeds), engine_sha256=engine_sha,
                          train_holdout_screen_confirm_layouts_disjoint=True,
                          screen_confirm_disjoint_from_regenerated_v7_layouts=True,
                          original_two_fingerprint_schemes_verified=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print('Wrote', args.output)
    for name, data in result['panels'].items():
        print(name, {k: data[k] for k in ('wins', 'draws', 'losses', 'mean_utility', 'mean_margin', 'map_bootstrap95', 'paired_map_results', 'seat0_minus_seat1', 'exploratory_leave_one_map_out_mean_range')})
    print('fit', result['fit'])
    print('gates', result['gates'])


if __name__ == '__main__':
    main()
