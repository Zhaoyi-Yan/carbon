"""Recompute saved calibration traces and verify the finite-horizon GAE identity.

This is an offline NumPy audit of published trajectories, not training or a new
game evaluation. Predictions/Phi were rounded to six decimals by the producer.
Usage: python advantage_audit.py /path/to/frozen-checkout output.json
"""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SOURCE_COMMIT = 'be792f64e3b25b392b1cd502e5b55cfbf11e0950'
SOURCE_FILE = 'experiments/carbon_zero/terminal_calibration.py'
DATA_FILE = 'analysis/carbon-zero-terminal/b1/calibration/audit-8maps-fixed.json'


def summary(x):
    return dict(mean=float(x.mean()), std=float(x.std()),
                rms=float(np.sqrt(np.mean(x*x))), mean_abs=float(np.abs(x).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    repo = args.repository.resolve()
    inputs = {}
    for name in (SOURCE_FILE, DATA_FILE):
        content = (repo / name).read_bytes()
        committed = subprocess.run(['git', '-C', str(repo), 'show',
                                    f'{SOURCE_COMMIT}:{name}'], check=True,
                                   stdout=subprocess.PIPE).stdout
        assert content == committed, f'Input differs from frozen snapshot: {name}'
        inputs[name] = hashlib.sha256(content).hexdigest()
    data = json.loads((repo / DATA_FILE).read_text())
    assert data['script_sha256'] == inputs[SOURCE_FILE]
    tree = ast.parse((repo / SOURCE_FILE).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'full_game_advantages')
    scope = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), SOURCE_FILE, 'exec'), scope)
    gae_function = scope['full_game_advantages']

    result = dict(source_commit=SOURCE_COMMIT, input_sha256=inputs,
                  evidence='Saved calibration trajectories only; not M/G training trajectories.',
                  method='Actual source GAE helper, float64 inputs rounded to six decimals.',
                  interpretation='Signed mean is not RMS. GAE-MC differences do not establish estimator bias.',
                  results={})
    identity_error = 0.0
    decomposition_error = 0.0
    for name, report in data['results'].items():
        mcs, gaes, phis, values, labels, steps = [], [], [], [], [], []
        for trace in report['traces']:
            value = np.asarray(trace['value'], dtype=float)
            phi = np.asarray(trace['phi'], dtype=float)
            utility = trace['utility']
            assert value.shape == phi.shape == (trace['turns'],)
            reward = np.empty_like(phi)
            reward[:-1] = phi[1:] - phi[:-1]
            reward[-1] = utility - phi[-1]
            mc = utility - phi - value
            gae = gae_function(reward, value, .95)
            identity_error = max(identity_error, float(np.max(np.abs(
                gae_function(reward, value, 1.0) - mc))))
            baseline = phi + value
            expanded = np.empty_like(phi)
            for t in range(len(value)):
                future = baseline[t+1:]
                expanded[t] = (-baseline[t] + .05 * np.dot(.95 ** np.arange(len(future)), future)
                               + .95 ** len(future) * utility)
            decomposition_error = max(decomposition_error,
                                      float(np.max(np.abs(expanded - gae))))
            mcs.append(mc); gaes.append(gae); phis.append(phi); values.append(value)
            labels.append(utility - phi); steps.append(np.arange(len(phi)))
        mc, gae, value, label, turn = map(np.concatenate, (mcs, gaes, values, labels, steps))
        phases = {'all': np.ones(len(turn), bool), 'opening': turn <= 60,
                  'middle': (turn > 60) & (turn < 230), 'ending': turn >= 230}
        computed = {}
        for phase, mask in phases.items():
            mse = float(np.mean((value[mask] - label[mask]) ** 2))
            ev = float(1 - np.var(value[mask] - label[mask]) / np.var(label[mask]))
            assert abs(mse - report['value'][phase]['MSE']) < 3e-6
            assert abs(ev - report['value'][phase]['EV']) < 3e-6
            corr = float(np.corrcoef(gae[mask], mc[mask])[0, 1])
            assert abs(corr - report['gae_vs_mc'][phase]['corr']) < 3e-6
            computed[phase] = dict(states=int(mask.sum()), M=summary(mc[mask]), G=summary(gae[mask]),
                                   M_rms_over_G_rms=float(np.sqrt(np.mean(mc[mask] ** 2) / np.mean(gae[mask] ** 2))),
                                   gae_mc_corr=corr, value_mse=mse, value_ev=ev)
        result['results'][name] = computed
    assert identity_error < 1e-12
    assert decomposition_error < 1e-12
    result['checks'] = dict(lambda1_mc_max_abs_error=identity_error,
                            finite_horizon_expansion_max_abs_error=decomposition_error)
    result['terminal_coefficient'] = {
        str(remaining_steps): .95 ** (remaining_steps - 1)
        for remaining_steps in (20, 50, 100, 299)}
    result['terminal_coefficient_note'] = (
        'Coefficient of the explicit terminal sample in finite-horizon GAE. Future value '
        'predictions carry additional long-term information; this is not the fraction of '
        'total useful policy-gradient signal and does not prove short-sightedness.')
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'checks': result['checks'],
                      'zero_step_all': result['results']['ZW0critic']['all'],
                      'terminal_coefficient': result['terminal_coefficient']}, indent=2))


if __name__ == '__main__':
    main()
