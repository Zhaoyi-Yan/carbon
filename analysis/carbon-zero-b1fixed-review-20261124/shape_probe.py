"""Reproduce the calibration shape bug without PyTorch or model weights.

Extracts the actual predict/analyse/metrics/corr functions from the audited
checkout. Only tensor conversion and a zero-output critic are stubbed. This
executes the real NumPy analysis, not neural inference or game rollouts.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np

EXPECTED_COMMIT = 'd6008ed2e8bc42df056ab08ac48711d19c4249c8'


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeTorch:
    float32 = np.float32

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def as_tensor(value, dtype=None, device=None):
        return np.asarray(value, dtype=dtype)


class ZeroCritic:
    def __init__(self, residual_critic=False):
        self.residual_critic = residual_critic

    def value(self, features):
        # nn.Sequential(..., Linear(..., 1)) output shape.
        return Tensor(np.zeros((len(features), 1)))

    def predict_value(self, features, pub):
        # Actor.predict_value uses [:, 0] in residual mode.
        return Tensor(np.zeros(len(features)))


def extract(repo):
    path = repo / 'experiments/carbon_zero/terminal_calibration.py'
    policy_path = repo / 'experiments/carbon_zero/policy.py'
    tree = ast.parse(path.read_text())
    names = {'predict', 'analyse', 'metrics', 'corr'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    policy = ast.parse(policy_path.read_text())
    functions.append(next(n for n in policy.body if isinstance(n, ast.FunctionDef)
                          and n.name == 'public_state_features'))
    ppo = ast.parse((repo / 'experiments/carbon_zero/ppo.py').read_text())
    functions.append(next(n for n in ppo.body if isinstance(n, ast.FunctionDef)
                          and n.name == 'advantages'))
    scope = {'np': np, 'torch': FakeTorch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    return scope, hashlib.sha256(path.read_bytes()).hexdigest()


def exact_critic_counterexample(scope):
    """Synthetic terminal coin flip independent of every state and action."""
    horizon = 299
    rewards = np.zeros((horizon, 2), dtype=np.float64)
    rewards[-1] = [1., -1.]
    values = np.zeros_like(rewards)
    terminal = np.zeros_like(rewards, dtype=bool)
    terminal[-1] = True
    gae = scope['advantages'](rewards, values, terminal, np.zeros(2), lam=.95)
    mc = np.broadcast_to(rewards[-1], rewards.shape)
    gae_lambda_one = scope['advantages'](rewards, values, terminal, np.zeros(2), lam=1.)
    assert np.array_equal(gae_lambda_one, mc)
    expected = (.95 ** np.arange(horizon - 1, -1, -1))[:, None] * rewards[-1]
    np.testing.assert_allclose(gae, expected, atol=1e-14)
    return {
        'type': 'Synthetic mathematical counterexample, not Carbon experiment data.',
        'environment': 'Horizon 299; Phi=0; only terminal reward is an independent fair +/-1 coin, independent of every state/action.',
        'critic': 'V=0 is the exact conditional expected return at every nonterminal state.',
        'trajectories': 'One realized +1 trajectory and one realized -1 trajectory; same length.',
        'lambda': .95,
        'GAE_formula': '0.95 ** (298-t) * R',
        'MC_formula': 'R',
        'gae_mc_corr': scope['corr'](gae, mc),
        'mean_abs_difference': float(np.mean(np.abs(gae - mc))),
        'lambda_one_equals_mc': True,
        'implication': 'Large realized GAE/MC difference or correlation below 0.5 alone cannot establish an inaccurate critic, incorrect GAE, or an expected policy-gradient error.',
    }


def game(gaps, utility, seed=1):
    trace = [(np.ones(2), gap, t, None) for t, gap in enumerate(gaps)]
    return ({'utility': utility, 'seed': seed, 'seat': 0}, trace, None)


def main(repo, out):
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f'Expected {EXPECTED_COMMIT}, got {commit}')
    sources = {}
    for relative in ('experiments/carbon_zero/terminal_calibration.py',
                     'experiments/carbon_zero/policy.py', 'experiments/carbon_zero/ppo.py'):
        raw = (repo / relative).read_bytes()
        pinned = subprocess.check_output(['git', '-C', str(repo), 'show', f'{EXPECTED_COMMIT}:{relative}'])
        if raw != pinned:
            raise ValueError(f'Source differs from pinned commit: {relative}')
        sources[relative] = hashlib.sha256(raw).hexdigest()
    scope, source_digest = extract(repo)
    original_predict = scope['predict']
    actor = ZeroCritic()
    played = [game([0., 200., -100., 400.], .8)]
    unequal = played + [game([0., 50., -20., 100., 75.], -.3, seed=2)]
    cases = {}
    for label, flatten in [('current_code', False), ('flatten_predictions', True)]:
        scope['predict'] = ((lambda *args: np.asarray(original_predict(*args)).reshape(-1))
                            if flatten else original_predict)
        values = scope['predict'](actor, np.ones((4, 2)), np.zeros((4, 6)), 'cpu')
        phi = .5 * np.tanh(np.asarray([0., 200., -100., 400.]) / 1000.)
        mc = .8 - phi - values
        with warnings.catch_warnings():
            # Four-step toy has no middle/ending states; only report 'all'.
            warnings.simplefilter('ignore', RuntimeWarning)
            report = scope['analyse'](played, actor, np.zeros((1, 6)), 'cpu')
            try:
                variable_report = scope['analyse'](unequal, actor, np.zeros((1, 6)), 'cpu')
                variable = {'success': True, 'states': variable_report['states']}
            except ValueError as exc:
                variable = {'success': False, 'error_type': type(exc).__name__, 'message': str(exc)}
        cases[label] = {
            'prediction_shape': list(values.shape),
            'mc_shape': list(mc.shape),
            'value_all': report['value']['all'],
            'gae_vs_mc_all': report['gae_vs_mc']['all'],
            'unequal_episode_lengths_4_and_5': variable,
        }
    current, corrected = cases['current_code'], cases['flatten_predictions']
    assert current['prediction_shape'] == [4, 1] and current['mc_shape'] == [4, 4]
    assert corrected['prediction_shape'] == [4] and corrected['mc_shape'] == [4]
    assert current['value_all'] == corrected['value_all']
    assert current['gae_vs_mc_all']['corr'] is None
    assert corrected['gae_vs_mc_all']['corr'] is not None
    assert not current['unequal_episode_lengths_4_and_5']['success']
    assert corrected['unequal_episode_lengths_4_and_5']['states'] == 9
    result = {
        'audited_commit': commit,
        'source_files_sha256': sources,
        'source_files_verified_against_git': True,
        'terminal_calibration_source_sha256': source_digest,
        'scope': 'Actual calibration NumPy functions; synthetic zero critic and tensor conversion; no neural inference or games.',
        'cases': cases,
        'exact_critic_counterexample': exact_critic_counterexample(scope),
        'assertions_passed': True,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    main(args.repo.resolve(), args.out.resolve())
