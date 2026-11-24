"""Focused tests for the full-game terminal-RL contract."""
import itertools
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pilot import np, torch
from env import Game
from ppo import joint_logprob
from terminal_games import group_matches, play
from terminal_rloo import archive_trace, leave_one_out, prepare, ratio_loss


class DummyEncoder:
    def __init__(self, _teacher, _seat):
        pass

    def encode(self, observation):
        if 'full_carbon' in observation or 'randomSeed' in observation:
            raise AssertionError('Hidden seed or resource state leaked')
        return np.zeros((54, 15, 15), np.uint8)

    def commit(self, _observation, _action):
        pass


class DummyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.workers = torch.nn.Linear(1, 5, bias=False)
        self.bases = torch.nn.Linear(1, 3, bias=False)

    def spatial(self, x):
        return torch.zeros((len(x), 225, 1), device=x.device)


class NoCashGame(Game):
    def __init__(self, seed, steps):
        super().__init__(seed, steps)
        for team in self.state[0].observation.players:
            team[0] = 0.


class TerminalTests(unittest.TestCase):
    def test_leave_one_out(self):
        returns = np.asarray([1., .5, -.5, -1.])
        expected = np.asarray([4/3, 2/3, -2/3, -4/3])
        np.testing.assert_allclose(leave_one_out(returns), expected)
        np.testing.assert_allclose(leave_one_out(returns + 8), expected)
        np.testing.assert_array_equal(leave_one_out([1., 1., 1., 1.]), np.zeros(4))
        self.assertAlmostEqual(float(leave_one_out(returns).sum()), 0.)

    def test_enumerated_two_action_game(self):
        gradient = 0.
        rewards = [-1., 1.]
        for actions in itertools.product(range(2), repeat=2):
            parameter = torch.zeros(2, requires_grad=True)
            score = parameter.expand(2, 1, 2)
            moves = torch.as_tensor(actions)[:, None]
            old = joint_logprob(score.detach(), moves)
            values = torch.as_tensor(leave_one_out([rewards[i] for i in actions]))
            loss, kl, _ = ratio_loss(score, moves, old, values, .5)
            self.assertEqual(float(kl), 0.)
            loss.backward()
            gradient += float(parameter.grad[1]) / 4
        self.assertAlmostEqual(gradient, -.5)
        score = torch.randn(4, 1, 2, requires_grad=True)
        a = torch.zeros((4, 1), dtype=torch.long)
        old = joint_logprob(score.detach(), a)
        loss, _, _ = ratio_loss(score, a, old, torch.zeros(4), .25)
        loss.backward()
        self.assertTrue(torch.count_nonzero(score.grad) == 0)

    @patch('terminal_games.Encoder', DummyEncoder)
    def test_a_full_batch_replays_independent_games(self):
        actor = DummyActor()
        matches = group_matches('test', 0, group_size=2)
        self.assertEqual(matches, group_matches('test', 0, group_size=2))
        self.assertEqual(len({m.learner_rng for m in matches}), len(matches))
        self.assertTrue(set(m.learner_rng for m in matches).isdisjoint(m.opponent_rng for m in matches))
        results, _ = play(None, actor, actor, matches, 'cpu', capture='policy', steps=6)
        arrays, rows, groups = prepare(results)
        self.assertEqual(len(arrays[0]), sum(r['turns'] for r in rows))
        self.assertEqual(len(groups), 4)
        for spec in matches:
            single, _ = play(None, actor, actor, [spec], 'cpu', capture='policy', steps=6)
            other = next(result for result in results if result[0]['learner_rng'] == spec.learner_rng)
            self.assertEqual(single[0][2], other[2])
            self.assertEqual(single[0][0]['scores'], other[0]['scores'])
            self.assertEqual(other[0]['turns'], 5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trace.jsonl.gz'
            self.assertEqual(len(archive_trace(path, results)), 64)

    @patch('terminal_games.Game', NoCashGame)
    @patch('terminal_games.Encoder', DummyEncoder)
    def test_early_elimination_is_a_real_terminal_tie(self):
        actor = DummyActor()
        matches = group_matches('early', 0, group_size=2)
        results, _ = play(None, actor, actor, matches, 'cpu', capture='policy')
        arrays, rows, groups = prepare(results)
        self.assertEqual(len(arrays[0]), len(matches))
        self.assertTrue(all(row['elimination'] and row['utility'] == 0 for row in rows))
        self.assertTrue(all(group['zero_advantage'] for group in groups))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
