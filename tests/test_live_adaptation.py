import unittest
from pathlib import Path

from benchmark import configure_engine, limit_math_threads, load_strategy


class LiveAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_engine()
        limit_math_threads()
        cls.model = load_strategy('live_adaptation_test', Path(__file__).resolve().parents[1] / 'submission.py')

    def test_observe_only_adjacent_frames_with_surviving_ids(self):
        profile = self.model._ReturnProfile()
        profile.observe(0, {})
        profile.forecasts = {'alive': (1, (8,), .25), 'lost': (1, (9,), .5)}
        units = {'alive': (8, 100., 'COLLECTOR')}
        profile.observe(1, units)
        profile.observe(1, units)
        self.assertEqual(profile.samples.tolist(), [0., 1.])
        self.assertEqual(profile.residual.tolist(), [0., .75])
        profile.forecasts = {'alive': (1, (8,), .25)}
        profile.observe(3, units)
        self.assertAlmostEqual(profile.samples[1], .96**2)
        self.assertFalse(profile.forecasts)
        self.assertFalse(self.model._ReturnProfile().samples.any())

    def test_home_odds_adjust_only_the_matching_pressure_group(self):
        m = self.model
        players = [[0., {'h': 0}, {'loaded': (2, 120., 'COLLECTOR')}, {}],
                   [0., {'e': 110}, {'empty': (3, 0., 'COLLECTOR')}, {}]]
        logits = m.np.zeros((m.N, 5), dtype=m.np.float32)
        profile = m._ReturnProfile()
        profile.samples[0] = 10.
        profile.residual[0] = 6.
        m.np.testing.assert_array_equal(profile.adjust(players, 0, logits), logits)
        profile.samples[1] = 10.
        profile.residual[1] = 6.
        updated = profile.adjust(players, 0, logits)
        left = m.WIDX['LEFT']
        self.assertGreater(updated[2, left], 0.)
        self.assertEqual(m.np.count_nonzero(updated), 1)
        self.assertFalse(logits.any())

    def test_lead_counts_deliverable_cargo_and_swaps_sign_with_seat(self):
        m = self.model
        players = [[80., {'h': 0}, {'cargo': (1, 300., 'COLLECTOR')}, {}],
                   [180., {'e': 110}, {}, {}]]
        self.assertEqual(m._late_lead(players, 0, m.LAST_TURN - 90), 0.)
        self.assertGreater(m._late_lead(players, 0, m.LAST_TURN), 0.)
        self.assertEqual(m._late_lead(players, 0, m.LAST_TURN), -m._late_lead(players, 1, m.LAST_TURN))
        players[0][2]['cargo'] = (3, 300., 'COLLECTOR')
        self.assertLess(m._late_lead(players, 0, m.LAST_TURN), 0.)

    def test_reply_spread_changes_preference_only_with_distinct_rollouts(self):
        score = self.model._reply_value
        values = {(0, 0): .4, (0, 1): .6, (1, 0): .1, (1, 1): .9}
        self.assertGreater(score(values, 0, 2, 1.), score(values, 1, 2, 1.))
        self.assertLess(score(values, 0, 2, -1.), score(values, 1, 2, -1.))
        self.assertEqual(score(values, 0, 2, 0.), score(values, 1, 2, 0.))
        self.assertEqual(score(values, 0, 1, -1.), values[0, 0])


if __name__ == '__main__':
    unittest.main()
