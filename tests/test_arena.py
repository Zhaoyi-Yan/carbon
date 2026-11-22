import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from arena import Player, PolicyFault, Strategy, run_game


def strategy(folder, name, code):
    source = code.encode()
    path = folder / (name + ".py")
    path.write_bytes(source)
    return Strategy(name, path, hashlib.sha256(source).hexdigest())


class ArenaTests(unittest.TestCase):
    def test_swapped_games_use_identical_map_and_partial_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ("def agent(obs, cfg):\n"
                    "    assert 'full_carbon' not in obs\n"
                    "    assert set(obs.carbon) == {0}\n"
                    "    return {}\n")
            a, b = [strategy(Path(directory), n, code) for n in ("a", "b")]
            first = run_game(((a, b), 20291, 3))
            second = run_game(((b, a), 20291, 3))
            self.assertEqual(first["map_sha256"], second["map_sha256"])
            for game, order in ((first, (a, b)), (second, (b, a))):
                self.assertEqual(game["status"], "completed", game)
                self.assertEqual(game["scores"], [80, 80])
                self.assertIsNone(game["winner"])
                self.assertEqual([p["name"] for p in game["players"]], [p.name for p in order])
                self.assertEqual(game["turns"], 2)

    def test_double_forfeit_records_both_causes(self):
        with tempfile.TemporaryDirectory() as directory:
            a = strategy(Path(directory), "illegal", "def agent(o,c): return {'missing':'UP'}")
            b = strategy(Path(directory), "crash", "def agent(o,c): raise RuntimeError('bad move')")
            game = run_game(((a, b), 71, 3))
            self.assertEqual(game["status"], "double_forfeit")
            self.assertIsNone(game["winner"])
            self.assertIsNone(game["scores"])
            self.assertEqual([p["failure"]["kind"] for p in game["players"]], ["invalid_action", "error"])

    def test_hanging_strategy_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            a = strategy(Path(directory), "sleep", "import time\ndef agent(o,c): time.sleep(60)")
            b = strategy(Path(directory), "idle", "def agent(o,c): return {}")
            game = run_game(((a, b), 11, 2))
            self.assertEqual(game["status"], "forfeit")
            self.assertEqual(game["winner"], 1)
            self.assertEqual(game["players"][0]["failure"]["kind"], "timeout")

    def test_soft_clock_never_replenishes_and_limits_the_fourth_slow_step(self):
        p = Player.__new__(Player)
        p.data = dict(calls=0, slow_steps=0, max_act_seconds=0., total_act_seconds=0.)
        p.remaining = 5.
        p.pipe = Mock()
        p.receive = Mock(return_value=({}, None))
        for elapsed in (1.5, .4, 1.7, 2.):
            with patch("arena.time.monotonic", side_effect=(1., 1.+elapsed)):
                p.act({})
        self.assertAlmostEqual(p.remaining, 2.8)
        self.assertEqual(p.data["slow_steps"], 3)
        p.receive.side_effect = PolicyFault("timeout", "out of slow moves")
        with patch("arena.time.monotonic", side_effect=(10., 11.)), self.assertRaises(PolicyFault):
            p.act({})
        p.receive.assert_called_with(11.)


if __name__ == "__main__":
    unittest.main()
