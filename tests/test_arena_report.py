import json
from pathlib import Path
import tempfile
import unittest

from arena_report import summarize


def match(names, cash, status='completed'):
    return dict(schema_version=1, seed=13, episode_steps=300, map_sha256='map',
                engine_sha256='engine', status=status,
                players=[dict(name=n, sha256=n*4, failure=None) for n in names],
                scores=cash, cash=cash, winner=0 if cash and cash[0] > cash[1] else 1)


class ReportTests(unittest.TestCase):
    def report(self, rows):
        with tempfile.TemporaryDirectory() as temp:
            shard = Path(temp)/'shard.jsonl'
            shard.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return summarize([shard])[0]

    def test_duplicate_rows_and_seat_orientation(self):
        first, second = match('ab', [8, 1]), match('ba', [4, 10])
        data = self.report([first, first, second])
        self.assertEqual((data['paired_maps'], data['wins'], data['mean_cash_margin']), (1, 2, 6.5))

    def test_forfeit_keeps_surviving_game_out_of_paired_scores(self):
        row = match('ba', None, 'forfeit')
        row['players'][0]['failure'] = {'kind':'timeout'}
        data = self.report([match('ab', [7, 2]), row])
        self.assertEqual((data['paired_maps'], data['unpaired_completed']), (0, 1))
        self.assertEqual(data['faults'], {'forfeit': 1})
        self.assertEqual(data['player_failures'], [dict(name='b', kind='timeout', count=1)])

    def test_conflicting_results_and_mixed_code_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting runs'):
            self.report([match('ab', [9, 0]), match('ab', [0, 9])])
        other = match('ba', [0, 9])
        other['players'][1]['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'multiple strategy hashes'):
            self.report([match('ab', [9, 0]), other])


if __name__ == '__main__':
    unittest.main()
