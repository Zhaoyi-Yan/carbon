import copy
import unittest
from experiments.v5.check_panel import review


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.records=[]
        for seat in (0,1):
            players=[dict(sha256='v5',failure=None,calls=299,max_act_seconds=.5,slow_steps=0),
                     dict(sha256='v4',failure=None,calls=299,max_act_seconds=.6,slow_steps=0)][::1 if seat==0 else -1]
            self.records.append(dict(players=players,seed=3,engine_sha256='engine',layout_sha256='layout',
                                     status='completed',episode_steps=300,turns=299,winner=seat,
                                     cash=[11,10] if seat==0 else [10,11]))

    def test_both_seats_must_beat_exact_baseline(self):
        self.assertTrue(review(self.records,[3],'v5','v4')['passed'])
        self.records[1]['cash']=[15,11]
        self.records[1]['winner']=0
        result=review(self.records,[3],'v5','v4')
        self.assertFalse(result['passed'])
        self.assertEqual([r['margin'] for r in result['rows']],[1,-4])

    def test_forfeit_does_not_count_as_win(self):
        self.records[0]['status']='forfeit'
        self.assertEqual(review(self.records,[3],'v5','v4')['wins'],1)

    def test_duplicate_or_missing_seat_rejected(self):
        with self.assertRaises(ValueError):
            review([self.records[0],copy.deepcopy(self.records[0])],[3],'v5','v4')
        with self.assertRaises(ValueError):
            review(self.records[:1],[3],'v5','v4')
