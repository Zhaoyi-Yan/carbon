"""Contract checks for the optional collector proposal, independent of game time and weights."""
import unittest
from unittest.mock import patch
import numpy as np
import harvest


class CollectorProposalTest(unittest.TestCase):
    def setUp(self):
        self.carbon=[0.]*225
        self.carbon[31]=70.
        self.own=[300,{"home":0},{"a":[31,80.,"COLLECTOR"]},{}]
        self.foe=[300,{"foe":224},{},{}]
        self.obs=dict(step=100,player=0,players=[self.own,self.foe],carbon=self.carbon)
        self.logits=np.zeros((225,5))
        self.root={"a":"RIGHT","home":"RECCOLLECTOR"}
        self.addCleanup(patch.stopall)
        patch.object(harvest,"_HV_THRESH",(.58,.64),create=True).start()
        patch.object(harvest,"_HV_GAPS",(.5,.5),create=True).start()
        self.predict=patch.object(harvest,"_hv_predict",return_value=np.array([[.92,.03,.05]])).start()

    def result(self, locked=()):
        return harvest._hv_proposal(self.obs,298,self.root,self.logits,locked)

    def test_stay_only_changes_one_worker_and_preserves_original(self):
        self.assertEqual(self.result(),{"home":"RECCOLLECTOR"})
        self.assertEqual(self.root,{"a":"RIGHT","home":"RECCOLLECTOR"})

    def test_blocks_friendly_landing_and_enemy_threat(self):
        self.own[2]["b"]=[30,0.,"PLANTER"]
        self.root["b"]="RIGHT"
        self.assertIsNone(self.result())
        self.own[2].pop("b")
        self.root.pop("b")
        self.foe[2]["e"]=[32,0.,"COLLECTOR"]
        self.assertIsNone(self.result())

    def test_respects_endgame_lock_and_old_policy_confidence(self):
        self.assertIsNone(self.result(locked=("a",)))
        self.obs["step"]=297
        self.assertIsNone(self.result())
        self.obs["step"]=100
        self.logits[31,2]=3
        self.assertIsNone(self.result())

    def test_preserves_a_deposit_already_chosen_by_root(self):
        self.own[1]["near"]=32
        with patch.object(harvest,"_HV_PRESERVE_BANK",True):
            self.assertIsNone(self.result())

    def test_return_uses_best_safe_inward_direction(self):
        self.predict.return_value=np.array([[.02,.94,.04]])
        self.root.pop("a")
        self.logits[31,4]=.4
        proposal=self.result()
        self.assertEqual(proposal["a"],"LEFT")
        self.assertLess(harvest._HV_DIST[30][0],harvest._HV_DIST[31][0])


if __name__=="__main__":
    unittest.main()
