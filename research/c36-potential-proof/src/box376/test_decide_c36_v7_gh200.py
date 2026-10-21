"""The GH200 study has two fresh runs; missing finals must not become a negative result."""
import json
import unittest
from pathlib import Path

from box376.decide_c36_v7 import decide
from box376.test_decide_c36_v7 import observation, reader

PROTOCOL = json.loads((Path(__file__).resolve().parents[2]
                      / 'protocol-v7-gh200-fresh-development-20261021.json').read_text())

class FreshGH200DecisionTests(unittest.TestCase):
    def data(self):
        return {'runs':[r for r in reader()['runs'] if r['arm']=='fresh']}

    def test_two_complete_finals_and_missing_run(self):
        data=self.data()
        self.assertEqual(decide(PROTOCOL,data)['outcome']['key'],'F3_fail')
        data['runs'].pop()
        result=decide(PROTOCOL,data)
        self.assertEqual(result['outcome']['key'],'F0_incomplete')
        self.assertEqual(result['runs_without_complete_final_observation'],['fresh:361902'])

    def test_pass_and_near_miss_precede_incomplete(self):
        data=self.data();data['runs'].pop()
        data['runs'][0]['physical']['final']=observation(pressure=1.4e-3)
        self.assertEqual(decide(PROTOCOL,data)['outcome']['key'],'F2_near_miss')
        data['runs'][0]['physical']['final']=observation(passed=True,pressure=8e-4)
        self.assertEqual(decide(PROTOCOL,data)['outcome']['key'],'F1_pass')

    def test_training_and_physical_gates_are_preserved(self):
        old=json.loads((Path(__file__).resolve().parents[2]
                        / 'protocol-v7-ssbroyden-development-20261019.json').read_text())
        for field in ('physical','physical_observation','source_sha256','calibration_sha256','seeds','topology','atol','rtol'):
            self.assertEqual(PROTOCOL[field],old[field],field)
        for key,value in old['training'].items():
            if key not in ('arms','maximum_training_seconds_reason'):
                self.assertEqual(PROTOCOL['training'][key],value,key)
        self.assertFalse(PROTOCOL['execution']['legacy_triton_target_compat'])
        self.assertEqual(set(PROTOCOL['decision_rule']['runs']),{'fresh:361901','fresh:361902'})

if __name__=='__main__':unittest.main()
