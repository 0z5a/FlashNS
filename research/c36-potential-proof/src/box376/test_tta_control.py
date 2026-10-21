import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from box376.tta_control import ActiveClock, cohort_result


class TTATests(unittest.TestCase):
    def test_failure_archive_retains_negative_evidence_and_rejects_overwrite(self):
        from box376.run_c36_tta import seal_archive, sha
        import tarfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); packet = d/'packet'; run = d/'run'
            packet.mkdir(); run.mkdir()
            (packet/'protocol.json').write_text('{"frozen":true}')
            (run/'failure.log').write_text('observer exited nonzero; partial evidence retained')
            archive = d/'negative.tar.gz'
            receipt = seal_archive(packet,run,archive)
            self.assertEqual(receipt['sha256'],sha(archive))
            self.assertEqual(receipt['members'],2)
            with tarfile.open(archive) as t:
                self.assertIn(b'nonzero',t.extractfile('run/failure.log').read())
            with self.assertRaises(RuntimeError):
                seal_archive(packet,run,archive)

    def test_outer_finalizer_archives_failure_before_driver_start(self):
        import subprocess, os, tarfile
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); packet=d/'packet';packet.mkdir()
            (packet/'protocol.json').write_text('{"frozen":true}')
            root=d/'seed1';(root/'control').mkdir(parents=True)
            (root/'batch.log').write_text('module admission failed before Python driver')
            p=subprocess.run([sys.executable,'-m','box376.finalize_c36_tta',str(root),str(packet),'fixture','1'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            receipt=json.loads((d/'archives/seed1-jobfixture.json').read_text())
            self.assertTrue(receipt['outer_failure_archive'])
            self.assertEqual(receipt['workload_exit'],1)
            with tarfile.open(d/'archives/seed1-jobfixture.tar.gz') as t:
                self.assertIn(b'admission failed',t.extractfile('run/batch.log').read())

    def test_observer_time_excluded_even_on_failure(self):
        t = [0.]
        clock = ActiveClock(lambda: t[0])
        t[0] = 1000.
        with self.assertRaises(ValueError):
            with clock.pause():
                t[0] += 3000.
                self.assertEqual(clock.elapsed(), 1000.)
                raise ValueError('observer failed')
        t[0] += 5.
        self.assertEqual(clock.elapsed(), 1005.)
        self.assertEqual(clock.excluded, 3000.)

    def test_missing_seed_and_negative_never_disappear(self):
        runs = [{'seed': 1, 'independent_audit_pass': True, 'physical_acceptance_pass': True,
                 'time_to_verified_accuracy_s': 4000.},
                {'seed': 2, 'independent_audit_pass': True, 'physical_acceptance_pass': False,
                 'right_censor_elapsed_s': 12000.}]
        result = cohort_result([1, 2, 3], runs)
        self.assertEqual((result['denominator'], result['passes']), (3, 1))
        self.assertIsNone(result['cohort_median_tta_s'])
        self.assertEqual(result['entries'][2]['state'], 'INCOMPLETE_OR_INVALID')
        with self.assertRaises(ValueError):
            cohort_result([1,2,3], runs+[runs[0]])

    def test_all_five_and_fixture_rejection(self):
        runs = [{'seed': i, 'independent_audit_pass': True, 'physical_acceptance_pass': True,
                 'time_to_verified_accuracy_s': float(4000+i)} for i in range(5)]
        result = cohort_result(list(range(5)), runs)
        self.assertEqual(result['state'], 'FORMAL_COHORT_PASS')
        self.assertEqual(result['cohort_median_tta_s'], 4002.)
        self.assertAlmostEqual(result['success_fraction_wilson95'][0], .5655175352)
        runs[0]['self_test'] = True
        with self.assertRaises(ValueError):
            cohort_result(list(range(5)), runs)

    def _tiny_run(self, immediate_pass=False):
        import torch
        from box376 import potential_train_tta as trainer
        from box376.data import make_data, torch_data
        from box376.normalized_objective import normalization_key
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root/'protocol-formal-tta-20261021.json').read_text())
        protocol['topology'] = [4, 3, 4]
        protocol['parameter_count'] = 31
        protocol['time_to_accuracy']['inspection_training_seconds'] = [0, 4000]
        calibration = json.loads((root/'data/calibration.json').read_text())
        data = torch_data(make_data(3))
        data['normalization_key'] = normalization_key(calibration)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p = d/'protocol.json'; p.write_text(json.dumps(protocol))
            argv = ['train','--inputs',str(root/'data'),'--protocol',str(p),'--output',str(d/'train'),
                    '--seed',str(protocol['seeds'][0]),'--arm','fresh','--self-test']
            from box376.tta_control import run_observer as real_observer
            def fixture(*args, **kwargs):
                if immediate_pass:
                    return {'physical_acceptance_pass': True, 'self_test_fixture_only': True, 'exit': 0}
                return real_observer(*args, **kwargs)
            with patch.object(sys,'argv',argv), patch.object(trainer,'_cpu_problem',return_value=(data,calibration)), patch.object(trainer,'run_observer',side_effect=fixture), contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            result = json.loads((d/'train/report.json').read_text())
            self.assertEqual(result['state'], 'COMPLETE')
            if immediate_pass:
                self.assertEqual(result['SSBroyden_iterations'],0)
                self.assertEqual(result['tta']['first_passing_tag'],'tta-first')
                self.assertEqual(len(result['observations']),1)
                return
            self.assertEqual(result['SSBroyden_iterations'], 3)
            self.assertEqual([x['physical_acceptance_pass'] for x in result['observations']], [False,True])
            self.assertTrue(all(x['parameters_and_optimizer_iteration_preserved'] for x in result['observations']))
            self.assertEqual(result['tta']['first_passing_tag'], 'final-observation')
            self.assertTrue(result['self_test'])

    def test_real_small_optimizer_continues_after_failed_inspection(self):
        self._tiny_run()

    def test_first_physical_pass_stops_optimizer_immediately(self):
        self._tiny_run(immediate_pass=True)


if __name__ == '__main__':
    unittest.main()
