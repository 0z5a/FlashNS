"""CPU control tests of the paired-engine trainer (tiny network fixtures, not physical evidence)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class PairTrainer(unittest.TestCase):
    def _tiny_run(self, engine, threshold):
        from box376 import potential_train_pair as trainer
        from box376.data import make_data, torch_data
        from box376.normalized_objective import normalization_key
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root/'protocol-formal-pair-20261021.json').read_text())
        protocol['topology'] = [4, 3, 4]
        protocol['parameter_count'] = 31
        protocol['pair']['threshold_loss'] = threshold
        calibration = json.loads((root/'data/calibration.json').read_text())
        data = torch_data(make_data(3))
        data['normalization_key'] = normalization_key(calibration)
        calls = []

        def observer(*args, **kwargs):
            calls.append(Path(args[3]).name)
            return {'physical_acceptance_pass': True, 'self_test_fixture_only': True, 'exit': 0}

        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p = d/'protocol.json'; p.write_text(json.dumps(protocol))
            argv = ['train', '--inputs', str(root/'data'), '--protocol', str(p), '--output', str(d/engine/'train'),
                    '--seed', str(protocol['seeds'][0]), '--arm', 'fresh', '--engine', engine, '--self-test']
            (d/engine).mkdir()
            with patch.object(sys, 'argv', argv), patch.object(trainer, '_cpu_problem', return_value=(data, calibration)), \
                    patch.object(trainer, 'run_observer', side_effect=observer), contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            return json.loads((d/engine/'train/report.json').read_text()), calls

    def test_threshold_stops_and_observes_once(self):
        for engine in ('analytic', 'parameter_ad'):
            report, calls = self._tiny_run(engine, 1e9)
            self.assertEqual(report['state'], 'COMPLETE')
            self.assertEqual(report['engine'], engine)
            self.assertEqual(report['pair']['state'], 'THRESHOLD_REACHED')
            self.assertEqual(report['SSBroyden_iterations'], 1)
            self.assertEqual(report['pair']['SSBroyden_iterations'], 1)
            self.assertEqual(report['stop_reason'], 'loss threshold')
            self.assertEqual(calls, ['physical-threshold'])
            self.assertTrue(report['pair']['physical_acceptance_pass'])
            self.assertIn('threshold', {c['tag'] for c in report['checks']})
            self.assertLessEqual(report['pair']['active_training_seconds'], report['training_seconds'])

    def test_missed_threshold_is_not_observed(self):
        report, calls = self._tiny_run('parameter_ad', 1e-300)
        self.assertEqual(report['pair']['state'], 'NOT_REACHED_WITHIN_BUDGET')
        self.assertEqual(report['SSBroyden_iterations'], 3)
        self.assertEqual(calls, [])
        self.assertEqual(report['observations'], [])
        self.assertIn('final', {c['tag'] for c in report['checks']})

    def test_pair_finalizer_archives_failure_before_driver_start(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); packet = d/'packet'; packet.mkdir()
            (packet/'protocol.json').write_text('{"frozen":true}')
            root = d/'seed1'; (root/'control').mkdir(parents=True)
            (root/'batch.log').write_text('admission failed')
            p = subprocess.run([sys.executable, '-m', 'box376.finalize_c36_pair', str(root), str(packet), 'fixture', '1'],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(json.loads((d/'archives/seed1-jobfixture.json').read_text())['outer_failure_archive'])


if __name__ == '__main__':
    unittest.main()
