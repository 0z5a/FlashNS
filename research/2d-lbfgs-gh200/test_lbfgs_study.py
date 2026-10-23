"""CPU checks of the 2D L-BFGS GH200 study: the source rewrite, the plan builder, the reader's
decisions on synthetic rows and the driver's control flow in self-test mode."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
import lbfgs_light  # noqa: E402
import make_lbfgs_plan  # noqa: E402
import read_lbfgs_study as reader  # noqa: E402

PROTOCOL = json.loads((HERE / 'protocol-2d-lbfgs-gh200-20261023.json').read_text())


def row(sha, evals=7000, iters=3000, opt=10.0, setup=0.4, status='passed', converged=True):
    return {'status': status, 'converged': converged,
            'identity': {'parameter_final_sha256': sha, 'loss_gradient_evaluations': evals, 'lbfgs_iterations': iters, 'adam_updates': 4000, 'converged': converged},
            'setup_seconds': setup, 'optimization_seconds': opt, 'adam_ms_per_evaluation': 0.2, 'lbfgs_ms_per_evaluation': 5.0,
            'light': None, 'lbfgs_module_sha256': 'x'}


class Rewrite(unittest.TestCase):
    def test_installed_torch_two_loop_is_replaced(self):
        try:
            original, rewritten = lbfgs_light.rewrite()
        except RuntimeError as error:  # another torch release: the study's runtime pin would refuse it
            self.skipTest(str(error))
        self.assertEqual(rewritten.count('self._two_loop('), 1)
        self.assertNotIn('alpha=-al[i]', rewritten)
        self.assertNotIn('alpha=al[i] - be_i', rewritten)
        self.assertIn('alpha=-al[i]', original)
        self.assertGreater(len(original) - len(rewritten), 200)

    def test_foreign_source_is_refused(self):
        with self.assertRaises(RuntimeError):
            lbfgs_light.rewrite('def step(self, closure):\n    return closure()\n')


class Wrapper(unittest.TestCase):
    def test_light_solve_imports_outside_the_source_tree(self):
        """light_solve.py's module level (imports and the lbfgs module hash) with the 2D modules stubbed."""
        import importlib
        import types
        stubs = {name: types.ModuleType(name) for name in ('bench_gpu', 'problem', 'solve')}
        stubs['bench_gpu'].BACKENDS = ('gl8_native',)
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            sys.modules.pop('light_solve', None)
            module = importlib.import_module('light_solve')
            self.assertEqual(len(module.LBFGS_MODULE_SHA256), 64)
            self.assertEqual(module.LBFGS_MODULE_SHA256, lbfgs_light.module_source_sha256())
        finally:
            for name, old in saved.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
            sys.modules.pop('light_solve', None)


class Plan(unittest.TestCase):
    def test_tasks(self):
        tasks = make_lbfgs_plan.build(PROTOCOL)
        solves = [t for t in tasks if t['tier'] == 0]
        drift = [t for t in tasks if t['tier'] == 1]
        self.assertEqual((len(solves), len(drift)), (36, 4))
        self.assertEqual(len({t['id'] for t in solves}), 36)
        self.assertEqual([d['repeat_of'] for d in drift], [t['id'] for t in solves[:4]])
        self.assertEqual(tasks, make_lbfgs_plan.build(PROTOCOL))
        self.assertNotEqual([t['id'] for t in solves], sorted(t['id'] for t in solves))
        for t in tasks:
            self.assertEqual(t['argv'][:3], ['light_solve.py', '--optimizer', t['cell']])


class Decisions(unittest.TestCase):
    def pairs(self, ratios, differ=(), accepted=6):
        out = {}
        for i, seed in enumerate(PROTOCOL['seeds']):
            sha = f'sha{seed}'
            stock = row(sha, opt=10.0 * ratios[i], status='passed' if i < accepted else 'failed', converged=i < accepted)
            light = row(sha + ('-different' if seed in differ else ''), opt=10.0)
            out[seed] = (stock, light)
        return out

    def test_gain_identical(self):
        d = reader.decide_size(self.pairs([2.5, 2.6, 2.4, 2.8, 2.7, 2.5]))
        self.assertEqual(d['decision'], 'EXECUTION_GAIN_IDENTICAL')
        self.assertEqual((d['identical_pairs'], d['pairs_favouring_light']), (6, 6))
        self.assertAlmostEqual(d['optimization_stock_over_light']['geometric_mean'], 2.58, places=1)

    def test_no_gain(self):
        d = reader.decide_size(self.pairs([1.01, 1.02, 0.99, 1.0, 1.01, 1.02]))
        self.assertEqual(d['decision'], 'IDENTICAL_NO_GAIN')

    def test_differing_trajectory_blocks_the_claim(self):
        d = reader.decide_size(self.pairs([2.5] * 6, differ=(862103,)))
        self.assertEqual(d['decision'], 'TRAJECTORY_DIFFERS')
        self.assertEqual(d['differing_seeds'], [862103])

    def test_not_decidable(self):
        d = reader.decide_size(self.pairs([2.5] * 6, accepted=4))
        self.assertEqual(d['decision'], 'NOT_DECIDABLE')

    def test_drift_and_validity(self):
        plan = {'tasks': make_lbfgs_plan.build(PROTOCOL)}
        rows = {}
        for t in plan['tasks']:
            opt = 10.0 if t['cell'] == 'light' else 25.0
            rows[t['id']] = row(f"sha-n{t['interior']}-s{t['seed']}", opt=opt * (1.04 if t['tier'] == 1 else 1.0))
        result = reader.analyse(PROTOCOL, plan, rows)
        self.assertTrue(result['timing_valid'])
        self.assertEqual(set(result['decisions'].values()), {'EXECUTION_GAIN_IDENTICAL'})
        bad = plan['tasks'][-1]
        rows[bad['id']] = row('sha-other', opt=10.0)
        result = reader.analyse(PROTOCOL, plan, rows)
        self.assertFalse(result['timing_valid'])
        self.assertTrue(all(v.endswith('(NOT_VALID timing)') for v in result['decisions'].values()))


class Driver(unittest.TestCase):
    def test_self_test_control_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = root / 'lbfgs'
            study.mkdir()
            (root / 'source').mkdir()
            subprocess.run(['git', 'init', '-q'], cwd=root / 'source', check=True)
            subprocess.run(['git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '--allow-empty', '-m', 'x'], cwd=root / 'source', check=True)
            head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root / 'source', capture_output=True, text=True).stdout.strip()
            bench = root / 'source' / 'experiments' / 'contracted_jets'
            bench.mkdir(parents=True)
            for name in ('light_solve.py', 'lbfgs_light.py'):
                (bench / name).write_text((HERE / 'src' / name).read_text())
            protocol = dict(PROTOCOL, source=dict(PROTOCOL['source'], commit=head))
            (study / 'protocol.json').write_text(json.dumps(protocol))
            plan = {'protocol': 'protocol.json', 'protocol_sha256': reader_sha(study / 'protocol.json'), 'tasks': make_lbfgs_plan.build(protocol)}
            (study / 'plan.json').write_text(json.dumps(plan))
            for name in ('read_lbfgs_study.py', 'run_lbfgs_study.py'):
                (study / name).write_text((HERE / name).read_text())
            files = {str(p.relative_to(root)): {'sha256': reader_sha(p)} for p in list(study.iterdir()) + list(bench.iterdir())}
            (study / 'manifest.json').write_text(json.dumps({'files': files}))
            code = subprocess.run([sys.executable, str(study / 'run_lbfgs_study.py'), '--root', str(root), '--plan', 'plan.json', '--job', 'x',
                                   '--node', 'y', '--deadline-unix', str(1e12), '--python', sys.executable, '--self-test'],
                                  capture_output=True, text=True)
            state = json.loads((study / 'control' / 'driver-state.json').read_text())
            self.assertEqual(code.returncode, 0, (code.stdout[-500:], code.stderr[-800:], state.get('state')))
            self.assertEqual(state['state'], 'COMPLETE')
            self.assertEqual(state['counts'], {'COMPLETE': 40})
            self.assertEqual(len(state['lanes']), 4)
            result = json.loads((study / 'lbfgs-result.json').read_text())
            self.assertEqual(set(result['decisions'].values()), {'NOT_DECIDABLE (NOT_VALID timing)'})
            self.assertTrue(state['archive']['members_match_sources'] and state['archive']['whole_file_sha_reread'])


def reader_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    unittest.main()
