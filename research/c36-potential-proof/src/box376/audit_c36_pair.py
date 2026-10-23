"""Independent CPU readback of one paired-engine C36 seed archive (both arms). No GPU allocation.

Every per-arm check is that of box376.audit_c36_tta_v2, applied to each arm directory; the TTA
checks at the end are replaced by threshold checks: the recorded threshold iterate is the first
accepted SSBroyden iterate at or below the registered loss, training stopped there, its
measurement precedes any later event, and an arm that misses the threshold has no observation.
The text below describes revision 2 of box376.audit_c36_tta, on which this auditor is built.

Revision 2 of box376.audit_c36_tta. Revision 1 compared the physical consumer's summed raw arrays,
recomputed on the auditing host, with the values saved on the GPU host under rtol=1e-10, a tolerance
relative to the result. These arrays are sums of terms up to about 1e5 that cancel to about 1e-2, so
two hosts that round the terms differently (ARM GPU host, x86 auditor) can differ by more than that.
Revision 2 replaces only those comparisons by the a-priori bound for two independently rounded
evaluations, 2*gamma_n*S elementwise, where S is the sum of the absolute values of the terms; the
scalar metrics built from these arrays get the bound that follows from it by the triangle
inequality. The protocol file and the output directory are explicit arguments, so the auditor never
writes into another cohort's audit directory. Every other check is unchanged.
"""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np

U = 2.0 ** -53
ROUNDINGS = 16  # a-priori upper bound on the roundings that enter any element of a summed raw array
GAMMA = ROUNDINGS * U / (1 - ROUNDINGS * U)
# scalar metric -> (summed raw array it is the weighted RMS of, calibration scale)
METRIC_ARRAY = {'normalized_momentum_rms': ('residual', 'momentum_residual'),
                'normalized_momentum_gradient_rms': ('spatial_residual_gradient', 'momentum_residual_gradient'),
                'normalized_divergence_rms': ('divergence', 'continuity_residual'),
                'normalized_divergence_gradient_rms': ('divergence_gradient', 'continuity_residual_gradient')}
# The exact expected manifest digest is supplied from the local frozen packet at invocation.
LIMITS = {'velocity_relative_l2': .01, 'normalized_divergence_rms': .001,
          'normalized_momentum_rms': .01, 'normalized_momentum_gradient_rms': .02,
          'normalized_divergence_gradient_rms': .02,
          'normalized_corrected_pressure_rms': .001, 'normalized_pressure_gauge_rms': .001}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def same(a, b, label):
    if isinstance(a, dict):
        assert set(a) == set(b), label
        for k in a:
            same(a[k], b[k], label + '/' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, label + '/' + str(i))
    elif a is None or isinstance(a, (str, bool)):
        assert a == b, (label, a, b)
    else:
        assert np.isfinite(a) and np.isfinite(b), label
        assert np.isclose(a, b, rtol=1e-10, atol=1e-15), (label, a, b)


def wrms(a, w):
    a = np.asarray(a)
    assert np.isfinite(a).all() and np.isfinite(w).all()
    assert np.all(w >= 0) and w.sum() > 0
    return float(np.sqrt(np.sum(w * np.mean(a.reshape(len(a), -1)**2, axis=1)) / w.sum()))


def metrics(state, reference, w, gauge, scales):
    assert state.shape == (len(w), 4, 17) and state.dtype == np.float64
    assert np.isfinite(state).all() and np.isfinite(gauge).all()
    h = np.empty((len(w), 4, 3, 3))
    for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        h[:, :, i, j] = h[:, :, j, i] = state[:, :, 4+k]
    u, g = state[:, :3, 0], state[:, :3, 1:4]
    base_inertia = state[:, :3, 13] + np.einsum('nj,nij->ni', u, g) - .01*np.trace(h[:, :3], axis1=2, axis2=3)
    inertia = base_inertia - reference['f']
    residual = base_inertia + state[:, 3, 1:4] - reference['f']
    spatial = (state[:, :3, 14:17] + np.einsum('nij,njk->nik', g, g)
               + np.einsum('nj,nijk->nik', u, h[:, :3]) - .01*state[:, :3, 10:13] + h[:, 3] - reference['df'])
    div = np.trace(g, axis1=1, axis2=2)
    dg = sum(h[:, i, i, :] for i in range(3))
    delta = u-reference['U']; denom = wrms(reference['U'], w)
    m = {'velocity_relative_l2': wrms(delta, w)/denom if denom > 0 else None,
         'velocity_reference_is_zero': denom == 0,
         'normalized_velocity_error_rms': wrms(delta, w)/scales['velocity'],
         'normalized_divergence_rms': wrms(div, w)/scales['continuity_residual'],
         'normalized_divergence_gradient_rms': wrms(dg, w)/scales['continuity_residual_gradient'],
         'normalized_momentum_rms': wrms(residual, w)/scales['momentum_residual'],
         'normalized_momentum_gradient_rms': wrms(spatial, w)/scales['momentum_residual_gradient'],
         'normalized_corrected_pressure_rms': wrms(state[:, 3, 0]-gauge, w)/scales['pressure'],
         'normalized_pressure_gauge_rms': wrms(gauge, w)/scales['pressure'],
         'velocity_component_error_rms': [wrms(delta[:, i], w) for i in range(3)],
         'velocity_component_relative_l2': [wrms(delta[:, i], w)/wrms(reference['U'][:, i], w) if wrms(reference['U'][:, i], w) else None for i in range(3)]}
    raw = {'residual': residual, 'spatial_residual_gradient': spatial, 'divergence': div, 'divergence_gradient': dg}
    return m, raw, inertia


def rounding_bounds(state, reference):
    """Elementwise 2*gamma*S for each summed raw array of metrics(): the most two independently rounded evaluations can differ."""
    h = np.empty((len(state), 4, 3, 3))
    for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        h[:, :, i, j] = h[:, :, j, i] = np.abs(state[:, :, 4+k])
    u, g = np.abs(state[:, :3, 0]), np.abs(state[:, :3, 1:4])
    inertia = np.abs(state[:, :3, 13]) + np.einsum('nj,nij->ni', u, g) + .01*np.trace(h[:, :3], axis1=2, axis2=3)
    S = {'residual': inertia + np.abs(state[:, 3, 1:4]) + np.abs(reference['f']),
         'spatial_residual_gradient': (np.abs(state[:, :3, 14:17]) + np.einsum('nij,njk->nik', g, g)
                                       + np.einsum('nj,nijk->nik', u, h[:, :3]) + .01*np.abs(state[:, :3, 10:13]) + h[:, 3] + np.abs(reference['df'])),
         'divergence': np.trace(g, axis1=1, axis2=2),
         'divergence_gradient': sum(h[:, i, i, :] for i in range(3))}
    return {k: 2*GAMMA*v for k, v in S.items()}


def same_metrics(m, reported, bounds, w, scales, label):
    """same() for every metric, except that metrics built from summed raw arrays may differ by the bound
    implied by the elementwise one: |wrms(a)-wrms(b)| <= wrms(a-b) <= wrms(bound), plus the rounding of the RMS itself."""
    assert set(m) == set(reported), label
    worst = 0.0
    for k in m:
        if k in METRIC_ARRAY and m[k] is not None:
            array, scale = METRIC_ARRAY[k]
            tol = wrms(bounds[array], w)/scales[scale] + 1e-12*abs(reported[k])
            assert np.isfinite(m[k]) and abs(m[k]-reported[k]) <= tol, (label+'/'+k, m[k], reported[k], tol)
            worst = max(worst, abs(m[k]-reported[k])/tol if tol > 0 else 0.0)
        else:
            same(m[k], reported[k], label+'/'+k)
    return worst


def gate_ratios(m):
    return {k: (m['normalized_velocity_error_rms']/.001 if k == 'velocity_relative_l2' and m['velocity_reference_is_zero'] else m[k]/limit) for k, limit in LIMITS.items()}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--manifest-sha',required=True)
    p.add_argument('--protocol-sha',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--protocol-name',required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    args=p.parse_args()
    ROOT=args.root; PROTOCOL_SHA=args.protocol_sha; expected_manifest=args.manifest_sha; PROTOCOL=args.protocol_name
    stem=f'seed{args.seed}-job{args.job}'
    receipt=json.loads((ROOT/'archives'/(stem+'.json')).read_text())
    archive=ROOT/'archives'/(stem+'.tar.gz')
    assert sha(archive) == receipt['sha256'] and archive.stat().st_size == receipt['bytes']
    output = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'state': 'AUDITING', 'cpu_only': True, 'archive': receipt,
              'audit_script_sha256': sha(__file__), 'numpy_version': np.__version__,
              'limitations': ['Readback verifies saved native AD arrays; it does not rerun GPU derivatives or optimizer.',
                             'TTA is confirmation latency under the frozen inspection policy; the first physical crossing is unknown. No baseline speedup claim.']}
    audit_root = args.audit_dir; audit_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='readback-', dir=audit_root) as tmp:
        tmp = Path(tmp)
        hashes = {}; live_differences = []
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            assert len(members) == receipt['members'] == len({m.name for m in members})
            for m in members:
                assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
                f = tar.extractfile(m)
                hashes[m.name] = hashlib.file_digest(f, 'sha256').hexdigest()
            tar.extractall(tmp, filter='data')
        for name, value in hashes.items():
            prefix, relative = name.split('/',1)
            assert prefix in ('packet','run')
            p = (ROOT/'packet' if prefix=='packet' else ROOT/('seed'+str(args.seed)))/relative
            if not p.is_file() or sha(p) != value:
                live_differences.append(name)
        # The finalizer prints its receipt only after sealing the archive.
        assert set(live_differences) <= {'run/control/driver-state.json','run/control/batch-exit.json','run/batch.log'}, live_differences
        assert hashes == receipt['member_sha256']
        (audit_root/(stem+'-member-sha256.json')).write_text(json.dumps(hashes, indent=2)+'\n')
        packet = tmp/'packet'
        assert sha(packet/'manifest.json') == expected_manifest
        manifest = json.loads((packet/'manifest.json').read_text())
        assert manifest['source_commit'] == args.source_commit
        for name, row in manifest['files'].items():
            assert sha(packet/name) == row['sha256'], name
        protocol = json.loads((packet/PROTOCOL).read_text())
        assert sha(packet/PROTOCOL) == PROTOCOL_SHA
        assert protocol['atol'] == 1e-11 and protocol['rtol'] == 1e-9
        assert protocol['training']['maximum_training_seconds'] == 4000
        output['integrity'] = {'archive_sha_verified': True, 'members_read': len(hashes),
            'all_members_compared_to_live_sources': True, 'expected_postseal_differences': live_differences,
            'frozen_files_verified': len(manifest['files']), 'manifest_sha256': expected_manifest,
            'protocol_sha256': PROTOCOL_SHA, 'source_commit': manifest['source_commit']}
        sys.path.insert(0, str(packet/'src'))
        from box376.read_c36_v7 import read_run
        from box376.source import C0, H0, trajectory
        # Frozen sampler is NumPy-only except the digest import from the Torch-containing data module.
        sampler_path = packet/'src/box376/validation_data.py'
        tree = ast.parse(sampler_path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'box376.data')]
        ns = {'__file__': str(sampler_path), 'digest': digest, 'MATH_COMMIT': 'adc7f1241b42e322a6451854ab7e4b4c146bf78a'}
        exec(compile(tree, str(sampler_path), 'exec'), ns)
        datasets, regenerated, _ = ns['make_holdouts'](seed=protocol['physical']['holdout_seed'], eta=protocol['instance']['eta'])
        scales = json.loads((packet/'data/calibration.json').read_text())['candidate_fixed_physical_scales']
        runs = []; physical_audits = []
        order = protocol['pair']['arm_order'][str(args.seed)]
        for arm in order:
            directory = tmp/'run'/arm
            run = read_run(directory, protocol)
            runs.append(run)
            training = json.loads((directory/'train/report.json').read_text())
            assert training['state'] == 'COMPLETE' and not training['self_test']
            assert training['seed']==args.seed and training['protocol_sha256']==PROTOCOL_SHA
            assert training['engine'] == arm and training['pair']['state'] in ('THRESHOLD_REACHED','NOT_REACHED_WITHIN_BUDGET')
            reached = training['pair']['state'] == 'THRESHOLD_REACHED'
            assert training['Adam_updates'] == protocol['training']['Adam_updates_per_seed'] and run['events_sealed']
            tags={c['tag'] for c in training['checks']}
            required = {'eager-fresh-denseAD','compiled-fresh-denseAD','graph-fresh-denseAD','Adam0','threshold' if reached else 'final'}
            if arm == 'analytic' and protocol['pair']['native_subset_in_analytic_arm']:
                required.add('fresh-native-subset')
            assert required <= tags, (arm, required - tags)
            assert len(training['observations']) == (1 if reached else 0)
            for check in training['checks']:
                with np.load(directory/'train'/check['filename'], allow_pickle=False) as z:
                    for comparison in check['comparisons']:
                        x, y = z[comparison['name']+'_actual'], z[comparison['name']+'_reference']
                        assert x.dtype == y.dtype == np.float64 and x.shape == y.shape
                        assert np.isfinite(x).all() and np.isfinite(y).all()
                        if comparison['name']=='gradient':assert x.shape==(50692,)
            for path in sorted(directory.glob('physical-*/report.json')):
                report = json.loads(path.read_text())
                if report['state'] != 'COMPLETE':
                    physical_audits.append({'run': directory.name, 'tag': path.parent.name, 'state': report['state']}); continue
                assert report['protocol_sha256'] == PROTOCOL_SHA
                tag = path.parent.name.removeprefix('physical-')
                checkpoint = directory/'train'/f"seed{training['seed']}-fresh-{tag}.pt"
                assert sha(checkpoint) == report['checkpoint_sha256']
                with np.load(path.parent/'raw-physical.npz', allow_pickle=False) as z:
                    assert z['theta'].shape == (50692,) and np.isfinite(z['theta']).all()
                    with np.load(directory/'train'/('check-'+tag+'.npz'),allow_pickle=False) as c:
                        assert np.array_equal(z['theta'],c['theta'])
                    for name, value in report['validation_metadata']['array_hashes'].items():
                        assert digest(z[name]) == value, name
                    regeneration_errors = {}
                    for name, value in regenerated.items():
                        assert z[name].shape == value.shape and z[name].dtype == value.dtype
                        if value.dtype.kind == 'f':
                            error = np.abs(z[name]-value)
                            ratio = float(np.max(error/(protocol['atol']+protocol['rtol']*np.abs(z[name]))))
                            regeneration_errors[name] = {'max_abs':float(error.max()),'max_original_budget_ratio':ratio}
                            assert np.isfinite(ratio) and ratio <= 1, ('regenerated holdout',name,regeneration_errors[name])
                        else:
                            assert np.array_equal(z[name],value), ('regenerated discrete holdout',name)
                    checks = {}; summary = {}; pressure = {}; rounding_margins = {}
                    for name, dataset in datasets.items():
                        state, gauge, w = z[name+'_native_state'], z[name+'_gauge'], z[name+'_weights']
                        reference = {k:z[name+'_source_'+k] for k in dataset['reference']}
                        m, raw, inertia = metrics(state, reference, w, gauge, scales)
                        bounds = rounding_bounds(state, reference)
                        worst_metric = same_metrics(m, report['datasets'][name]['global'], bounds, w, scales, name)
                        worst_array = 0.0
                        for key, array in raw.items():
                            err = np.abs(array - z[name+'_'+key])
                            assert np.all(err <= bounds[key]), (name, key, float(np.max(err - bounds[key])))
                            if err.any():
                                worst_array = max(worst_array, float(np.max(err[err > 0]/bounds[key][err > 0])))
                        for region in np.unique(dataset['regions']):
                            selected = dataset['regions'] == region
                            regional, _, _ = metrics(state[selected], {k: v[selected] for k, v in reference.items()}, w[selected], gauge[selected], scales)
                            worst_metric = max(worst_metric, same_metrics(regional, report['datasets'][name]['regions'][str(region)],
                                                                          {k: v[selected] for k, v in bounds.items()}, w[selected], scales, name+'/region'))
                        rounding_margins[name] = {'raw_array_error_over_bound': worst_array, 'metric_difference_over_bound': worst_metric}
                        ratios = gate_ratios(m)
                        assert (max(ratios.values()) <= 1) == report['datasets'][name]['field_gate_pass']
                        summary[name] = m
                        if name in protocol['physical']['gated_datasets']:
                            checks.update({name+':'+k: v for k, v in ratios.items()})
                        if name == 'stratified':
                            for region, label in enumerate(('plateau','face','edge','corner')):
                                sel = (dataset['regions'] == region) & (dataset['time_regions'] == 0)
                                weights = w[sel]; gp = state[sel, 3, 1:4]; v = -inertia[sel]; rr = raw['residual'][sel]
                                def inner(a, b): return float(np.sum(weights*np.sum(a*b, axis=1)))
                                pressure[label] = {'cosine_pressure_velocity': inner(gp,v)/np.sqrt(inner(gp,gp)*inner(v,v)) if inner(gp,gp)*inner(v,v)>0 else None,
                                    'pressure_fraction_of_residual_energy': inner(gp,gp)/inner(rr,rr) if inner(rr,rr)>0 else None,
                                    'corrected_pressure_over_limit': wrms(state[sel,3,0]-gauge[sel],weights)/scales['pressure']/.001,
                                    'velocity_over_pressure_gradient': wrms(v,weights)/wrms(gp,weights) if wrms(gp,weights)>0 else None,
                                    'cosine_noise_floor': 1/np.sqrt(gp.size)}
                    for name in ('initial','boundary'):
                        values = z[name+'_native_values']
                        value = wrms(values[:, :3], np.ones(len(values)))/scales['velocity']
                        same(value, report[name+'_normalized_velocity_rms'], name)
                        checks[name] = value/protocol['physical']['initial_boundary_velocity_normalized_rms_limit']
                    import itertools
                    initial = np.array(list(itertools.product((-1.,0.,1.),repeat=3)))*H0+C0
                    times = np.unique(np.r_[np.linspace(0,1,129),.25,.375,.5,.625,.75])
                    assert np.array_equal(initial,z['particle_initial']) and np.array_equal(times,z['particle_times'])
                    exact = trajectory(times[:,None],initial[None,:,:])
                    assert np.allclose(exact,z['particle_exact'],rtol=protocol['rtol'],atol=protocol['atol'])
                    exact = z['particle_exact']
                    for i, record in enumerate(report['trajectories']):
                        actual = z[f'particle_integrated{i}']; assert np.isfinite(actual).all()
                        err = np.linalg.norm(actual-exact,axis=-1)
                        same(float(err.max()),record['maximum_path_error'],'path error')
                        same(float(err[-1].max()),record['maximum_endpoint_error'],'endpoint error')
                        assert [record[k] for k in ('rtol','atol','max_step')] == protocol['physical']['trajectory_refinements'][i]
                        if i:
                            change = float(np.linalg.norm(actual-z[f'particle_integrated{i-1}'],axis=-1).max())
                            same(change,record['maximum_refinement_change'],'refinement')
                    checks['trajectory_endpoint'] = float(err[-1].max())/protocol['physical']['maximum_trajectory_endpoint_error']
                    checks['trajectory_refinement'] = change/protocol['physical']['maximum_trajectory_refinement_change']
                    passed = max(checks.values()) <= 1
                    assert passed == report['physical_acceptance_pass']
                    physical_audits.append({'run':directory.name,'tag':tag,'state':'COMPLETE',
                        'raw_hash_verified':True,'holdout_arrays_regenerated':len(regenerated),
                        'cross_architecture_regeneration_errors':regeneration_errors,
                        'all_global_and_regional_metrics_recomputed':True,'gate_ratios':checks,
                        'physical_acceptance_pass':passed,'datasets':summary,
                        'trajectory':report['trajectories'],'pressure_attribution':pressure,
                        'rounding_margins':rounding_margins})
        entries = {}
        for arm in order:
            directory = tmp/'run'/arm
            training = json.loads((directory/'train/report.json').read_text())
            pair = training['pair']; reached = pair['state'] == 'THRESHOLD_REACHED'
            assert pair['threshold_loss'] == protocol['pair']['threshold_loss'] and pair['engine'] == arm
            observations = training['observations']
            audits = [p for p in physical_audits if p['run'] == arm]
            assert {o['tag'] for o in observations} == {p['tag'] for p in audits} == ({'threshold'} if reached else set())
            for o in observations:
                q = audits[0]
                assert q['state'] == 'COMPLETE' and q['physical_acceptance_pass'] == o['physical_acceptance_pass'] == pair['physical_acceptance_pass']
                path = directory/('physical-'+o['tag'])
                assert sha(path/'report.json') == o['report_sha256'] and sha(path/'raw-physical.npz') == o['raw_sha256']
                assert o['parameters_and_optimizer_iteration_preserved'] and not o.get('self_test_fixture_only', False)
            iterations = [json.loads(l) for l in (directory/'train/iterations.jsonl').read_text().splitlines() if l.strip()]
            events = [json.loads(l) for l in (directory/'train/events.jsonl').read_text().splitlines() if l.strip()]
            first = next((r['iteration'] for r in iterations if r['loss'] <= pair['threshold_loss']), None)
            if reached:
                assert first == pair['SSBroyden_iterations'] == training['SSBroyden_iterations'] == len(iterations)
                assert pair['SSBroyden_evaluations'] == training['SSBroyden_evaluations'] and pair['Adam_updates'] == training['Adam_updates']
                assert training['stop_reason'] == 'loss threshold' and pair['loss'] == iterations[-1]['loss']
                assert events[-1]['phase'] == 'SSBroyden' and events[-1]['training_seconds'] <= pair['active_training_seconds']
                # The guard is tested before each step, so the last step may end just past it.
                assert 0 < pair['active_training_seconds'] <= protocol['pair']['maximum_training_seconds'] + 60
            else:
                assert first is None and (training['budget_truncated'] or training['stop_reason'] in ('iteration cap','two consecutive resets'))
            entries[arm] = {'seed': args.seed, 'arm': arm, 'independent_audit_pass': True, 'self_test': False,
                            'threshold_reached': reached, 'active_training_seconds': pair.get('active_training_seconds'),
                            'physical_acceptance_pass': bool(pair.get('physical_acceptance_pass')),
                            'SSBroyden_iterations': training['SSBroyden_iterations'], 'SSBroyden_evaluations': training['SSBroyden_evaluations']}
        entry = {'seed': args.seed, 'arms': entries}
        output.update(state='PASS_INDEPENDENT_ARCHIVE_READBACK',training_reader={'runs':runs,'protocol_sha256':PROTOCOL_SHA},
                      physical_audits=physical_audits,cohort_entry=entry,
                      driver=json.loads((tmp/'run/control/driver-state.json').read_text()))
    output['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = audit_root/(stem+'-independent-readback.json')
    assert not out.exists(), out
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':output['state'],'output':str(out),'sha256':sha(out),
                      'arms':{a:(e['threshold_reached'],e['active_training_seconds'],e['physical_acceptance_pass']) for a,e in output['cohort_entry']['arms'].items()},'integrity':output['integrity']}))


if __name__ == '__main__':
    main()
