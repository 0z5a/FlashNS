"""Independent CPU readback of a prospective C36 TTA seed archive. No GPU allocation."""
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

PROTOCOL = 'protocol-formal-tta-20261021.json'
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
    args=p.parse_args()
    ROOT=args.root; PROTOCOL_SHA=args.protocol_sha; expected_manifest=args.manifest_sha
    stem=f'seed{args.seed}-job{args.job}'
    receipt=json.loads((ROOT/'archives'/(stem+'.json')).read_text())
    archive=ROOT/'archives'/(stem+'.tar.gz')
    assert sha(archive) == receipt['sha256'] and archive.stat().st_size == receipt['bytes']
    output = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'state': 'AUDITING', 'cpu_only': True, 'archive': receipt,
              'audit_script_sha256': sha(__file__), 'numpy_version': np.__version__,
              'limitations': ['Readback verifies saved native AD arrays; it does not rerun GPU derivatives or optimizer.',
                             'TTA is confirmation latency under the frozen inspection policy; the first physical crossing is unknown. No baseline speedup claim.']}
    audit_root = ROOT/'audit'; audit_root.mkdir(exist_ok=True)
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
        for directory in [tmp/'run']:
            run = read_run(directory, protocol)
            runs.append(run)
            training = json.loads((directory/'train/report.json').read_text())
            assert training['state'] == 'COMPLETE' and not training['self_test']
            assert training['seed']==args.seed and training['protocol_sha256']==PROTOCOL_SHA
            assert training['tta']['state'] in ('PHYSICALLY_CONFIRMED','NOT_REACHED_WITHIN_BUDGET')
            assert training['Adam_updates'] == 4000 and run['events_sealed']
            tags={c['tag'] for c in training['checks']}
            assert {'fresh-native-subset','eager-fresh-denseAD','compiled-fresh-denseAD','graph-fresh-denseAD','Adam0','final'} <= tags
            assert 1 <= len(training['observations']) <= 2
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
                    checks = {}; summary = {}; pressure = {}
                    for name, dataset in datasets.items():
                        state, gauge, w = z[name+'_native_state'], z[name+'_gauge'], z[name+'_weights']
                        reference = {k:z[name+'_source_'+k] for k in dataset['reference']}
                        m, raw, inertia = metrics(state, reference, w, gauge, scales)
                        same(m, report['datasets'][name]['global'], name)
                        for key, array in raw.items():
                            assert np.allclose(array, z[name+'_'+key], rtol=1e-10, atol=1e-13), (name, key)
                        for region in np.unique(dataset['regions']):
                            selected = dataset['regions'] == region
                            regional, _, _ = metrics(state[selected], {k: v[selected] for k, v in reference.items()}, w[selected], gauge[selected], scales)
                            same(regional, report['datasets'][name]['regions'][str(region)], name+'/region')
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
                        'trajectory':report['trajectories'],'pressure_attribution':pressure})
        observations=training['observations']
        assert {r['tag'] for r in observations} == {p['tag'] for p in physical_audits}
        assert len(observations)==len(physical_audits)
        for i,o in enumerate(observations):
            q=next(p for p in physical_audits if p['tag']==o['tag'])
            assert q['state']=='COMPLETE' and q['physical_acceptance_pass']==o['physical_acceptance_pass']
            path=tmp/'run'/('physical-'+o['tag'])
            assert sha(path/'report.json')==o['report_sha256']
            assert sha(path/'raw-physical.npz')==o['raw_sha256']
            assert o['parameters_and_optimizer_iteration_preserved'] and not o.get('self_test_fixture_only',False)
            if o['tag']=='tta-first':
                assert i==0 and o['active_training_seconds'] >= protocol['time_to_accuracy']['inspection_training_seconds'][0]
            if i<len(observations)-1: assert not o['physical_acceptance_pass']
        passed=any(o['physical_acceptance_pass'] for o in observations)
        tta=training['tta']; assert passed==(tta['state']=='PHYSICALLY_CONFIRMED')
        if passed:
            last=observations[-1]
            assert last['tag']==tta['first_passing_tag'] and last['checkpoint_sha256']==tta['checkpoint_sha256']
            delta=tta['confirmation_unix']-training['started_unix']
            assert abs(delta-tta['time_to_verified_accuracy_s']) < 1
            assert training['finished_unix']>=tta['confirmation_unix']>=last['finished_unix']
        else:
            assert observations[-1]['tag']=='final-observation'
            assert training['budget_truncated'] or training['stop_reason'] in ('iteration cap','two consecutive resets')
        entry={'seed':args.seed,'independent_audit_pass':True,'self_test':False,
               'physical_acceptance_pass':passed,**tta}
        output.update(state='PASS_INDEPENDENT_ARCHIVE_READBACK',training_reader={'runs':runs,'protocol_sha256':PROTOCOL_SHA},
                      physical_audits=physical_audits,cohort_entry=entry,
                      driver=json.loads((tmp/'run/control/driver-state.json').read_text()))
    output['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = audit_root/(stem+'-independent-readback.json')
    assert not out.exists(), out
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':output['state'],'output':str(out),'sha256':sha(out),
                      'physical_pass':output['cohort_entry']['physical_acceptance_pass'],'integrity':output['integrity']}))


if __name__ == '__main__':
    main()
