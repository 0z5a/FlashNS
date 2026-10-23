"""Generate fixed CPU source calibration and uniform physical observation data.

Weights describe the explicit diagnostic mixed measure; no training outcome or
formal performance acceptance is implied by producing these caches.
"""
import hashlib
import itertools
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from box376 import oracle
from box376.source import motion, source

ROOT = Path(__file__).resolve().parent
N = 4096
SEED = 793701


def main():
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    regions = np.repeat(np.arange(5), [614, 1024, 1024, 1024, 410])
    time_active = np.zeros(N, dtype=bool)
    time_active[rng.permutation(N)[:3277]] = True
    t = rng.uniform(.25, .75, N)
    flat_count = (~time_active).sum()
    flat_half = rng.random(flat_count) < .5
    flat_times = rng.uniform(0, .25, flat_count)+.75*flat_half
    t[~time_active] = flat_times
    m = motion(t)
    eta = 1/32
    y = rng.uniform(-1, 1, (N, 3))*(m["h"]+eta)
    for index, region in enumerate(regions):
        if 1 <= region <= 3:
            axes = rng.choice(3, int(region), replace=False)
            q = rng.random(int(region))
            signs = rng.choice((-1, 1), int(region))
            y[index, axes] = signs*(m["h"][index, axes]+(2-q)*eta)
    points = np.c_[m["c"]+y, t]
    for index in np.flatnonzero(regions == 4):
        while True:
            trial = rng.random(3)
            if np.any(np.abs(trial-m["c"][index]) > m["h"][index]+2*eta):
                points[index, :3] = trial
                break
    mixture = [.15, .25, .25, .25, .10]
    weights = np.empty(N)
    for region in range(5):
        selected = regions == region
        weights[selected] = mixture[region]/selected.sum()
    assert abs(weights.sum()-1) < 1e-14
    uniform_points = rng.random((N, 4))
    uniform_weights = np.full(N, 1/N)
    records = {}
    raw = {"stratified_points": points, "stratified_weights": weights,
           "stratified_regions": regions, "stratified_time_active": time_active,
           "uniform_points": uniform_points, "uniform_weights": uniform_weights}
    outputs = {}
    for name, x, w in (("stratified", points, weights), ("uniform", uniform_points, uniform_weights)):
        tick = time.perf_counter()
        result = source(x)
        seconds = time.perf_counter()-tick
        outputs[name] = result
        for key in ("U", "G", "f", "df"):
            raw[f"{name}_{key}"] = result[key]
        def weighted_component_rms(value):
            return float(np.sqrt(np.sum(w*np.mean(value.reshape(len(x), -1)**2, axis=-1))))
        records[name] = {"points": len(x), "source_generation_seconds": seconds,
                         "velocity_component_rms": weighted_component_rms(result["U"]),
                         "force_component_rms": weighted_component_rms(result["f"]),
                         "force_gradient_component_rms": weighted_component_rms(result["df"]),
                         "maximum_velocity_magnitude": float(np.linalg.norm(result["U"], axis=-1).max()),
                         "maximum_force_magnitude": float(np.linalg.norm(result["f"], axis=-1).max()),
                         "maximum_force_gradient_component": float(np.abs(result["df"]).max()),
                         "nonzero_velocity_points": int(np.count_nonzero(np.any(result["U"] != 0, axis=-1)))}
        assert records[name]["velocity_component_rms"] > 0
    # High-precision holdout points drawn from the actual saved calibration data,
    # in addition to the separate exhaustive face/edge/corner source suite.
    holdouts = []
    for region in range(5):
        selected = np.flatnonzero((regions == region)&time_active)
        for index in selected[:2]:
            holdouts.append((int(index), int(region)))
    checks = []
    for index, region in holdouts:
        previous = {}
        check = {"calibration_index": index, "region": region, "quantities": {}}
        for precision in (80, 160):
            with mp.workdps(precision):
                expected = oracle.source([mp.mpf(float(v)) for v in points[index]])
                for key, dimensional_scale in (("U", 1.), ("G", 1/eta), ("f", 1+1/eta+.01/eta**2),
                                                ("df", (1+1/eta+.01/eta**2)/eta)):
                    reference_mp = np.asarray(expected[key], dtype=object)
                    reference = np.asarray(reference_mp, dtype=float)
                    computed = outputs["stratified"][key][index]
                    error = np.abs(computed-reference)
                    budget = 1e-11*dimensional_scale+1e-9*np.abs(reference)
                    assert np.all(error <= budget)
                    if precision == 160:
                        doubling = max(abs(a-b) for a, b in zip(reference_mp.flat, previous[key].flat))
                        assert doubling/dimensional_scale < mp.mpf("1e-13")
                        check["quantities"][key] = {"maximum_abs_error": float(error.max()),
                                                   "maximum_error_over_budget": float((error/budget).max()),
                                                   "precision_doubling_error": mp.nstr(doubling, 15)}
                    else:
                        previous[key] = reference_mp.copy()
        checks.append(check)
    raw_file = ROOT / "source-calibration-v1.npz"
    np.savez_compressed(raw_file, **raw)
    tick = time.perf_counter()
    with np.load(raw_file, allow_pickle=False) as loaded:
        for key, value in raw.items():
            assert np.array_equal(loaded[key], value)
    read_seconds = time.perf_counter()-tick
    U = records["stratified"]["velocity_component_rms"]
    scales = {"domain_length": 1., "velocity": U, "pressure": U**2,
              "momentum_residual": records["stratified"]["force_component_rms"],
              "momentum_residual_gradient": records["stratified"]["force_gradient_component_rms"],
              "continuity_residual": U, "continuity_residual_gradient": U}
    for value in scales.values():
        assert value > 0 and np.isfinite(value)
    report = {"state": "BOX376_CPU_FIXED_SOURCE_CALIBRATION_PASS", "actual_exit_code": 0,
              "seed": SEED, "source_nu": .01, "source_eta": eta,
              "records": records, "calibration_spatial_mixture": mixture,
              "spatial_region_counts": np.bincount(regions).tolist(),
              "time_active_count": int(time_active.sum()), "time_flat_count": int((~time_active).sum()),
              "candidate_fixed_physical_scales": scales,
              "scales_depend_on_trainable_parameters": False,
              "actual_saved_calibration_point_80_160dps_checks": checks,
              "raw_sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256((ROOT / "source.py").read_bytes()).hexdigest(),
              "calibration_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cache_read_and_verify_seconds": read_seconds,
              "wall_seconds": time.perf_counter()-started,
              "formal_training_protocol_frozen": False, "GPU_qualification": False,
              "trained_velocity_or_time_to_accuracy": False}
    (ROOT / "source-calibration-report-v1.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"state": report["state"], "points": 2*N, "holdouts": len(checks),
                      "candidate_scales": scales, "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
