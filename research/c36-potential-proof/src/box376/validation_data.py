"""Independent development holdouts for the fixed compact-curl box source.

The stratified mixed measure and uniform physical volume are separate datasets.
Interior points do not reuse scale-calibration or training diagnostics.
This freezes validation data only, not a formal training or performance budget.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from box376.data import digest, MATH_COMMIT
from box376.source import motion, source

REGIONS = ("plateau", "face", "edge", "corner", "outside")
SPATIAL_MASSES = np.array((.15, .25, .25, .25, .10))
TIME_MASSES = np.array((.8, .1, .1))
TIMES = np.array((0., .25, .375, .5, .625, .75, 1.))
SEED = 794701


def spatial_points(rng, times, regions, eta):
    m = motion(times)
    y = rng.uniform(-1, 1, (len(times), 3))*(m["h"]+eta)
    for index, region in enumerate(regions):
        if 1 <= region <= 3:
            axes = rng.choice(3, int(region), replace=False)
            signs = rng.choice((-1, 1), int(region))
            y[index, axes] = signs*(m["h"][index, axes]+eta*(1+rng.random(int(region))))
    points = np.c_[m["c"]+y, times]
    for index in np.flatnonzero(regions == 4):
        while True:
            trial = rng.random(3)
            if np.any(np.abs(trial-m["c"][index]) > m["h"][index]+2*eta):
                points[index, :3] = trial
                break
    assert np.all((points >= 0) & (points <= 1))
    return points


def classify(points, eta):
    m = motion(points[:, 3])
    distance = np.abs(points[:, :3]-m["c"])
    transition = distance > m["h"]+eta
    outside = np.any(distance >= m["h"]+2*eta, axis=1)
    regions = transition.sum(axis=1).astype(np.int64)
    regions[outside] = 4
    return regions


def make_holdouts(seed=SEED, eta=1/32, nu=.01):
    rng = np.random.default_rng(seed)
    counts = (614, 1024, 1024, 1024, 410)
    regions, time_regions, weights, times = [], [], [], []
    for region, count in enumerate(counts):
        cell_counts = (count-2*(count//10), count//10, count//10)
        for time_region, (number, interval) in enumerate(zip(cell_counts, ((.25,.75),(0,.25),(.75,1)))):
            regions.extend([region]*number)
            time_regions.extend([time_region]*number)
            times.extend(rng.uniform(*interval, number))
            weights.extend([SPATIAL_MASSES[region]*TIME_MASSES[time_region]/number]*number)
    regions = np.asarray(regions, dtype=np.int64)
    time_regions = np.asarray(time_regions, dtype=np.int64)
    points = spatial_points(rng, np.asarray(times), regions, eta)
    weights = np.asarray(weights, dtype=np.float64)
    assert np.array_equal(classify(points,eta),regions)
    datasets = {"stratified":{"points":points,"weights":weights,"regions":regions,"time_regions":time_regions},
                "uniform":{"points":rng.random((4096,4)),"weights":np.full(4096,1/4096)}}
    datasets["uniform"]["regions"] = classify(datasets["uniform"]["points"],eta)
    for time in TIMES:
        slice_regions = np.repeat(np.arange(5), (38,64,64,64,26)).astype(np.int64)
        slice_points = spatial_points(rng,np.full(256,time),slice_regions,eta)
        slice_weights = np.concatenate([np.full(np.count_nonzero(slice_regions==r),SPATIAL_MASSES[r]/np.count_nonzero(slice_regions==r)) for r in range(5)])
        datasets[f"time-{time:g}"] = {"points":slice_points,"weights":slice_weights,"regions":slice_regions}
    initial = np.c_[rng.random((512,3)),np.zeros(512)]
    boundary = rng.random((384,4))
    for face in range(6):
        boundary[face*64:(face+1)*64,face//2] = face%2
    raw = {"initial_points":initial,"boundary_points":boundary,
           "gauge_points":np.c_[np.full((len(TIMES),3),1/16),TIMES]}
    for name, dataset in datasets.items():
        dataset["reference"] = source(dataset["points"],eta=eta,nu=nu)
        for key in ("points","weights","regions","time_regions"):
            if key in dataset:raw[name+"_"+key] = dataset[key]
        for key,value in dataset["reference"].items():
            raw[name+"_source_"+key] = value
        assert abs(dataset["weights"].sum()-1) < 1e-14
    metadata = {"state":"INDEPENDENT_DEVELOPMENT_HOLDOUTS_FROZEN", "seed":int(seed),
                "math_commit":MATH_COMMIT,
                "source_sha256":hashlib.sha256((Path(__file__).parent/"source.py").read_bytes()).hexdigest(),
                "sampler_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "eta":eta,"nu":nu,"spatial_masses":SPATIAL_MASSES.tolist(),
                "stratified_time_masses":TIME_MASSES.tolist(),"time_slices":TIMES.tolist(),
                "stratified_points":4096,"uniform_physical_volume_points":4096,
                "time_slice_points":256,"initial_points":512,"boundary_points":384,
                "scale_calibration_interior_points_reused":False,"training_diagnostic_interior_points_reused":False,
                "gauge_position_is_shared_by_definition":True,
                "formal_training_protocol_frozen":False,
                "array_hashes":{k:digest(v) for k,v in raw.items()}}
    metadata["validation_key"] = hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return datasets, raw, metadata
