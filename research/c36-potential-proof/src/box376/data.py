"""Hashed fixed-source diagnostic data; this is not a formal training protocol."""
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from box376.source import C0, motion, source

ROOT = Path(__file__).resolve().parent
MATH_COMMIT = "adc7f1241b42e322a6451854ab7e4b4c146bf78a"
FAMILIES = ("interior", "initial", "boundary", "gauge")
COEFFICIENTS = (1., 5., 5., 5.)


def digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def make_data(n, version=0):
    rng = np.random.default_rng(736001+n)
    eta, nu = 1/32, (.01 if version == 0 else .011)
    t = rng.uniform(.27, .73, n)
    if n > 20:
        flat = np.arange(n) % 5 == 0
        t[flat] = np.where(np.arange(n)[flat] % 2, rng.uniform(.8, 1, flat.sum()), rng.uniform(0, .2, flat.sum()))
    if n == 3:
        t[:] = [.375, .5, .625]
    m = motion(t)
    regions = np.arange(n) % 5
    if n == 3:
        regions[:] = [1, 2, 3]
    y = rng.uniform(-1, 1, (n, 3)) * (m["h"] + .8*eta)
    for i in range(n):
        count = int(regions[i])
        if 1 <= count <= 3:
            axes = rng.choice(3, size=count, replace=False)
            q = rng.uniform(.125, .875, count)
            signs = rng.choice((-1, 1), count)
            y[i, axes] = signs*(m["h"][i, axes]+2*eta-q*eta)
    interior = np.c_[m["c"]+y, t]
    outside = regions == 4
    interior[outside, :3] = rng.uniform(.02, .12, (outside.sum(), 3))
    counts = {"interior": n, "initial": min(17, max(2, n//11)),
              "boundary": min(13, max(1, n//19)), "gauge": 2}
    points = {"interior": interior}
    points["initial"] = np.c_[rng.uniform(0, 1, (counts["initial"], 3)), np.zeros(counts["initial"])]
    b = rng.uniform(0, 1, (counts["boundary"], 4))
    for i in range(len(b)):
        b[i, (i//2) % 3] = i % 2
    points["boundary"] = b
    points["gauge"] = np.c_[np.full((2, 3), 1/16), [.375, .625]]
    tick = time.perf_counter()
    exact = source(interior, eta=eta, nu=nu)
    generation_seconds = time.perf_counter()-tick
    data = {}
    for family in FAMILIES:
        size = counts[family]
        # Independent nonuniform weights for every loss family.
        weights = .2 + rng.uniform(.1, 1.5, size) + np.arange(size)/(size+1)
        data[family] = {"points": points[family], "weights": weights,
                        "ids": np.arange(size, dtype=np.int64),
                        "target": np.zeros((size, 4)),
                        "W": float(weights.sum())}
    data["interior"].update(force=exact["f"], force_gradient=exact["df"],
                              exact_velocity=exact["U"], region=regions)
    metadata = {"math_commit": MATH_COMMIT, "source_version": version,
                "fixture": {"eta": eta, "nu": nu, "c0": C0.tolist(),
                            "displacement": [1/8, 1/16, 1/32], "h0": [1/16]*3,
                            "final_D": [2, 1.5, 1/3]},
                "coordinate_order": ["physical_x", "physical_y", "physical_z", "physical_t"],
                "dtype": "float64", "source_channels": "f[i], df[i,k] = partial_xk f_i",
                "generator_sha256": hashlib.sha256((ROOT / "source.py").read_bytes()).hexdigest(),
                "sampler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "family_counts": counts,
                "family_coefficients": dict(zip(FAMILIES, COEFFICIENTS)),
                "global_weight_sums": {k: data[k]["W"] for k in FAMILIES},
                "array_hashes": {f"{k}/{name}": digest(value) for k in FAMILIES for name, value in data[k].items() if isinstance(value, np.ndarray)},
                "component_rule": "sum 4 residual components + 0.2 * sum 12 spatial residual derivatives; value losses sum selected components",
                "formal_training_protocol": False}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata.update(cache_key=key, source_generation_seconds=generation_seconds)
    return {"families": data, "metadata": metadata, "nu": nu, "eta": eta, "lambda_g": .2}


def save_cache(data, path):
    path = Path(path)
    tick = time.perf_counter()
    arrays = {f"{family}_{name}": value for family, values in data["families"].items() for name, value in values.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)
    # Read and hash all actual loaded bytes once; generation and read costs are separate.
    read_started = time.perf_counter()
    with np.load(path, allow_pickle=False) as cache:
        hashes = {name: digest(cache[name]) for name in cache.files}
    read_seconds = time.perf_counter()-read_started
    expected = {f"{family}_{name}": digest(value) for family, values in data["families"].items() for name, value in values.items() if isinstance(value, np.ndarray)}
    assert hashes == expected
    result = dict(data["metadata"], file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  cache_write_and_read_seconds=time.perf_counter()-tick,
                  cache_read_and_hash_seconds=read_seconds, loaded_arrays_verified=True)
    path.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def torch_data(data, rank=0, world_size=1):
    result = {k: data[k] for k in ("metadata", "nu", "eta", "lambda_g")}
    result["families"] = {}
    for name, family in data["families"].items():
        selected = family["ids"] % world_size == rank
        result["families"][name] = {k: (torch.from_numpy(v[selected].copy()) if isinstance(v, np.ndarray) else v) for k, v in family.items()}
    return result


def layouts(topology):
    return [(out, inp) for inp, out in zip(topology[:-1], topology[1:])]


def initialize(topology, seed=361901):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn((out, inp), dtype=torch.float64, generator=generator)*(.7/np.sqrt(inp)),
             torch.randn(out, dtype=torch.float64, generator=generator)*.05) for out, inp in layouts(topology)]


def flatten(parameters):
    return torch.cat([value.reshape(-1) for layer in parameters for value in layer])


def views(flat, topology):
    result = []
    offset = 0
    for out, inp in layouts(topology):
        W = flat[offset:offset+out*inp].view(out, inp)
        offset += out*inp
        b = flat[offset:offset+out]
        offset += out
        result.append((W, b))
    assert offset == flat.numel()
    return result
