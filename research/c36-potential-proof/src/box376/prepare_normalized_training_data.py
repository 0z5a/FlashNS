"""Reconstruct immutable regional training inputs used by the CPU development.

This generates source data only. It does not train, create a GPU lease, or freeze
a formal training budget. Holdouts and scale calibration use separate seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from box376.data import FAMILIES, COEFFICIENTS, digest, save_cache
from box376.normalized_objective import normalization_key
from box376.validation_data import make_holdouts


def prepare(calibration, training_point_seed=795701):
    training, raw, sampler = make_holdouts(seed=training_point_seed)
    key = normalization_key(calibration)
    interior = training["stratified"]
    families = {}
    for name in FAMILIES:
        points = interior["points"] if name == "interior" else raw[name+"_points"]
        weights = interior["weights"] if name == "interior" else np.full(len(points), 1/len(points))
        families[name] = {"points": points.copy(), "weights": weights.copy(),
                          "ids": np.arange(len(points), dtype=np.int64),
                          "target": np.zeros((len(points), 4)), "W": float(weights.sum())}
    families["interior"].update(force=interior["reference"]["f"].copy(),
                              force_gradient=interior["reference"]["df"].copy())
    metadata = {"state": "FIXED_REGIONAL_DEVELOPMENT_TRAINING_DATA_NOT_FORMAL", "source_version": 0,
                "fixture": {"eta": 1/32, "nu": .01}, "training_point_seed": int(training_point_seed),
                "source_sampler_metadata": sampler, "normalization_key": key,
                "family_counts": {n: len(families[n]["points"]) for n in FAMILIES},
                "global_weight_sums": {n: families[n]["W"] for n in FAMILIES},
                "family_coefficients": dict(zip(FAMILIES, COEFFICIENTS)),
                "array_hashes": {f"{n}/{k}": digest(v) for n in FAMILIES
                                 for k, v in families[n].items() if isinstance(v, np.ndarray)},
                "formal_training_protocol": False}
    metadata["cache_key"] = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"families": families, "metadata": metadata, "nu": .01, "eta": 1/32, "lambda_g": .2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Refuse to overwrite frozen inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text())
    prepared = prepare(calibration)
    save_cache(prepared, args.output)
    print(json.dumps({"cache_key": prepared["metadata"]["cache_key"],
                      "family_counts": prepared["metadata"]["family_counts"], "GPU_count": 0}), flush=True)


if __name__ == "__main__":
    main()
