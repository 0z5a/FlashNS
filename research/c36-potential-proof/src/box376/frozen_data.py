"""Read a fixed-source cache and verify its bytes and individual arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

from box376.data import FAMILIES, digest


def load_cache(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["file_sha256"]:
        raise ValueError("Source-cache file hash mismatch")
    families = {}
    with np.load(path, allow_pickle=False) as cache:
        for name in FAMILIES:
            prefix = name+"_"
            family = {key[len(prefix):]: cache[key].copy() for key in cache.files if key.startswith(prefix)}
            for key, value in family.items():
                if digest(value) != metadata["array_hashes"][f"{name}/{key}"]:
                    raise ValueError(f"Source-cache array hash mismatch: {name}/{key}")
            if len(family["ids"]) != metadata["family_counts"][name]:
                raise ValueError("Source-cache family count mismatch")
            family["W"] = metadata["global_weight_sums"][name]
            families[name] = family
    return {"metadata": metadata, "families": families,
            "nu": metadata["fixture"]["nu"], "eta": metadata["fixture"]["eta"],
            "lambda_g": .2}
