"""Calibrate the native physical observer's chunk size on one GPU.

Times native C17 observation of the first stratified holdout points at several chunk
sizes, records peak memory, and compares every state with the default chunk 128. The
observer itself (box376.potential_physical) is not modified; its --observer-chunk option
selects the chunk size, which changes batching only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from box376.c36_coordinate_oracle import native_c17
from box376.data import views
from box376.potential_geometry import native_field
from box376.potential_run import sha, write_json
from box376.validation_data import make_holdouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--chunks", default="128,256,512,1024")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    cuda = args.device == "cuda"
    if cuda:
        assert torch.cuda.device_count() == 1
    protocol = json.loads(args.protocol.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["representation"] == "C36-curl-potential"
    parameters = views(checkpoint["theta"].to(args.device), tuple(protocol["topology"]))
    datasets, _, _ = make_holdouts(seed=protocol["physical"]["holdout_seed"])
    points = datasets["stratified"]["points"][:args.points]
    report = {"checkpoint_sha256": sha(args.checkpoint), "points": int(len(points)), "rows": [],
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if cuda else None}
    reference = None
    for chunk in [int(c) for c in args.chunks.split(",")]:
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        states = []
        try:
            for begin in range(0, len(points), chunk):
                x = torch.from_numpy(points[begin:begin + chunk].copy()).to(args.device).requires_grad_(True)
                states.append(native_c17(lambda p: native_field(parameters, p), x).detach().cpu().numpy())
            if cuda:
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as error:
            report["rows"].append({"chunk": chunk, "failed": repr(error)[:500]})
            write_json(args.output / "calibration.json", report)
            del states
            torch.cuda.empty_cache()
            continue
        seconds = time.perf_counter() - started
        state = np.concatenate(states)
        row = {"chunk": chunk, "seconds": seconds, "seconds_per_point": seconds / len(points),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None}
        if reference is None:
            reference = state
            report["reference_chunk"] = chunk
            report["reference_max_abs"] = float(np.abs(state).max())
        else:
            difference = np.abs(state - reference)
            row["max_abs_difference_vs_first"] = float(difference.max())
            row["max_relative_difference_vs_first"] = float((difference / (1e-300 + np.abs(reference))).max())
            row["bitwise_equal_vs_first"] = bool(np.array_equal(state, reference))
        report["rows"].append(row)
        write_json(args.output / "calibration.json", report)
        print(json.dumps(row), flush=True)
        del states, state
        if cuda:
            torch.cuda.empty_cache()
    write_json(args.output / "calibration.json", report)


if __name__ == "__main__":
    main()
