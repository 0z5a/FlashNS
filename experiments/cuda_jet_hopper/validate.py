"""Reject numerical, stream, tail, graph, and guard failures before timing."""

import argparse
import ctypes
import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from hopper import Hopper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cuda_jet_h100"))
from common import comparison, local_case, make_case, nested_reference, step
from gpu import Native


def make_local(batch, dim, cin, cout, native, seed=71983):
    generator = torch.Generator(device="cpu").manual_seed(
        seed + batch + dim + cin + cout
    )
    q = 10 if dim == 2 else 20
    z = (
        torch.randn(batch, q, cin, generator=generator, dtype=torch.float64) * 0.25
    ).cuda()
    stress = torch.tensor(
        [-24.0, -20.0, -18.0, -12.0, -2.0, 0.0, 2.0, 12.0, 18.0, 20.0, 24.0],
        dtype=torch.float64,
        device="cuda",
    )
    if batch:
        z[:, 0] = stress[
            torch.arange(batch * cin, device="cuda").reshape(batch, cin) % len(stress)
        ]
    d = torch.randn(batch, q, cout, generator=generator, dtype=torch.float64).cuda()
    weight = (
        torch.randn(cout, cin, generator=generator, dtype=torch.float64) * 0.2
    ).cuda()
    h, aux = native.forward(z, dim)
    return d, weight, h, aux


def local_checks(native, baseline, exhaustive):
    records = []
    batches = (1, 7, 61, 130) if exhaustive else (1, 7, 61)
    with torch.no_grad():
        for dim, batch, cin, cout in itertools.product(
            (2, 3), batches, (32, 64), (32, 64)
        ):
            data = make_local(batch, dim, cin, cout, baseline)
            d, weight, h, aux = data
            bar = d @ weight
            expected = baseline.vjp(h, aux, bar, dim)
            q = 10 if dim == 2 else 20
            guarded = torch.full(
                (batch * q * cin + 16,), 9876543.25, dtype=torch.float64, device="cuda"
            )
            target = guarded[8:-8].view_as(h)
            actual = native.raw_dgrad(*data, dim, True, target)
            result = comparison(actual, expected)
            assert bool(
                (guarded[:8] == 9876543.25).all() and (guarded[-8:] == 9876543.25).all()
            )
            unfused = native.raw_dgrad(*data, dim, False)
            gemm_result = comparison(unfused, bar)
            uf = comparison(native.vjp(h, aux, unfused, dim), actual)
            records.append(
                {
                    "dimension": dim,
                    "batch": batch,
                    "cin": cin,
                    "cout": cout,
                    "fused": result,
                    "raw_dgrad": gemm_result,
                    "U_vs_F": uf,
                    "canaries": "passed",
                }
            )
    return records


def integration_checks(native, baseline):
    record = {}
    case = make_case(67, weighted=True, seed=91370)
    local = local_case(case, 1, 0, "cuda:0")
    x, quadrature, weights, biases, meta = local
    arguments = (x, quadrature, meta["global_denominator"], weights, biases)
    expected_loss, expected_gradients = nested_reference(*arguments)
    for mode in ("U", "F"):
        loss, gradients = step(*arguments, native, mode)
        record["network_" + mode] = {
            "loss": comparison(loss, expected_loss),
            "parameter_gradients": [
                comparison(a, b) for a, b in zip(gradients, expected_gradients)
            ],
        }
    with torch.no_grad():
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            torch.cuda._sleep(500000)
            data = make_local(19, 3, 64, 64, baseline, seed=99371)
            actual = native.dgrad(*data, 3, "F")
            ready = torch.cuda.Event()
            ready.record()
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            expected = baseline.dgrad(*data, 3, "B1")
            copied = actual.clone()
        consumer.synchronize()
        record["producer_consumer_streams"] = comparison(copied, expected)
        data = make_local(17, 2, 64, 64, baseline, seed=91811)
        for _ in range(3):
            native.dgrad(*data, 2, "F")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = native.dgrad(*data, 2, "F")
        data[0].add_(0.03125)
        for _ in range(4):
            graph.replay()
        record["graph_replay_updated_inputs"] = comparison(
            captured, baseline.dgrad(*data, 2, "B1")
        )
        try:
            native.raw_dgrad(*data, 2, True, data[0])
        except RuntimeError as error:
            assert "returned 1" in str(error)
        else:
            raise AssertionError("output/input overlap accepted")
        invalid = ctypes.c_void_p()
        code = native.create(
            2,
            data[0].data_ptr(),
            data[1].data_ptr(),
            2**31 // 20 + 1,
            64,
            64,
            ctypes.byref(invalid),
        )
        assert code == 1 and not invalid.value
        record["host_guards"] = "passed"
        empty = make_local(0, 2, 64, 64, baseline)
        record["empty_batch"] = list(native.dgrad(*empty, 2, "F").shape)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite validation")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    build = json.loads((args.build / "build.json").read_text())
    baseline = Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "build_sha256": hashlib.sha256(
            (args.build / "build.json").read_bytes()
        ).hexdigest(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob("*.py")
        },
        "candidates": {},
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for identifier in build["libraries"]:
        print({"validating": identifier}, flush=True)
        native = Hopper(args.build, identifier)
        candidate = {
            "binary_sha256": native.entry["sha256"],
            "resources": native.resources(),
            "passed": False,
        }
        report["candidates"][identifier] = candidate
        save()
        try:
            candidate["local"] = local_checks(native, baseline, args.exhaustive)
            candidate["integration"] = integration_checks(native, baseline)
            candidate["plan_statistics"] = native.plan_statistics
            candidate["passed"] = True
        except Exception as error:
            candidate["error"] = repr(error)
            save()
            raise
        finally:
            native.close()
        save()
        print(
            {"passed": identifier, "local_cases": len(candidate["local"])}, flush=True
        )
    report["passed"] = all(item["passed"] for item in report["candidates"].values())
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
