"""Read all attempts, NumPy-check saved gradients, and repeat CPU acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    frozen = json.loads((root / "frozen-H20-inputs.json").read_text())
    thresholds = dict(pde_rms=.02, boundary_uv_rms=.01, outflow_p_rms=.01,
                      relative_l2_u=.01, relative_l2_v=.02, relative_l2_p=.02)
    rows, numerical = [], []
    for index in range(4):
        lane = root / ("lane" + str(index))
        report = json.loads((lane / "result.json").read_text())
        assert report["owned_children_joined"] and report["original_GPU_lock_released"]
        for row in report["rows"]:
            if row["status"] == "NOT_RUN":
                rows.append({"lane": index, "name": row["name"], "outcome": "NOT_RUN", "reason": row["reason"]})
                continue
            assert row["joined"]
            path = lane / (row["name"] + ".json")
            outcome = {"lane": index, "name": row["name"], "exit": row["exit"], "case": row["case"]}
            if row["exit"] != 0 or not path.exists():
                outcome["outcome"] = "FAILED_OR_NOT_CONVERGED"
                if path.exists():
                    actual = json.loads(path.read_text())
                    outcome.update(converged=actual.get("solve", {}).get("converged"), error=actual.get("error"))
                rows.append(outcome)
                continue
            actual = json.loads(path.read_text())
            assert actual["status"] == "passed"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["output_sha256"]
            proof = path.with_suffix(".proof.json")
            if proof.exists():
                saved = json.loads(proof.read_text())
                assert saved["state"] == "PASS" and saved["arrays"] == 15
                raw_path = path.with_suffix(".proof.npz")
                assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == saved["raw_sha256"]
                with np.load(raw_path, allow_pickle=False) as raw:
                    assert len(raw.files) == 15
                    for trial in range(3):
                        theta = raw[f"trial{trial}_theta"]
                        assert theta.shape == (4547,) and theta.dtype == np.float64 and np.isfinite(theta).all()
                        if trial == 0:
                            assert hashlib.sha256(theta.tobytes()).hexdigest() == frozen["theta_hashes"]["361902"]
                        for kind in ["loss", "gradient"]:
                            a, b = raw[f"trial{trial}_{kind}_actual"], raw[f"trial{trial}_{kind}_reference"]
                            assert a.shape == b.shape and a.dtype == b.dtype == np.float64
                            assert a.size == (4547 if kind == "gradient" else 1)
                            assert np.isfinite(a).all() and np.isfinite(b).all()
                            error = float(np.max(np.abs(a - b) / (2e-12 + 2e-11 * np.abs(b))))
                            assert error <= 1, (row["name"], trial, kind, error)
                            numerical.append({"name": row["name"], "lane": index, "trial": trial, "kind": kind, "error_over_budget": error})
            if row["case"].get("solve"):
                result = actual["solve"]
                seed = str(row["case"]["seed"])
                assert result["converged"] and result["parameter_initial_sha256"] == frozen["theta_hashes"][seed]
                assert actual["protocol"]["adam_steps"] == 4000 and actual["protocol"]["lbfgs_max_blocks"] == 500
                assert actual["protocol"]["thresholds"] == thresholds
                for metrics in [result["metrics"], result["independent_metrics"]]:
                    assert set(metrics) == set(thresholds)
                    assert all(np.isfinite(metrics[k]) and metrics[k] <= v for k, v in thresholds.items())
                # Additional complete scalar coordinate AD on CPU; no optimizer rerun.
                import torch
                torch.set_num_threads(1)
                sys.path[:0] = [str(root / "source/src"), str(root / "source/experiments/pinn_solver")]
                from problem import Problem, independent_metrics, digest
                state = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
                assert state.dtype == torch.float64 and state.shape == (4547,)
                assert digest(state) == result["parameter_final_sha256"]
                cpu_metrics = independent_metrics(Problem(device="cpu"), state)
                for k, limit in thresholds.items():
                    assert np.isfinite(cpu_metrics[k]) and cpu_metrics[k] <= limit, (row["name"], k, cpu_metrics[k])
                    assert np.isclose(cpu_metrics[k], result["independent_metrics"][k], atol=2e-12, rtol=2e-11)
                outcome.update(outcome="ACCEPTED_CPU_RECHECKED", seed=int(seed), backend=row["case"]["backend"],
                    setup_plus_optimization_seconds=result["setup_plus_optimization_seconds"],
                    CPU_independent_metrics=cpu_metrics)
            elif "wall_ms_per_gradient" in actual:
                times = np.array(actual["wall_ms_per_gradient"], dtype=np.float64)
                assert len(times) == 5 and np.isfinite(times).all() and np.all(times > 0)
                assert np.isclose(np.median(times), actual["median_ms"], atol=0, rtol=1e-12)
                outcome.update(outcome="QUALIFIED", median_ms=actual["median_ms"])
            else:
                assert len(actual["samples"]) == 7 and actual["correctness_against_q10"]["passed"]
                assert all(np.isfinite(s["gpu_ms_per_gradient"]) and s["gpu_ms_per_gradient"] > 0 for s in actual["samples"])
                outcome.update(outcome="MEMORY_REPLAY_CHECKED", median_ms=actual["median_ms"],
                    engine_setup_peak_allocated_bytes=actual["engine_setup_peak_allocated_bytes"])
            rows.append(outcome)
    comparisons = []
    for lane, candidate in [(0, "gl8_native"), (1, "gl8_native"), (2, "gl8_native")]:
        accepted = [r for r in rows if r["lane"] == lane and r["outcome"] == "ACCEPTED_CPU_RECHECKED"]
        for backend in sorted({r["backend"] for r in accepted} - {candidate}):
            pairs = []
            for seed in range(862101, 862107):
                a = next((r for r in accepted if r["seed"] == seed and r["backend"] == candidate), None)
                b = next((r for r in accepted if r["seed"] == seed and r["backend"] == backend), None)
                if a and b:
                    pairs.append({"seed": seed, "baseline_seconds": b["setup_plus_optimization_seconds"],
                        "candidate_seconds": a["setup_plus_optimization_seconds"],
                        "ratio": b["setup_plus_optimization_seconds"] / a["setup_plus_optimization_seconds"]})
            comparisons.append({"lane": lane, "baseline": backend, "candidate": candidate, "accepted_pairs": len(pairs),
                "paired_geometric_mean_ratio": float(np.exp(np.log([p["ratio"] for p in pairs]).mean())) if pairs else None,
                "pairs": pairs, "scope": "Setup plus optimization; matched seed and physical GPU; every attempt retained"})
    result = {"state": "AUDITED", "numerical_reader": "NumPy raw-array and timing recomputation",
        "additional_acceptance_reader": "Complete CPU scalar coordinate AD for every accepted trained checkpoint",
        "numerical_comparisons": numerical, "attempts": rows, "paired_comparisons": comparisons,
        "all_attempts_retained": True, "Slurm_exit_is_scientific_pass": False}
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = (root / "prepared-summary.zh-CN.md").read_text()
    summary += "\n## 本次 H100 二维补测结果\n\n"
    summary += "以下均为本次二维 Kovasznay 补测的实际输出，与前文的 H100 三维 C36 梯度吞吐结果分开。时间比值为同卡、同初始化、双方均通过验收的设置加优化时间之比。\n\n"
    summary += "| 队列 | 对照 / GL8 | 合格配对数 | 配对几何平均加速比 |\n| --- | --- | ---: | ---: |\n"
    for row in comparisons:
        ratio = row["paired_geometric_mean_ratio"]
        text = f"{ratio:.6f}×" if ratio is not None else "未建立"
        summary += f"| {row['lane']} | {row['baseline']} / {row['candidate']} | {row['accepted_pairs']} / 6 | {text} |\n"
    counts = {name: sum(r["outcome"] == name for r in rows) for name in
        ["ACCEPTED_CPU_RECHECKED", "FAILED_OR_NOT_CONVERGED", "NOT_RUN", "QUALIFIED", "MEMORY_REPLAY_CHECKED"]}
    summary += "\n全部尝试的结果计数：" + "；".join(f"{name}={count}" for name, count in counts.items()) + "。所有通过的训练检查点另经完整 CPU 标量坐标 AD 验收。失败和未运行项保留在 audit.json；准备检查或 Slurm 完成状态不等于科学验收通过。\n"
    (root / "chinese-summary.zh-CN.md").write_text(summary)
    print(json.dumps({"state": "AUDITED", "attempts": len(rows), "raw_gradient_checks": len(numerical),
        "accepted_solves_CPU_rechecked": sum(r["outcome"] == "ACCEPTED_CPU_RECHECKED" for r in rows)}))


if __name__ == "__main__":
    main()
