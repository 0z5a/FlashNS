"""Summarize preserved measurements and render standalone convergence figures."""

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/hopper_followup"
BACKENDS = (
    "B1",
    "cuBLASLt",
    "F",
    "HopperTMA",
    "TorchJetCompiled",
    "CuEquivariance",
    "TorchNestedAD",
    "PhysicsNeMo",
)


def path_from_remote(value):
    marker = "/root/flashns-hopper-20260909/"
    return ROOT / value.split(marker, 1)[-1]


def main():
    original_path = ROOT / "experiments/pinn_solver/artifacts/formal1/suite.json"
    extension_path = ROOT / "experiments/pinn_solver/artifacts/extension1/suite.json"
    original, extension = [
        json.loads(path.read_text()) for path in (original_path, extension_path)
    ]
    assert original["completed"] and extension["all_runs_converged"]
    points = {(row["backend"], row["seed"]): row for row in extension["runs"]}
    old_points = {(row["backend"], row["seed"]): row for row in original["runs"]}
    seeds = sorted({row["seed"] for row in extension["runs"]})
    summary = {
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (original_path, extension_path)
        },
        "original_converged": sum(
            row["result"]["converged"] for row in original["runs"]
        ),
        "final_converged": 24,
        "seeds": seeds,
        "backends": {},
    }
    med = statistics.median
    for backend in BACKENDS:
        rows = [points[backend, seed] for seed in seeds]
        values = {
            "original_converged": sum(
                old_points[backend, seed]["result"]["converged"] for seed in seeds
            ),
            "converged": len(rows),
            **{
                f"median_{key}": med(row["result"][key] for row in rows)
                for key in (
                    "optimization_wall_seconds",
                    "total_wall_seconds",
                    "adam_updates",
                    "lbfgs_iterations",
                    "loss_gradient_evaluations",
                )
            },
            "median_process_wall_seconds": med(
                row["process_wall_seconds"] for row in rows
            ),
            "observed_total_wall_range_seconds": [
                min(row["result"]["total_wall_seconds"] for row in rows),
                max(row["result"]["total_wall_seconds"] for row in rows),
            ],
            "paired_b1_over_backend_total": med(
                points["B1", seed]["result"]["total_wall_seconds"]
                / points[backend, seed]["result"]["total_wall_seconds"]
                for seed in seeds
            ),
            "all_attempts_process_wall_seconds_sum": sum(
                row["all_attempts_process_wall_seconds"] for row in rows
            ),
            "runs": [
                {
                    "seed": row["seed"],
                    "total_wall_seconds": row["result"]["total_wall_seconds"],
                    "optimization_wall_seconds": row["result"][
                        "optimization_wall_seconds"
                    ],
                    "lbfgs_iterations": row["result"]["lbfgs_iterations"],
                    "loss_gradient_evaluations": row["result"][
                        "loss_gradient_evaluations"
                    ],
                    "metrics": row["result"]["metrics"],
                    "independent_metrics": row["result"]["independent_metrics"],
                }
                for row in rows
            ],
        }
        summary["backends"][backend] = values
    summary["all_formal_attempts_process_wall_seconds_sum"] = sum(
        row["all_attempts_process_wall_seconds"] for row in extension["runs"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "solver-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "| 后端 | 首轮收敛 | 优化+验收 (s) | 求解总计 (s) | 进程总计 (s) | L-BFGS 次数 | loss/grad 评估数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, row in summary["backends"].items():
        lines.append(
            f"| {backend} | {row['original_converged']}/3 | {row['median_optimization_wall_seconds']:.3f} | {row['median_total_wall_seconds']:.3f} | {row['median_process_wall_seconds']:.3f} | {row['median_lbfgs_iterations']} | {row['median_loss_gradient_evaluations']} |"
        )
    (OUT / "solver-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(
        "paired ratios",
        {
            backend: row["paired_b1_over_backend_total"]
            for backend, row in summary["backends"].items()
        },
    )

    colors = plt.get_cmap("tab10").colors
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for axis, seed in zip(axes, seeds):
        for index, backend in enumerate(BACKENDS):
            row = points[backend, seed]
            report = json.loads(path_from_remote(row["output"]).read_text())
            threshold = report["protocol"]["thresholds"]
            history = report["result"]["history"]
            times = [point["optimization_wall_seconds"] for point in history]
            ratios = [
                max(point["metrics"][key] / bound for key, bound in threshold.items())
                for point in history
            ]
            axis.plot(
                times,
                ratios,
                label=backend,
                color=colors[index],
                linewidth=1.4,
                alpha=0.9,
            )
            final_ratio = max(
                report["result"]["independent_metrics"][key] / bound
                for key, bound in threshold.items()
            )
            axis.scatter(times[-1], final_ratio, s=14, color=colors[index], zorder=5)
        axis.axhline(1, color="#59636b", linestyle="--", linewidth=1)
        axis.set_title(f"Seed {seed}")
        axis.set_yscale("log")
        axis.set_xlabel("Optimization + acceptance checks (s)")
        axis.grid(True, alpha=0.18)
    axes[0].set_ylabel("Worst validation metric / its fixed threshold")
    fig.suptitle(
        "Kovasznay FP64 PINN: full solves to the same acceptance criteria", fontsize=14
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"solver-convergence.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.0))
    for index, backend in enumerate(BACKENDS):
        times = np.array(
            [points[backend, seed]["result"]["total_wall_seconds"] for seed in seeds]
        )
        axis.plot(
            [times.min(), times.max()],
            [index, index],
            color=colors[index],
            linewidth=2.5,
            alpha=0.5,
        )
        axis.scatter(
            times, index + np.array([-0.09, 0.0, 0.09]), color=colors[index], s=25
        )
        axis.scatter(
            [np.median(times)],
            [index],
            marker="|",
            color="#20262b",
            s=180,
            linewidths=2,
        )
        axis.text(
            times.max() + 2,
            index,
            f"median {np.median(times):.2f} s",
            va="center",
            fontsize=9,
        )
    axis.set_yticks(range(len(BACKENDS)), BACKENDS)
    axis.invert_yaxis()
    axis.set_xlim(
        0,
        max(row["median_total_wall_seconds"] for row in summary["backends"].values())
        * 1.32,
    )
    axis.set_xlabel("Complete converged-attempt solve wall time (s), including setup")
    axis.set_title(
        "Three paired seeds per backend; dots are observations, lines show range"
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"solver-time.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
