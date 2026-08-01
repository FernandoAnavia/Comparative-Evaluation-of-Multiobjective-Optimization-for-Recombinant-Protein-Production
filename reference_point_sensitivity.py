"""Hypervolume reference-point sensitivity analysis.

The script reads run-level non-dominated solutions from Python and MATLAB,
validates that each proposed reference point is dominated by every point in every
front, and recomputes hypervolume for each run. It then reports mean/SD/median,
mean ranks, and rank stability across reference points.

Default expected files:
- solutions_recuperado.csv
- matlab_fmincon_solutions_nd_30runs.csv

The second file was not included in the reviewed package. Use --allow-partial to
run the evolutionary configurations only; for the manuscript, run without that
flag after adding the MATLAB front-level file.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_REFERENCE_POINTS = [
    (0.00, 0.00),
    (0.05, 0.02),
    (0.10, 0.05),
    (0.20, 0.10),
]
ROUND_DECIMALS = 8
ALGORITHM_ORDER = [
    "NSGA2", "MOEAD_TCH", "MOEAD_PBI", "MOEAD_WS",
    "MOEAD_ASF", "MOEAD_AASF", "FMINCON_49", "FMINCON_100",
]


def unique_nondominated_max(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    valid = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    rounded = np.round(pts, ROUND_DECIMALS)
    _, first_idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[np.sort(first_idx)]

    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if np.any(np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)):
            keep[i] = False
    return pts[keep]


def hypervolume_2d_max(points: np.ndarray, ref: tuple[float, float]) -> float:
    pts = unique_nondominated_max(points)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    previous_x = ref[0]
    for x, y in pts:
        width = x - previous_x
        height = y - ref[1]
        if width > 0 and height > 0:
            hv += width * height
            previous_x = x
    return float(hv)


def detect_columns(frame: pd.DataFrame) -> tuple[str, str]:
    prod_candidates = ["productivity_DH", "productivity", "Productivity"]
    yield_candidates = ["process_yield_H_Gin", "process_yield", "ProcessYield", "yield"]
    prod = next((c for c in prod_candidates if c in frame.columns), None)
    yld = next((c for c in yield_candidates if c in frame.columns), None)
    if prod is None or yld is None:
        raise ValueError(
            "Could not detect productivity/yield columns. "
            f"Available columns: {frame.columns.tolist()}"
        )
    return prod, yld


def load_fronts(input_dir: Path, allow_partial: bool) -> pd.DataFrame:
    paths = [
        input_dir / "solutions_recuperado.csv",
        input_dir / "matlab_fmincon_solutions_nd_30runs.csv",
    ]
    frames = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path.name)
            continue
        frame = pd.read_csv(path)
        prod, yld = detect_columns(frame)
        required = {"algorithm", "seed", prod, yld}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} is missing required columns")
        frame = frame[["algorithm", "seed", prod, yld]].rename(
            columns={prod: "productivity", yld: "process_yield"}
        )
        frames.append(frame)

    if missing and not allow_partial:
        raise FileNotFoundError(
            "Missing front-level file(s): " + ", ".join(missing) + ". "
            "Add the files or rerun with --allow-partial for a diagnostic only."
        )
    if missing:
        warnings.warn(
            "Partial sensitivity analysis: missing " + ", ".join(missing) +
            ". Do not use the partial ranking as the final cross-platform result."
        )
    if not frames:
        raise FileNotFoundError("No solution files were found.")
    return pd.concat(frames, ignore_index=True)


def validate_reference(front: np.ndarray, ref: tuple[float, float], algorithm: str, seed: int) -> None:
    if len(front) == 0:
        raise ValueError(f"Empty front for {algorithm}, seed={seed}")
    if np.any(front[:, 0] <= ref[0]) or np.any(front[:, 1] <= ref[1]):
        minima = front.min(axis=0)
        raise ValueError(
            f"Reference point {ref} is not strictly dominated by every point in "
            f"{algorithm}, seed={seed}. Front minima={tuple(minima)}. "
            "Choose a worse reference point."
        )


def compute_sensitivity(solutions: pd.DataFrame, refs: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for (algorithm, seed), group in solutions.groupby(["algorithm", "seed"], sort=False):
        front = unique_nondominated_max(
            group[["productivity", "process_yield"]].to_numpy(float)
        )
        for ref_prod, ref_yield in refs:
            ref = (ref_prod, ref_yield)
            validate_reference(front, ref, str(algorithm), int(seed))
            rows.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "reference_productivity": ref_prod,
                "reference_process_yield": ref_yield,
                "HV": hypervolume_2d_max(front, ref),
            })
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = runs.groupby(
        ["reference_productivity", "reference_process_yield", "algorithm"],
        as_index=False,
    ).agg(
        runs=("seed", "count"),
        HV_mean=("HV", "mean"),
        HV_std=("HV", "std"),
        HV_median=("HV", "median"),
        HV_q1=("HV", lambda x: x.quantile(0.25)),
        HV_q3=("HV", lambda x: x.quantile(0.75)),
    )

    summary["rank_by_mean_HV"] = summary.groupby(
        ["reference_productivity", "reference_process_yield"]
    )["HV_mean"].rank(method="min", ascending=False)

    rank_pivot = summary.pivot_table(
        index="algorithm",
        columns=["reference_productivity", "reference_process_yield"],
        values="rank_by_mean_HV",
    ).reset_index()
    return summary, rank_pivot


def plot_means(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for algorithm in ALGORITHM_ORDER:
        sub = summary[summary["algorithm"] == algorithm].sort_values(
            ["reference_productivity", "reference_process_yield"]
        )
        if sub.empty:
            continue
        labels = [
            f"({p:.2f}, {y:.2f})"
            for p, y in zip(sub["reference_productivity"], sub["reference_process_yield"])
        ]
        ax.plot(labels, sub["HV_mean"], marker="o", label=algorithm)
    ax.set_xlabel("Hypervolume reference point (productivity, process yield)")
    ax.set_ylabel("Mean hypervolume")
    ax.set_title("Reference-point sensitivity of hypervolume")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_refs(values: list[str] | None) -> list[tuple[float, float]]:
    if not values:
        return DEFAULT_REFERENCE_POINTS
    refs = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid reference '{value}'. Use productivity,yield")
        refs.append((float(parts[0]), float(parts[1])))
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reference_sensitivity"))
    parser.add_argument(
        "--reference", action="append",
        help="Reference point as productivity,yield. Repeat for multiple points.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    refs = parse_refs(args.reference)
    solutions = load_fronts(args.input_dir, args.allow_partial)
    runs = compute_sensitivity(solutions, refs)
    summary, ranks = summarize(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "hv_reference_sensitivity_runs.csv", index=False)
    summary.to_csv(args.output_dir / "hv_reference_sensitivity_summary.csv", index=False)
    ranks.to_csv(args.output_dir / "hv_reference_rank_stability.csv", index=False)
    plot_means(summary, args.output_dir / "hv_reference_sensitivity.png")

    print("Reference-point sensitivity completed.")
    print(summary.to_string(index=False))
    if args.allow_partial:
        print("\nWARNING: partial diagnostic only; add MATLAB front-level solutions before manuscript use.")


if __name__ == "__main__":
    main()
