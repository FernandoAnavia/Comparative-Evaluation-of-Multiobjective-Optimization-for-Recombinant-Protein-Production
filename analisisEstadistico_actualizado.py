"""Updated statistical analysis for the WITCOM manuscript.

Main corrections
----------------
1. Uses the same objective-vector post-processing for every configuration:
   vectors are rounded to 8 decimals, duplicates are removed, and the
   non-dominated set is reconstructed before cardinality and spacing.
2. Uses the sample standard deviation (ddof=1) for spacing, matching MATLAB.
3. Treats cross-platform runs as independent in the primary inference:
   Kruskal-Wallis omnibus tests and Mann-Whitney U comparisons against NSGA-II.
4. Applies a proper monotone Holm correction within each metric.
5. Reports Cliff's delta and leaves cardinality/evaluation counts descriptive
   because they are ceiling/tie-heavy or fixed by design.

Expected files in the working directory:
- metrics_recuperado.csv
- solutions_recuperado.csv
- matlab_fmincon_metrics_30runs.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu


ALGORITHM_ORDER = [
    "NSGA2", "MOEAD_TCH", "MOEAD_PBI", "MOEAD_WS",
    "MOEAD_ASF", "MOEAD_AASF", "FMINCON_49", "FMINCON_100",
]
TEST_METRICS = ["HV", "Spacing", "time_seconds"]
REFERENCE_ALGORITHM = "NSGA2"
ROUND_DECIMALS = 8


def unique_nondominated_max(points: np.ndarray, decimals: int = ROUND_DECIMALS) -> np.ndarray:
    """Return unique non-dominated points for a two-objective maximization problem."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)

    valid = np.all(np.isfinite(pts), axis=1) & np.all(pts > 0, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    rounded = np.round(pts, decimals)
    _, first_idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[np.sort(first_idx)]

    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        dominated_by_any = np.any(
            np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        )
        if dominated_by_any:
            keep[i] = False

    return pts[keep]


def hypervolume_2d_max(points: np.ndarray, ref: tuple[float, float] = (0.0, 0.0)) -> float:
    """Two-dimensional hypervolume for maximization."""
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


def spacing_sample(points: np.ndarray) -> float:
    """Sample SD of consecutive Euclidean distances after sorting by productivity."""
    pts = unique_nondominated_max(points)
    if len(pts) < 3:
        return float("nan")
    pts = pts[np.argsort(pts[:, 0])]
    distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.std(distances, ddof=1))


def reconstruct_python_metrics(metrics: pd.DataFrame, solutions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "algorithm", "seed", "productivity_DH", "process_yield_H_Gin"
    }
    missing = required.difference(solutions.columns)
    if missing:
        raise ValueError(f"solutions_recuperado.csv is missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for (algorithm, seed), group in solutions.groupby(["algorithm", "seed"], sort=False):
        points = group[["productivity_DH", "process_yield_H_Gin"]].to_numpy(float)
        front = unique_nondominated_max(points)
        rows.append({
            "algorithm": algorithm,
            "seed": int(seed),
            "HV_corrected": hypervolume_2d_max(front),
            "Spacing_corrected": spacing_sample(front),
            "n_non_dominated_corrected": int(len(front)),
        })

    corrected = pd.DataFrame(rows)
    out = metrics.merge(corrected, on=["algorithm", "seed"], how="left", validate="one_to_one")
    if out[["HV_corrected", "Spacing_corrected", "n_non_dominated_corrected"]].isna().any().any():
        bad = out[out["HV_corrected"].isna()][["algorithm", "seed"]]
        raise ValueError(f"Missing solution fronts for runs:\n{bad.to_string(index=False)}")

    out["HV_original_file"] = out["HV"]
    out["Spacing_original_file"] = out["Spacing"]
    out["n_non_dominated_original_file"] = out["n_non_dominated"]
    out["HV"] = out.pop("HV_corrected")
    out["Spacing"] = out.pop("Spacing_corrected")
    out["n_non_dominated"] = out.pop("n_non_dominated_corrected")
    return out


def combine_metrics(py_metrics: pd.DataFrame, mat_metrics: pd.DataFrame) -> pd.DataFrame:
    common = [
        "algorithm", "seed", "HV", "Spacing", "n_non_dominated",
        "time_seconds", "n_function_evaluations",
    ]
    for name, frame in [("Python", py_metrics), ("MATLAB", mat_metrics)]:
        missing = set(common).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} metrics missing columns: {sorted(missing)}")

    all_metrics = pd.concat([py_metrics[common], mat_metrics[common]], ignore_index=True)
    counts = all_metrics.groupby("algorithm")["seed"].nunique()
    bad = counts[counts != 30]
    if not bad.empty:
        warnings.warn(f"Expected 30 runs per configuration; observed:\n{bad}")
    return all_metrics


def retention_ratio(row: pd.Series) -> float:
    maximum = 49 if row["algorithm"] == "FMINCON_49" else 100
    return float(row["n_non_dominated"] / maximum)


def descriptive_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHM_ORDER:
        group = metrics[metrics["algorithm"] == algorithm]
        if group.empty:
            continue
        row: dict[str, float | str | int] = {"algorithm": algorithm, "runs": len(group)}
        for metric in ["HV", "Spacing", "n_non_dominated", "time_seconds", "n_function_evaluations"]:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q1"] = float(values.quantile(0.25))
            row[f"{metric}_q3"] = float(values.quantile(0.75))
        row["retention_ratio_mean"] = float(group.apply(retention_ratio, axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(X>Y)-P(X<Y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    greater = np.sum(x[:, None] > y[None, :])
    lower = np.sum(x[:, None] < y[None, :])
    return float((greater - lower) / (len(x) * len(y)))


def delta_magnitude(delta: float) -> str:
    value = abs(delta)
    if value < 0.147:
        return "negligible"
    if value < 0.330:
        return "small"
    if value < 0.474:
        return "medium"
    return "large"


def holm_adjust(p_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Proper Holm adjusted p-values with monotonicity enforcement."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    raw_adjusted = np.array([(m - i) * sorted_p[i] for i in range(m)], dtype=float)
    monotone = np.maximum.accumulate(raw_adjusted)
    monotone = np.minimum(monotone, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = monotone
    return adjusted, adjusted < 0.05


def inferential_analysis(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    post_rows = []

    for metric in TEST_METRICS:
        groups = [
            metrics.loc[metrics["algorithm"] == alg, metric].dropna().to_numpy(float)
            for alg in ALGORITHM_ORDER
        ]
        statistic, p_value = kruskal(*groups)
        global_rows.append({
            "metric": metric,
            "test": "Kruskal-Wallis",
            "statistic": float(statistic),
            "p_value": float(p_value),
        })

        reference = metrics.loc[
            metrics["algorithm"] == REFERENCE_ALGORITHM, metric
        ].dropna().to_numpy(float)

        metric_rows = []
        raw_p = []
        for algorithm in ALGORITHM_ORDER:
            if algorithm == REFERENCE_ALGORITHM:
                continue
            other = metrics.loc[metrics["algorithm"] == algorithm, metric].dropna().to_numpy(float)
            statistic_u, p_value_u = mannwhitneyu(
                reference, other, alternative="two-sided", method="asymptotic"
            )
            delta = cliffs_delta(reference, other)
            metric_rows.append({
                "metric": metric,
                "reference": REFERENCE_ALGORITHM,
                "algorithm": algorithm,
                "statistic_U": float(statistic_u),
                "p_value_raw": float(p_value_u),
                "cliffs_delta": delta,
                "effect_magnitude": delta_magnitude(delta),
                "reference_median": float(np.median(reference)),
                "algorithm_median": float(np.median(other)),
                "median_difference_reference_minus_algorithm": float(
                    np.median(reference) - np.median(other)
                ),
            })
            raw_p.append(float(p_value_u))

        adjusted, reject = holm_adjust(raw_p)
        for row, p_adj, rejected in zip(metric_rows, adjusted, reject):
            row["p_value_holm"] = float(p_adj)
            row["reject_0_05"] = bool(rejected)
            post_rows.append(row)

    return pd.DataFrame(global_rows), pd.DataFrame(post_rows)


def make_boxplots(metrics: pd.DataFrame, output: Path) -> None:
    panels = [
        ("HV", "Hypervolume"),
        ("Spacing", "Spacing (unique objective vectors)"),
        ("n_non_dominated", "Unique non-dominated solutions"),
        ("time_seconds", "Runtime (seconds)"),
        ("n_function_evaluations", "Outer objective evaluations"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, panels):
        data = [
            metrics.loc[metrics["algorithm"] == alg, metric].dropna().to_numpy()
            for alg in ALGORITHM_ORDER
        ]
        ax.boxplot(data, showmeans=True, meanline=True)
        ax.set_xticks(range(1, len(ALGORITHM_ORDER) + 1))
        ax.set_xticklabels(ALGORITHM_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("updated_statistics"))
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    py_metrics = pd.read_csv(input_dir / "metrics_recuperado.csv")
    py_solutions = pd.read_csv(input_dir / "solutions_recuperado.csv")
    mat_metrics = pd.read_csv(input_dir / "matlab_fmincon_metrics_30runs.csv")

    py_corrected = reconstruct_python_metrics(py_metrics, py_solutions)
    combined = combine_metrics(py_corrected, mat_metrics)
    summary = descriptive_summary(combined)
    omnibus, posthoc = inferential_analysis(combined)

    py_corrected.to_csv(output_dir / "python_metrics_corrected.csv", index=False)
    combined.to_csv(output_dir / "metrics_all_corrected.csv", index=False)
    summary.to_csv(output_dir / "summary_all_corrected.csv", index=False)
    omnibus.to_csv(output_dir / "kruskal_wallis_results.csv", index=False)
    posthoc.to_csv(output_dir / "mannwhitney_holm_cliffs_results.csv", index=False)
    make_boxplots(combined, output_dir / "metric_boxplots_corrected.png")

    print("Updated analysis completed.")
    print(f"Outputs: {output_dir.resolve()}")
    print("\nCorrected summary:\n", summary.to_string(index=False))
    print("\nOmnibus tests:\n", omnibus.to_string(index=False))
    print("\nPost-hoc tests:\n", posthoc.to_string(index=False))


if __name__ == "__main__":
    main()
