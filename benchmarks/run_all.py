"""Run all six-dataset reference comparisons and write CSV, JSON, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from benchmarks import (
    benchmark_kmeans,
    benchmark_linear_regression,
    benchmark_logistic_regression,
    benchmark_pca,
)
from benchmarks.common import Row, Run, environment_metadata, markdown_table


def update_readme(summary: str) -> None:
    """Replace only the README's explicitly delimited generated comparison table."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    contents = readme.read_text(encoding="utf-8")
    start = "<!-- benchmark-summary:start -->"
    end = "<!-- benchmark-summary:end -->"
    if contents.count(start) != 1 or contents.count(end) != 1:
        raise ValueError("README must contain exactly one benchmark-summary start/end marker pair.")
    prefix, remainder = contents.split(start, maxsplit=1)
    if end not in remainder:
        raise ValueError("README benchmark-summary end marker must follow the start marker.")
    _, suffix = remainder.split(end, maxsplit=1)
    readme.write_text(prefix + start + "\n\n" + summary + "\n" + end + suffix, encoding="utf-8")


def main() -> None:
    """Run as ``python -m benchmarks.run_all --output-dir benchmarks/results``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Refresh the generated README comparison table.",
    )
    args = parser.parse_args()
    rows: list[Row] = []
    runs: list[Run] = []
    with threadpool_limits(limits=1):
        for module in (
            benchmark_linear_regression,
            benchmark_logistic_regression,
            benchmark_kmeans,
            benchmark_pca,
        ):
            module_rows, module_runs = module.run()
            rows.extend(module_rows)
            runs.extend(module_runs)
    metadata = environment_metadata()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "runs": runs, "results": rows}, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report = (
        "# Generated benchmark results\n\n"
        f"Generated UTC: {metadata['generated_at_utc']} · seed: {metadata['seed']} · "
        f"Python: {metadata['python']} · NumPy: {metadata['versions']['numpy']} · "
        f"scikit-learn: {metadata['versions']['scikit-learn']}\n\n"
        "Regenerate with `python -m benchmarks.run_all`. Full environment, settings, "
        "preprocessing, and convergence data are recorded in `results.json`. "
        "Relative difference is |ours − reference| / |reference|; — means a zero reference. "
        "Runtimes are single measurements with one BLAS thread and are not speed claims. "
        "Prediction RMSE and projector distance compare implementations directly and have "
        "zero as their ideal reference; reference ARI has one as its ideal.\n\n"
    )
    (args.output_dir / "results.md").write_text(report + markdown_table(rows), encoding="utf-8")
    summary = [
        row
        for row in rows
        if (
            (row["algorithm"] == "LinearRegression(gd)" and row["metric"] == "test_mse")
            or (row["algorithm"] == "LogisticRegression" and row["metric"] == "test_accuracy")
            or (row["algorithm"] == "KMeans" and row["metric"] == "inertia")
            or (row["algorithm"] == "PCA" and row["metric"] == "reconstruction_mse")
        )
    ]
    summary_table = markdown_table(summary)
    (args.output_dir / "summary.md").write_text(summary_table, encoding="utf-8")
    if args.update_readme:
        update_readme(summary_table)
    print(f"Wrote {len(rows)} metric comparisons from {len(runs)} runs to {args.output_dir}")
    print(summary_table)


if __name__ == "__main__":
    main()
