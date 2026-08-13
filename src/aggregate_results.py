"""
Aggregates results/<scene>/metrics.json across all scenes into:
  - results/comparison_table.csv
  - results/comparison_table.md   (paste straight into the report)
  - results/comparison_figures.png (bar charts per metric, grouped by method)

Usage (run from project root, after evaluate.py has been run for every scene):
    python src/aggregate_results.py
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

METHODS = ["3dgs", "colmap_dense_mvs", "poisson_mesh"]
METHOD_LABELS = {"3dgs": "3D Gaussian Splatting", "colmap_dense_mvs": "COLMAP Dense MVS", "poisson_mesh": "Poisson Mesh"}
METRIC_KEYS = ["PSNR", "SSIM", "coverage", "reconstruction_time_sec", "train_time_sec",
               "peak_gpu_memory_mb", "peak_memory_mb"]


def load_all_scene_metrics():
    rows = []
    for scene_dir in sorted(RESULTS_ROOT.glob("scene*")):
        metrics_path = scene_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        scene = data["scene"]
        for method in METHODS:
            m = data.get(method, {})
            row = {"scene": scene, "method": METHOD_LABELS[method]}
            for k in METRIC_KEYS:
                row[k] = m.get(k)
            rows.append(row)
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, out_path: Path):
    plot_metrics = ["PSNR", "SSIM", "coverage",
                     "reconstruction_time_sec" if "reconstruction_time_sec" in df else "train_time_sec"]
    plot_metrics = [m for m in plot_metrics if m in df.columns]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 4))
    if len(plot_metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, plot_metrics):
        pivot = df.pivot(index="scene", columns="method", values=metric)
        pivot.plot(kind="bar", ax=ax, legend=(metric == plot_metrics[0]))
        ax.set_title(metric)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    df = load_all_scene_metrics()
    if df.empty:
        print("No results/<scene>/metrics.json files found yet -- run src/evaluate.py per scene first.")
        return

    RESULTS_ROOT.mkdir(exist_ok=True)
    csv_path = RESULTS_ROOT / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    md_path = RESULTS_ROOT / "comparison_table.md"
    md_path.write_text(df.to_markdown(index=False))
    print(f"Wrote {md_path}")

    make_figures(df, RESULTS_ROOT / "comparison_figures.png")

    # Simple text summary useful for the "Discussion and summary" report section
    print("\n=== Per-metric averages across all scenes ===")
    numeric_cols = [c for c in METRIC_KEYS if c in df.columns]
    print(df.groupby("method")[numeric_cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
