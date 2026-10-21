"""Collect held-out metrics from every ``summary.json`` under a runs directory.

    python scripts/collect_results.py runs            # prints Markdown tables
    python scripts/collect_results.py runs --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens_lejepa.models import is_encoder

COLUMNS = {
    "classification": ["accuracy", "mean_auroc", "macro_f1"],
    "regression": ["mae_dex", "rmse_dex", "r2", "pearson_r"],
    "super_resolution": ["psnr", "ssim", "mae", "mse"],
}

# Row keys that are never metrics: they identify the run or say which epoch it is.
FIXED_KEYS = ("task", "run", "backbone", "adaptation", "train", "shots", "test", "epoch")


def _columns(task: str, test: dict) -> list[str]:
    """Metric columns for a task; unknown tasks report their numeric metrics."""
    known = COLUMNS.get(task)
    if known is not None:
        return known
    return sorted(
        key for key, value in test.items()
        if key not in FIXED_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def collect(runs: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(runs.glob("*/*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if "test" not in summary:
            continue
        config = json.loads((summary_path.parent / "config.json").read_text())
        test = summary["test"]
        rows.append(
            {
                "task": config["task"],
                "run": summary_path.parent.name,
                "backbone": config["backbone"],
                "adaptation": config["adaptation"] if is_encoder(config["backbone"]) else "-",
                "train": config["train_dataset"],
                "shots": "full" if config["shots"] is None else config["shots"],
                "test": Path(test["test_root"]).name,
                **{key: test.get(key) for key in _columns(config["task"], test)},
            }
        )
    return rows


def _task_columns(task: str, rows: list[dict]) -> list[str]:
    """Metric columns of a task's rows; an unknown task uses the metrics it carries."""
    known = COLUMNS.get(task)
    if known is not None:
        return known
    return [key for key in dict.fromkeys(key for row in rows for key in row) if key not in FIXED_KEYS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = collect(args.runs)
    tasks = list(COLUMNS) + [task for task in dict.fromkeys(row["task"] for row in rows) if task not in COLUMNS]
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        metrics = _task_columns(task, task_rows)
        header = ["run", "backbone", "adaptation", "train", "shots", "test", *metrics]
        print(f"\n### {task}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
        for row in task_rows:
            values = [row.get(key) for key in header]
            cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
            print("| " + " | ".join(cells) + " |")
    if args.csv and rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
