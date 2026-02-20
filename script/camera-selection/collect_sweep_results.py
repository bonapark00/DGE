#!/usr/bin/env python3
"""
Collect eval_clip.txt from all result folders under a base directory and show
a table sorted by a chosen metric so you can find the best config.

Usage:
  # Default: sort by clip_dir_similarity (maximize), base dir = current
  python script/camera-selection/collect_sweep_results.py BASE_DIR

  # Sort by another metric (maximize)
  python script/camera-selection/collect_sweep_results.py BASE_DIR --metric clip_f_scaled

  # Save to CSV
  python script/camera-selection/collect_sweep_results.py BASE_DIR --csv results.csv

  # Example base dirs
  #   Sweep:  /data/users/jaeyeonpark/DGE-outputs/sweep/in2n-GSEditor/face
  #   Single: /data/users/jaeyeonpark/DGE-outputs/camera-selection/w-MaskUpdate/iter1/lambda_d5.0/lens/in2n-GSEditor/face/25
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# Metrics we parse from eval_clip.txt (same keys as in the file)
METRIC_KEYS = [
    "clip_dir_consistency",
    "clip_f_scaled",
    "clip_score",
    "clip_dir_similarity",
]
# Line patterns: "CLIP directional consistency: 13.5" etc.
PATTERNS = {
    "clip_dir_consistency": r"^CLIP directional consistency:\s*([-\d.]+)",
    "clip_f_scaled": r"^CLIP_F \(scaled\):\s*([-\d.]+)",
    "clip_score": r"^CLIP Score:\s*([-\d.]+)",
    "clip_dir_similarity": r"^CLIP directional similarity:\s*([-\d.]+)",
}


def parse_eval_clip(path: Path) -> dict:
    """Parse an eval_clip.txt file; return dict of metric name -> float."""
    text = path.read_text()
    result = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            result[key] = float(m.group(1))
    return result


def config_from_path(path: Path, base_dir: Path) -> dict:
    """Infer lambda_d, strategy, lambda_ism from full path (any folder order)."""
    path_str = path.as_posix()
    config = {"lambda_d": "", "strategy": "", "lambda_ism": "", "run": path.parent.name}
    # Match lambda_d3, lambda_d5.0, lambda_d10 etc.
    m = re.search(r"lambda_d([\d.]+)", path_str)
    if m:
        config["lambda_d"] = m.group(1)
    for s in ("lens", "random"):
        if f"/{s}/" in path_str or path_str.endswith(f"/{s}"):
            config["strategy"] = s
            break
    # Match lambda_ism0, lambda_ism0.0001, lambda_ism1e-05 etc.
    m = re.search(r"lambda_ism([\d.e+-]+)", path_str)
    if m:
        config["lambda_ism"] = m.group(1)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Collect eval_clip.txt from result folders and show table sorted by metric"
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Base directory to search recursively for eval_clip.txt",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="clip_dir_similarity",
        choices=METRIC_KEYS,
        help="Metric to sort by (default: clip_dir_similarity). Higher is better.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="If set, write results to this CSV file",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.is_dir():
        print(f"Error: Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all eval_clip.txt
    eval_files = list(base_dir.rglob("eval_clip.txt"))
    if not eval_files:
        print(f"No eval_clip.txt found under {base_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in eval_files:
        metrics = parse_eval_clip(p)
        if not metrics:
            continue
        config = config_from_path(p, base_dir)
        row = {**config, **metrics, "_path": str(p)}
        rows.append(row)

    if not rows:
        print("No valid eval_clip.txt content found.", file=sys.stderr)
        sys.exit(1)

    # Sort by chosen metric (higher is better)
    def sort_key(r):
        v = r.get(args.metric)
        if v is None:
            return float("-inf")
        return v

    rows.sort(key=sort_key, reverse=True)

    # Print table
    col_config = ["lambda_d", "strategy", "lambda_ism", "run"]
    col_metrics = [k for k in METRIC_KEYS if any(r.get(k) is not None for r in rows)]
    headers = col_config + col_metrics
    col_widths = [max(len(str(h)), 4) for h in headers]
    for i, r in enumerate(rows):
        for j, h in enumerate(headers):
            col_widths[j] = max(col_widths[j], len(str(r.get(h, ""))))

    sep = " | "
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(f"\nFound {len(rows)} result(s). Sorted by {args.metric} (higher is better).\n")
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(sep.join(str(r.get(h, "")).ljust(col_widths[i]) for i, h in enumerate(headers)))

    print("\n" + "=" * len(header_line))
    best = rows[0]
    print(f"Best config (by {args.metric}):")
    print(f"  lambda_d={best.get('lambda_d')}, strategy={best.get('strategy')}, lambda_ism={best.get('lambda_ism')}")
    print(f"  run: {best.get('run')}")
    for k in col_metrics:
        print(f"  {k}: {best.get(k)}")
    print(f"  path: {best.get('_path')}")

    if args.csv:
        import csv
        out_path = Path(args.csv)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers + ["_path"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
