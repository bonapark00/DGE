#!/usr/bin/env python3
"""
Run CLIP-based metrics only (no training step).

This script mirrors the behavior of the 3d-ovs `run_metrics_only.py`
from DGE-camera-selection, adapted to this repository layout.
"""

import os
import subprocess
from pathlib import Path
from typing import List


# ==========================
# User-configurable settings
# ==========================

# Root directory of a single trial (directory that contains `save/it{MAX_STEPS}-test/`)
# Example:
#   /data/users/jaeyeonpark/DGE-outputs/camera-selection/w-MaskUpdate/iter1/...
OUTPUT_DIR = (
    "/data/users/jaeyeonpark/DGE-outputs/"
    "camera-selection/w-MaskUpdate/iter1/lambda_d10.0/y-axis/25/"
    "Turn_the_man_into_a_clown@20250101-000000/"
)

MAX_STEPS = "1500"
GPU = "1"

# Ground-truth directory (same as in `launch_and_metrics.py`)
GT_DIR = "/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"

# Style condition
STYLE_IMAGE = ""  # set to an image path to use style image instead of text
STYLE_TARGET_PROMPT = "A man with a leather jacket"
STYLE_SOURCE_PROMPT = "A man with a fleece jacket"

INTERVAL = 1
DEVICE = "cuda"  # "cuda" or "cpu"


def get_script_dir() -> Path:
    return Path(__file__).parent.absolute()


def get_root_dir() -> Path:
    # script/camera-selection -> repo root
    return get_script_dir().parent.parent


def build_metrics_cmd(render_dir: Path) -> List[str]:
    device = f"cuda:{GPU}" if DEVICE == "cuda" else DEVICE
    cmd = [
        "python",
        "metrics.py",
        "--gt",
        GT_DIR,
        "--render",
        str(render_dir),
        "--device",
        device,
        "--interval",
        str(INTERVAL),
        "--style_source_prompt",
        STYLE_SOURCE_PROMPT,
    ]
    if STYLE_IMAGE:
        cmd.extend(["--style_image", STYLE_IMAGE])
    elif STYLE_TARGET_PROMPT:
        cmd.extend(["--style_target_prompt", STYLE_TARGET_PROMPT])
    return cmd


def main() -> int:
    root_dir = get_root_dir()
    os.chdir(root_dir)

    save_dir = Path(OUTPUT_DIR.rstrip("/")) / "save"
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if not render_dir.exists():
        test_dirs = list(save_dir.glob("it*-test"))
        render_dir = test_dirs[0] if test_dirs else render_dir
    render_dir = render_dir.resolve()

    if not render_dir.exists():
        print(
            f"Error: render dir not found: {render_dir} "
            f"(OUTPUT_DIR/save/it{{N}}-test)"
        )
        return 1

    if not list(render_dir.glob("*.png")):
        print(f"Warning: No .png files in {render_dir}")

    if not Path(GT_DIR).exists():
        print(f"Error: GT_DIR not found: {GT_DIR}")
        return 1

    print("=" * 50)
    print("Run CLIP metrics only")
    print("=" * 50)
    print(f"Render: {render_dir}")
    print(f"GT:     {GT_DIR}")
    print("=" * 50)

    metrics_cmd = build_metrics_cmd(render_dir)
    print(f"Running: {' '.join(metrics_cmd)}\n")

    result = subprocess.run(metrics_cmd)
    if result.returncode != 0:
        print("\nMetrics failed.")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

