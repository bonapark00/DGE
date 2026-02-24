#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Run metrics only (Step 2 + 3 of launch_and_metrics).
Render directory is specified at the top; no training/launch step.
"""

import os
import subprocess
from pathlib import Path
from typing import List

OUTPUT_DIR = "/data/users/jaeyeonpark/DGE-outputs/camera-selection/w-MaskUpdate/iter1/lambda_d0.0/lens-multiview/3d-ovs/covered_desk/25/Make_the_pooh_look_like_a_robot@20260224-164458/"

MAX_STEPS = "1500"
GPU = "2"
DATA_TYPE = "3d-ovs"
DATA_NAME = "covered_desk"
ORIGIN_RENDER_BASE = "/data/users/jaeyeonpark/DGE-outputs/origin_render"
EDIT_VIEW_SELECTION_STRATEGY = "lens"
ORIGIN_RENDER_DIR = f"{ORIGIN_RENDER_BASE}/{DATA_TYPE}/{DATA_NAME}/{EDIT_VIEW_SELECTION_STRATEGY}"
USE_ORIGIN_RENDER = False  # True: use origin renders as GT (like 3d-ovs); False: use training renders
GT_DIR = ORIGIN_RENDER_DIR if USE_ORIGIN_RENDER else f"/data/users/jaeyeonpark/3dgs-trained/{DATA_TYPE}/{DATA_NAME}/train/ours_30000/renders"
STYLE_IMAGE = ""
STYLE_TARGET_PROMPT = "A man with a leather jacket"
STYLE_SOURCE_PROMPT = "A man with a fleece jacket"
INTERVAL = 1
DEVICE = "cuda"


def get_script_dir() -> Path:
    return Path(__file__).parent.absolute()


def get_root_dir() -> Path:
    return get_script_dir().parent.parent.parent


def build_metrics_cmd(render_dir: Path) -> List[str]:
    device = f"cuda:{GPU}" if DEVICE == "cuda" else DEVICE
    cmd = [
        "python", "metrics.py",
        "--gt", GT_DIR,
        "--render", str(render_dir),
        "--device", device,
        "--interval", str(INTERVAL),
        "--style_source_prompt", STYLE_SOURCE_PROMPT,
    ]
    if STYLE_IMAGE:
        cmd.extend(["--style_image", STYLE_IMAGE])
    elif STYLE_TARGET_PROMPT:
        cmd.extend(["--style_target_prompt", STYLE_TARGET_PROMPT])
    return cmd


def main():
    root_dir = get_root_dir()
    os.chdir(root_dir)

    save_dir = Path(OUTPUT_DIR.rstrip("/")) / "save"
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if not render_dir.exists():
        test_dirs = list(save_dir.glob("it*-test"))
        render_dir = test_dirs[0] if test_dirs else render_dir
    render_dir = render_dir.resolve()
    if not render_dir.exists():
        print(f"Error: render dir not found: {render_dir} (OUTPUT_DIR/save/it{{N}}-test)")
        return 1
    if not list(render_dir.glob("*.png")):
        print(f"Warning: No .png files in {render_dir}")

    if not Path(GT_DIR).exists():
        print(f"Error: GT_DIR not found: {GT_DIR}")
        return 1

    print("=" * 50)
    print("Run metrics only")
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
    exit(main())
