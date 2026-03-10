#!/usr/bin/env python3
"""
Launch training and compute metrics in batch mode.
Combines launch_and_metrics.sh and launch_and_metrics_batch.sh functionality.
"""

import os
import sys
import subprocess
import re
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

# ==========================
# Configuration
# ==========================


# Training config
CONFIG = "configs/dge_camera-selection.yaml"
GPU = "1"

# Batch settings
NUM_RUNS = 5  # Number of times to run launch + metrics

# Task-specific overrides
# PROMPT = "Turn him into spider man with a mask"
# PROMPT = "Turn the man into a clown"
PROMPT = "Make the man wear fashion sunglasses"
PROMPT = "Turn the man's fleece jacket into a leather jacket"

DATA_SOURCE = "/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE = "/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
GUIDANCE_SCALE = "12.5"
SEG_PROMPT = "A man"

# TARGET_PROMPT = "A man with curly hair in a checkered cloth"
# TARGET_PROMPT = "A spider man with a mask and curly hair"
TARGET_PROMPT = "Fashion sunglasses"
TARGET_PROMPT = "A man with a leather jacket"


PROMPT = "Make the man wear fashion sunglasses"
PROMPT = "Turn the man's fleece jacket into a leather jacket"

DATA_SOURCE = "/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE = "/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
GUIDANCE_SCALE = "12.5"
SEG_PROMPT = "A man"

# TARGET_PROMPT = "A man with curly hair in a checkered cloth"
# TARGET_PROMPT = "A spider man with a mask and curly hair"
TARGET_PROMPT = "Fashion sunglasses"
TARGET_PROMPT = "A man with a leather jacket"

MASK_THRES = "0.6"
LAMBDA_D = "10.0"
MAX_VIEW_NUM = "25"
MAX_EDIT_VIEW_NUM = "20"
EDIT_VIEW_SELECTION_STRATEGY = "y-axis"  # row, quadrant, manual-20, manual-15, random, depth
CAMERA_UPDATE_PER_STEP = "1500"
MASK_UPDATE_AT_STEP = "400" # -1: wo-MaskUpdate, 400: w-MaskUpdate
NAME = f"camera-selection/w-MaskUpdate/iter1/lambda_d{LAMBDA_D}/{EDIT_VIEW_SELECTION_STRATEGY}"

# Metrics config
GT_DIR = "/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"
STYLE_PROMPT = TARGET_PROMPT  # leave empty "" to disable
STYLE_PROMPT = "A man with fashion sunglasses"  # leave empty "" to disable
STYLE_PROMPT = "A man with a leather jacket"  # leave empty "" to disable

STYLE_IMAGE = ""  # set to an image path to use style image instead of text
OBJECT_PROMPT = "A Man without fashion sunglasses"   # default: "a Photo"
OBJECT_PROMPT = "A man with a fleece jacket"   # default: "a Photo"

INTERVAL = 1  # temporal interval k for consistency metrics
DEVICE = "cuda"  # cuda or cpu
MAX_STEPS = "1500"

# Log directory for batch runs
BATCH_LOG_DIR = "nohups/batch_runs"

# ==========================


def get_script_dir() -> Path:
    """Get the directory where this script is located."""
    return Path(__file__).parent.absolute()


def get_root_dir() -> Path:
    """Get the project root directory (two levels up from script)."""
    return get_script_dir().parent.parent


def build_launch_cmd() -> List[str]:
    """Build the launch.py command."""
    return [
        "python", "launch.py",
        "--config", CONFIG,
        "--train",
        "--gpu", GPU,
        f"trainer.max_steps={MAX_STEPS}",
        f"system.prompt_processor.prompt={PROMPT}",
        f"data.source={DATA_SOURCE}",
        f"system.guidance.guidance_scale={GUIDANCE_SCALE}",
        f"system.gs_source={GS_SOURCE}",
        f"system.seg_prompt={SEG_PROMPT}",
        f"system.target_prompt={TARGET_PROMPT}",
        f"system.mask_thres={MASK_THRES}",
        f"system.loss.lambda_d={LAMBDA_D}",
        f"data.max_view_num={MAX_VIEW_NUM}",
        f"data.max_edit_view_num={MAX_EDIT_VIEW_NUM}",
        f"data.edit_view_selection_strategy={EDIT_VIEW_SELECTION_STRATEGY}",
        f"system.guidance.edit_view_selection_strategy={EDIT_VIEW_SELECTION_STRATEGY}",
        f"system.camera_update_per_step={CAMERA_UPDATE_PER_STEP}",
        f"system.mask_update_at_step={MASK_UPDATE_AT_STEP}",
        f"name={NAME}",
    ]


def find_save_directory(output: str) -> Optional[Path]:
    """Find the save directory from launch.py output."""
    # Look for "Test results saved to ..." pattern
    pattern = r"Test results saved to (.+)"
    matches = re.findall(pattern, output)
    
    if matches:
        save_dir = Path(matches[-1].strip())
        if save_dir.exists() and save_dir.is_dir():
            return save_dir
    
    # Fallback: try to find latest directory based on config
    exp_root_dir = Path("/data/users/jaeyeonpark/DGE-outputs")
    exp_dir = exp_root_dir / NAME / str(MAX_VIEW_NUM)
    
    if exp_dir.exists():
        # Find latest trial directory (with @timestamp pattern)
        trial_dirs = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and "@" in d.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if trial_dirs:
            save_dir = trial_dirs[0] / "save"
            if save_dir.exists():
                return save_dir
    
    # Try without MAX_VIEW_NUM
    exp_dir = exp_root_dir / NAME
    if exp_dir.exists():
        trial_dirs = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and "@" in d.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if trial_dirs:
            save_dir = trial_dirs[0] / "save"
            if save_dir.exists():
                return save_dir
    
    return None


def find_render_directory(save_dir: Path) -> Optional[Path]:
    """Find the test render directory."""
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if render_dir.exists():
        return render_dir
    
    # Try to find any it*-test directory
    test_dirs = list(save_dir.glob("it*-test"))
    if test_dirs:
        return test_dirs[0]
    
    return None


def build_metrics_cmd(render_dir: Path) -> List[str]:
    """Build the metrics.py command."""
    # Set device with GPU number if using CUDA
    device = DEVICE
    if device == "cuda":
        device = f"cuda:{GPU}"
    
    cmd = [
        "python", "metrics.py",
        "--gt", GT_DIR,
        "--render", str(render_dir),
        "--device", device,
        "--interval", str(INTERVAL),
        "--object_prompt", OBJECT_PROMPT,
    ]
    
    if STYLE_IMAGE:
        cmd.extend(["--style_image", STYLE_IMAGE])
    elif STYLE_PROMPT:
        cmd.extend(["--style_prompt", STYLE_PROMPT])
    
    return cmd


def run_single_iteration(run_num: int, total_runs: int, log_file: Optional[Path] = None) -> Tuple[bool, str]:
    """Run a single iteration of launch + metrics."""
    root_dir = get_root_dir()
    os.chdir(root_dir)
    
    print(f"\n{'='*50}")
    print(f"[Run {run_num}/{total_runs}] Starting...")
    print(f"{'='*50}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Launch training
    print(f"{'='*50}")
    print("[1/3] Launch training...")
    print(f"{'='*50}")
    
    launch_cmd = build_launch_cmd()
    print(f"Running: {' '.join(launch_cmd)}")
    print()
    
    launch_output = ""
    try:
        result = subprocess.run(
            launch_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        launch_output = result.stdout + result.stderr
        
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("=== LAUNCH OUTPUT ===\n")
                f.write(launch_output)
                f.write("\n")
        
        print(launch_output)
        
        if result.returncode != 0:
            print("\nTraining failed. Check the output above.")
            return False, launch_output
        
        print("\nTraining completed successfully.")
    except Exception as e:
        print(f"\nError running launch.py: {e}")
        return False, str(e)
    
    # Step 2: Discover render directory
    print()
    print(f"{'='*50}")
    print("[2/3] Discover render directory from output...")
    print(f"{'='*50}")
    
    save_dir = find_save_directory(launch_output)
    if not save_dir:
        print("Error: Could not find save directory.")
        print("Expected pattern: [INFO] Test results saved to <path>")
        return False, "Could not find save directory"
    
    print(f"Found save directory: {save_dir}")
    
    render_dir = find_render_directory(save_dir)
    if not render_dir:
        print(f"Error: Could not find test render directory in {save_dir}")
        print(f"Expected: {save_dir}/it{MAX_STEPS}-test or similar")
        return False, "Could not find render directory"
    
    print(f"Using render directory: {render_dir}")
    print()
    
    # Validate directories
    if not Path(GT_DIR).exists():
        print(f"Error: GT_DIR not found: {GT_DIR}")
        return False, f"GT_DIR not found: {GT_DIR}"
    
    if not render_dir.exists():
        print(f"Error: RENDER_DIR not found: {render_dir}")
        return False, f"RENDER_DIR not found: {render_dir}"
    
    # Step 3: Run metrics
    print(f"{'='*50}")
    print("[3/3] Run metrics...")
    print(f"{'='*50}")
    
    metrics_cmd = build_metrics_cmd(render_dir)
    print(f"Running: {' '.join(metrics_cmd)}")
    print()
    
    try:
        result = subprocess.run(
            metrics_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        metrics_output = result.stdout + result.stderr
        
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("=== METRICS OUTPUT ===\n")
                f.write(metrics_output)
                f.write("\n")
        
        print(metrics_output)
        
        if result.returncode != 0:
            print("\nMetrics calculation failed.")
            return False, metrics_output
        
        print()
        print(f"{'='*50}")
        print("Done!")
        print(f"{'='*50}")
        
        return True, ""
    except Exception as e:
        print(f"\nError running metrics.py: {e}")
        return False, str(e)


def main():
    """Main function to run batch iterations."""
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_log_base = Path(BATCH_LOG_DIR) / timestamp
    batch_log_base.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Batch Run Configuration")
    print("="*50)
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Config: {CONFIG}")
    print(f"GPU: {GPU}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Name: {NAME}")
    print(f"Log directory: {batch_log_base}")
    print("="*50)
    print()
    
    # Track results
    success_count = 0
    fail_count = 0
    failed_runs = []
    
    # Run each iteration
    for run in range(1, NUM_RUNS + 1):
        log_file = batch_log_base / f"run_{run}.log"
        
        success, error_msg = run_single_iteration(run, NUM_RUNS, log_file)
        
        if success:
            success_count += 1
            print(f"\n[Run {run}/{NUM_RUNS}] ✓ Completed successfully")
        else:
            fail_count += 1
            failed_runs.append(run)
            print(f"\n[Run {run}/{NUM_RUNS}] ✗ Failed")
            if error_msg:
                print(f"Error: {error_msg}")
        
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Add a small delay between runs
        if run < NUM_RUNS:
            print("Waiting 5 seconds before next run...")
            import time
            time.sleep(5)
    
    # Print summary
    print()
    print("="*50)
    print("Batch Run Summary")
    print("="*50)
    print(f"Total runs: {NUM_RUNS}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    
    if fail_count > 0:
        print(f"Failed runs: {failed_runs}")
        print()
        print(f"Check logs in: {batch_log_base}")
        sys.exit(1)
    else:
        print("All runs completed successfully!")
        print()
        print(f"All logs saved in: {batch_log_base}")
        sys.exit(0)


if __name__ == "__main__":
    main()

