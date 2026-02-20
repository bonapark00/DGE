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
GPU = "2"

# Batch settings
NUM_RUNS = 1  # Number of times to run launch + metrics

# Task-specific overrides
# PROMPT = "Turn him into spider man with a mask"
PROMPT = "Turn the man's fleece jacket into a leather jacket"
PROMPT = "Wear the man a brown-colored cowboy hat"
PROMPT = "Make him look like Vincent Van Gogh"
PROMPT = "Turn him into the Tolkien Elf"
PROMPT = "Turn his face into a spider man with a mask"
PROMPT = "Make the man wear black sunglasses"
PROMPT = "Turn the man into a clown"
PROMPT = "Make him wear large, black over-ear headphones with thick padded ear cups and a prominent headband sitting comfortably on his head, clearly visible in the image."
PROMPT = "Make him wear blue earrings on his ears"
PROMPT = "Add rounded earrings to his ears"

DATA_SOURCE = "/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE = "/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
GUIDANCE_SCALE = "12.5"
SEG_PROMPT = "A man"
MMR_SEG_PROMPT = SEG_PROMPT  # for text-based segmentation

# TARGET_PROMPT = "A man with curly hair in a checkered cloth"
# TARGET_PROMPT = "A spider man with a mask and curly hair"
TARGET_PROMPT = "A man with a leather jacket"
TARGET_PROMPT = "A man looking like Vincent Van Gogh"
TARGET_PROMPT = "A Tolkien Elf"
TARGET_PROMPT = "Fashion sunglasses"
TARGET_PROMPT = "A spider man with a mask"
TARGET_PROMPT = "A man with black sunglasses"
TARGET_PROMPT = "A clown"
TARGET_PROMPT = "Each ear with blue earrings"
TARGET_PROMPT = "A man wearing headphones"
TARGET_PROMPT = "A man with blue rounded earrings"


MASK_THRES = "0.6"
LAMBDA_D = "0.0"
MAX_VIEW_NUM = "25"
MAX_EDIT_VIEW_NUM = "20"
EDIT_VIEW_SELECTION_STRATEGY = "mmr"  # row, quadrant, manual-20, manual-15, random, depth, lens

LENS_USE_IP2P_SCORING = "true"  # match generate_by_lens: use IP2P for SAGE probing
LENS_IP2P_STEPS = "5"  # IP2P num_inference_steps for SAGE probing (기존은 20임)
LENS_IP2P_GUIDANCE_SCALE = "12.5"  # IP2P guidance_scale for SAGE probing
LENS_IP2P_IMAGE_GUIDANCE_SCALE = "1.5"  # IP2P image_guidance_scale for SAGE probing
LENS_DISTANCE_MULTIPLIERS = "2.0,2.5,3.0,4.0,5.0,6.0"  # match run_generate_by_lens.sh
LENS_CONE_HALF_ANGLE_DEG = "60"  # Cone constraint around COLMAP mean direction (default 60)

CAMERA_UPDATE_PER_STEP = "1500"
MASK_UPDATE_AT_STEP = "400" # -1: wo-MaskUpdate, 400: w-MaskUpdate
MASK_MAX_RATIO = "0.6"  # Skip views where mask covers > this fraction (0~1)
MASK_MIN_RATIO = "0.01"  # Skip views where mask covers < this fraction (likely failed seg)
MASK_OUTLIER_IQR = "1.5"  # 작으면 허용 구간이 좁아짐 → 더 많은 뷰가 outlier로 제외됨
MASK_UPDATE_VIEW_NUM = "5"  # MAX_VIEW_NUM - MAX_EDIT_VIEW_NUM 보다는 작아야함
PRUNE_FLOATER_AT_STEP = "600"  # -1: disabled, otherwise prune at this step
NAME = f"camera-selection/w-MaskUpdate/iter1/lambda_d{LAMBDA_D}/{EDIT_VIEW_SELECTION_STRATEGY}"

# Metrics config: origin renders from original PLY (or fallback path)
ORIGIN_RENDER_BASE = "/data/users/jaeyeonpark/DGE-outputs/origin_render"
ORIGIN_RENDER_DIR = "/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"
USE_ORIGIN_RENDER = False  # True: use origin renders as GT, run render_origin_views if needed; False: use GT_DIR as-is
GT_DIR = ORIGIN_RENDER_DIR if USE_ORIGIN_RENDER else "/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"
STYLE_TARGET_PROMPT = TARGET_PROMPT  # 목표 스타일 (편집 후); leave empty "" to disable
# STYLE_TARGET_PROMPT = "A man with fashion sunglasses"  # leave empty "" to disable
# STYLE_TARGET_PROMPT = "A man with a leather jacket"  # leave empty "" to disable
# STYLE_TARGET_PROMPT = "A man looking like Vincent Van Gogh"  # leave empty "" to disable
# STYLE_TARGET_PROMPT = "A Tolkien Elf"  # leave empty "" to disable

STYLE_IMAGE = ""  # set to an image path to use style image instead of text
STYLE_SOURCE_PROMPT = "A Man without fashion sunglasses"   # 편집 전/원본; default: "a Photo"
# STYLE_SOURCE_PROMPT = "A man with a fleece jacket"   # default: "a Photo"
# STYLE_SOURCE_PROMPT = "A man"

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
        f"data.mmr_seg_prompt={MMR_SEG_PROMPT}",
        f"system.target_prompt={TARGET_PROMPT}",
        f"system.mask_thres={MASK_THRES}",
        f"system.mask_max_ratio={MASK_MAX_RATIO}",
        f"system.mask_min_ratio={MASK_MIN_RATIO}",
        f"system.mask_outlier_iqr={MASK_OUTLIER_IQR}",
        f"system.loss.lambda_d={LAMBDA_D}",
        f"data.max_view_num={MAX_VIEW_NUM}",
        f"data.max_edit_view_num={MAX_EDIT_VIEW_NUM}",
        f"data.edit_view_selection_strategy={EDIT_VIEW_SELECTION_STRATEGY}",
        f"data.lens_ply_path={GS_SOURCE}",  # required for lens strategy
        f"data.lens_seg_prompt={SEG_PROMPT}",
        f"data.lens_edit_prompt={PROMPT}",
        f"data.lens_use_ip2p_scoring={LENS_USE_IP2P_SCORING}",
        f"data.lens_ip2p_steps={LENS_IP2P_STEPS}",
        f"data.lens_ip2p_guidance_scale={LENS_IP2P_GUIDANCE_SCALE}",
        f"data.lens_ip2p_image_guidance_scale={LENS_IP2P_IMAGE_GUIDANCE_SCALE}",
        f"data.lens_distance_multipliers={LENS_DISTANCE_MULTIPLIERS}",
        f"data.lens_cone_half_angle_deg={LENS_CONE_HALF_ANGLE_DEG}",
        f"system.guidance.edit_view_selection_strategy={EDIT_VIEW_SELECTION_STRATEGY}",
        f"system.camera_update_per_step={CAMERA_UPDATE_PER_STEP}",
        f"system.mask_update_at_step={MASK_UPDATE_AT_STEP}",
        f"system.mask_update_view_num={MASK_UPDATE_VIEW_NUM}",
        f"system.prune_floater_at_step={PRUNE_FLOATER_AT_STEP}",
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
        "--style_source_prompt", STYLE_SOURCE_PROMPT,
    ]
    
    if STYLE_IMAGE:
        cmd.extend(["--style_image", STYLE_IMAGE])
    elif STYLE_TARGET_PROMPT:
        cmd.extend(["--style_target_prompt", STYLE_TARGET_PROMPT])
    
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
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            launch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        if log_file:
            log_f = open(log_file, "a", encoding="utf-8")
            log_f.write("=== LAUNCH OUTPUT ===\n")
        
        try:
            for line in process.stdout:
                print(line, end='', flush=True)  # Print immediately
                output_lines.append(line)
                if log_file:
                    log_f.write(line)
                    log_f.flush()
        finally:
            if log_file:
                log_f.write("\n")
                log_f.close()
        
        process.wait()
        launch_output = ''.join(output_lines)
        
        if process.returncode != 0:
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
    
    # Step 2.5: Ensure origin renders exist (for metrics GT when USE_ORIGIN_RENDER)
    if USE_ORIGIN_RENDER:
        origin_render_dir = Path(ORIGIN_RENDER_DIR)
        cameras_pt = save_dir / "cameras_for_origin.pt"
        n_origin = len(list(origin_render_dir.glob("*.png"))) if origin_render_dir.exists() else 0
        n_render = len(list(render_dir.glob("*.png")))
        need_origin_render = n_origin == 0 or n_origin != n_render
        if need_origin_render:
            print(f"{'='*50}")
            print("[2.5/3] Render origin views from original PLY...")
            print(f"{'='*50}")
            if not cameras_pt.exists():
                print(f"Warning: cameras_for_origin.pt not found at {cameras_pt}")
                print("  Skipping origin render. Metrics will fail if GT_DIR is required.")
            else:
                render_cmd = [
                    "python", str(get_root_dir() / "gaussiansplatting" / "render_origin_views.py"),
                    "--gs_source", GS_SOURCE,
                    "--cameras_path", str(cameras_pt),
                    "--out_dir", str(origin_render_dir),
                    "--force",  # overwrite when count changed (val 8 vs test 31)
                ]
                print(f"Running: {' '.join(render_cmd)}")
                try:
                    result = subprocess.run(render_cmd, cwd=get_root_dir(), capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"Origin render failed: {result.stderr}")
                        return False, f"Origin render failed: {result.stderr}"
                    print(result.stdout)
                except Exception as e:
                    print(f"Error running render_origin_views: {e}")
                    return False, str(e)
        else:
            print(f"Origin renders already exist: {origin_render_dir}")
        print()
    
    # Validate directories
    if USE_ORIGIN_RENDER:
        origin_render_dir = Path(ORIGIN_RENDER_DIR)
        if not origin_render_dir.exists() or not list(origin_render_dir.glob("*.png")):
            print(f"Error: GT (origin render) directory empty or missing: {origin_render_dir}")
            return False, f"GT_DIR not found or empty: {origin_render_dir}"
    else:
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
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            metrics_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        if log_file:
            log_f = open(log_file, "a", encoding="utf-8")
            log_f.write("=== METRICS OUTPUT ===\n")
        
        try:
            for line in process.stdout:
                print(line, end='', flush=True)  # Print immediately
                output_lines.append(line)
                if log_file:
                    log_f.write(line)
                    log_f.flush()
        finally:
            if log_file:
                log_f.write("\n")
                log_f.close()
        
        process.wait()
        metrics_output = ''.join(output_lines)
        
        if process.returncode != 0:
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

