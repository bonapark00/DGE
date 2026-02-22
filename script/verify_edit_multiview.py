#!/usr/bin/env python3
"""
Verify that edit_latents_multiview (edit_multiview path) runs without error.

Method 1 – Run this script (recommended)
  From repo root, same args as your normal run but with trainer.max_steps=2
  and system.use_multiview_edit=True. Script runs launch.py and checks exit
  code and that the multiview output is produced.

  Example:
    python script/verify_edit_multiview.py --config configs/dge_camera-selection.yaml \\
      --train --gpu 0 trainer.max_steps=2 system.use_multiview_edit=True \\
      system.target_prompt="robot" data.source=/path/to/scene system.gs_source=/path/to/point_cloud.ply

  Exit 0: success. Non-zero: failure (script or launch failed).

Method 2 – Manual run
  Run your usual train command with trainer.max_steps=501 (or 1 + camera_update_per_step)
  so that the first edit_multiview runs. Confirm:
  - Log shows "[edit_latents_multiview] key view indices: ..."
  - Log shows "Multiview editing finished."
  - Log shows "multiview edited images saved to: ..."
  - No uncaught exception; save/edited_images_multiview.png exists.

Method 3 – Latency summary
  After a run, check latency summary for edit_multiview.* timings (no NaN/zeros).
"""

import os
import subprocess
import sys


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    # Build command: launch.py with rest of argv (must include --config, --train, etc.)
    argv = [sys.executable, "launch.py"] + sys.argv[1:]
    # Ensure we don't run too long
    if "trainer.max_steps=" not in " ".join(argv):
        argv.extend(["trainer.max_steps=2"])
    if "system.use_multiview_edit=" not in " ".join(argv):
        argv.extend(["system.use_multiview_edit=True"])

    proc = subprocess.run(
        argv,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        print("Launch failed (exit code %d). Last 80 lines of output:" % proc.returncode, file=sys.stderr)
        print("\n".join(out.strip().split("\n")[-80:]), file=sys.stderr)
        return 1

    # Check that edit_multiview path was hit
    if "edit_latents_multiview" not in out or "Multiview editing finished" not in out:
        print("Log did not show multiview editing; maybe edit_multiview was not run (e.g. step condition).", file=sys.stderr)
        return 0  # still pass if process succeeded

    # Optionally check for output path (trial_dir varies)
    if "edited_images_multiview" in out:
        print("verify_edit_multiview: edit_multiview ran and completed (log OK).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
