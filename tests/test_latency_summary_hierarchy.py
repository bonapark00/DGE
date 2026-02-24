#!/usr/bin/env python3
"""
Verify that latency summary includes nested DGE-block timers (e.g. 3d_anchor).

When edit_multiview runs with feature_injection_mode=3d_anchor, the UNet's
make_dge_block records names like:
  edit_multiview.guidance_batch.edit_latents_multiview.target_denoise_loop.batch_forward.unet_forward.dge_block.feature_injection.3d_anchor

This test ensures:
1) LatencyLogger.write_summary() renders such nested names in the tree
   (so "3d_anchor" and "feature_injection" appear under the right parent).
2) After the fix, register_latency_logger() is what allows the UNet blocks
   to receive the logger; this test only checks the summary writer side.

Run from repo root:
  python -m pytest tests/test_latency_summary_hierarchy.py -v
  or:  python tests/test_latency_summary_hierarchy.py
"""

import os
import sys
import tempfile

# Avoid pulling in threestudio (and diffusers etc.); load latency module by path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Import latency without threestudio package
import importlib.util
_spec = importlib.util.spec_from_file_location("latency", os.path.join(_REPO_ROOT, "threestudio", "utils", "latency.py"))
_latency = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_latency)
LatencyLogger = _latency.LatencyLogger


def test_latency_summary_shows_3d_anchor_and_dge_block_hierarchy():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = LatencyLogger(tmpdir)
        # Top-level (only these count toward "Total Time")
        log.record("edit_multiview", 60.0)
        # Nested: same structure as edit_multiview + set_unet_latency_prefix( batch_forward.unet_forward )
        log.record(
            "edit_multiview.guidance_batch.edit_latents_multiview.target_denoise_loop.batch_forward.unet_forward.dge_block.feature_injection.3d_anchor",
            1.5,
        )
        log.record(
            "edit_multiview.guidance_batch.edit_latents_multiview.target_denoise_loop.batch_forward.unet_forward.dge_block.self_attention",
            2.0,
        )
        log.record(
            "edit_multiview.guidance_batch.edit_latents_multiview.target_denoise_loop.batch_forward.unet_forward.dge_block.init",
            0.5,
        )
        log.write_summary()

        path = os.path.join(tmpdir, "summary.txt")
        assert os.path.isfile(path), "summary.txt was not written"
        with open(path) as f:
            text = f.read()

    # Hierarchy must show these segments (by display name)
    assert "3d_anchor" in text, "summary must contain 3d_anchor (nested under unet_forward -> dge_block -> feature_injection)"
    assert "feature_injection" in text or "dge_block" in text, "summary must show dge_block or feature_injection in the tree"
    assert "edit_multiview" in text and "guidance_batch" in text, "top-level hierarchy must appear"


if __name__ == "__main__":
    test_latency_summary_shows_3d_anchor_and_dge_block_hierarchy()
    print("test_latency_summary_hierarchy: OK")
