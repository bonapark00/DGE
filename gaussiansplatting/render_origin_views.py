#!/usr/bin/env python3
"""
Render origin (unedited) views from the original PLY using the same cameras as the test renders.
Used to generate GT for metrics when comparing edited vs. original renders.

Usage:
  python gaussiansplatting/render_origin_views.py \\
    --gs_source /path/to/point_cloud.ply \\
    --cameras_path /path/to/save/cameras_for_origin.pt \\
    --out_dir /data/users/jaeyeonpark/DGE-outputs/origin_render/{DATA_TYPE}/{DATA_NAME}/{STRATEGY}

If out_dir already exists and has the expected PNG files, skips rendering.
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

# Ensure project root is in path when run as gaussiansplatting/render_origin_views.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torchvision

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel
from gaussiansplatting.scene.cameras import Simple_Camera
from gaussiansplatting.arguments import PipelineParams


def load_gaussians(ply_path: str, sh_degree: int = 3) -> GaussianModel:
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


def main():
    parser = ArgumentParser(description="Render origin views from original PLY for metrics GT")
    parser.add_argument("--gs_source", type=str, required=True, help="Path to original point_cloud.ply")
    parser.add_argument("--cameras_path", type=str, required=True, help="Path to cameras_for_origin.pt (saved during test)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (e.g. origin_render/DATA_TYPE/DATA_NAME/STRATEGY)")
    parser.add_argument("--force", action="store_true", help="Re-render even if out_dir has existing files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if we can skip (existing files)
    if not args.force:
        existing = list(out_dir.glob("*.png"))
        if existing:
            print(f"[render_origin_views] Found {len(existing)} existing PNGs in {out_dir}, skipping.")
            return 0

    # Load cameras
    cameras_path = Path(args.cameras_path)
    if not cameras_path.exists():
        print(f"[render_origin_views] ERROR: cameras_for_origin.pt not found: {cameras_path}")
        print("  Run training+test first; cameras are saved during test.")
        return 1

    cam_data = torch.load(cameras_path, map_location="cpu")
    if isinstance(cam_data, dict):
        indices = cam_data.get("indices", list(range(len(cam_data.get("cameras", [])))))
        cameras_list = cam_data["cameras"]
    else:
        cameras_list = cam_data
        indices = list(range(len(cameras_list)))

    if len(cameras_list) == 0:
        print("[render_origin_views] ERROR: No cameras in file.")
        return 1

    # Load original PLY (load_ply already puts tensors on CUDA)
    print(f"[render_origin_views] Loading PLY from {args.gs_source}")
    gaussians = load_gaussians(args.gs_source)

    # Pipeline params
    pp_parser = ArgumentParser()
    pp = PipelineParams(pp_parser)
    pp_args = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    pipe = pp.extract(pp_args)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"[render_origin_views] Rendering {len(cameras_list)} views to {out_dir}")
    for i, cam_dict in enumerate(cameras_list):
        R = cam_dict["R"]
        T = cam_dict["T"]
        FoVx = cam_dict["FoVx"]
        FoVy = cam_dict["FoVy"]
        h = int(cam_dict["h"])
        w = int(cam_dict["w"])

        if isinstance(R, torch.Tensor):
            R = R.cpu().numpy()
        if isinstance(T, torch.Tensor):
            T = T.cpu().numpy()

        cam = Simple_Camera(
            colmap_id=i,
            R=R.astype("float32"),
            T=T.astype("float32"),
            FoVx=float(FoVx),
            FoVy=float(FoVy),
            h=h,
            w=w,
            image_name=f"origin_{i:05d}",
            uid=i,
            data_device="cuda",
            qvec=None,
        )
        cam = cam.cuda()

        with torch.no_grad():
            out = render(cam, gaussians, pipe, background)
        rgb = out["render"]

        idx = indices[i] if i < len(indices) else i
        out_path = out_dir / f"{idx:05d}.png"
        torchvision.utils.save_image(rgb, out_path)

        if (i + 1) % 5 == 0:
            print(f"  Rendered {i+1}/{len(cameras_list)}")

    print(f"[render_origin_views] Done. Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
