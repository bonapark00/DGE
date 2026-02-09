#!/usr/bin/env python3
"""
Quick sanity check:
Render frames using COLMAP registered camera views (images.bin + cameras.bin)
and a trained 3D Gaussian point_cloud.ply, then write PNGs + a video.

Run:
  cd /working/style-transfer/DGE-camera-selection
  python render_colmap_views.py \
    --ply_path /path/to/point_cloud/iteration_30000/point_cloud.ply \
    --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
    --out_dir output/colmap_views \
    --video_path output/colmap_views.mp4
"""

import os
from argparse import ArgumentParser
from typing import List, Tuple

import torch
import torchvision

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.camera_scene import CamScene
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussiansplatting.utils.general_utils import safe_state


def load_gaussians(ply_path: str, sh_degree: int):
    """Load 3D Gaussians from a .ply file path."""
    ply_path = os.path.abspath(ply_path)
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    gaussians = GaussianModel(sh_degree)
    print(f"Loading Gaussian model from {ply_path}")
    gaussians.load_ply(ply_path)
    return gaussians


def load_colmap_views_as_cameras(colmap_path: str, h: int, w: int):
    """
    DGE(threestudio)와 동일한 경로로 COLMAP 카메라를 로드.
    - CamScene -> dataset_readers.readColmapSceneInfo_hw -> camera_utils.cameraList_load
    - 결과: Simple_Camera 리스트 (FoV/pose/transform이 렌더러 기대값과 동일)
    """
    scene = CamScene(os.path.abspath(colmap_path), h=h, w=w)
    return scene.cameras


def main():
    parser = ArgumentParser(description="Render with COLMAP views (sanity check)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--ply_path", type=str, required=True, help="Path to point_cloud.ply file")
    parser.add_argument("--colmap_path", type=str, required=True, help="Dataset root containing sparse/0")
    parser.add_argument("--out_dir", type=str, default="output/colmap_views")
    parser.add_argument("--video_path", type=str, default="output/colmap_views.mp4")
    parser.add_argument("--fps", type=int, default=24)

    parser.add_argument("--render_width", type=int, default=4029)
    parser.add_argument("--render_height", type=int, default=3021)

    parser.add_argument("--stride", type=int, default=1, help="Render every Nth COLMAP view")
    parser.add_argument("--max_views", type=int, default=-1, help="Limit number of rendered views (-1 = all)")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    # When using --ply_path only there is no cfg_args, so ModelParams/PipelineParams fields stay None
    # (sentinel=True). Fill defaults so model.extract() / pipeline.extract() return valid objects.
    defaults = [
        ("source_path", ""),
        ("model_path", ""),
        ("sh_degree", 3),
        ("white_background", False),
    ]
    for name, val in defaults:
        if getattr(args, name, None) is None:
            setattr(args, name, val)

    pipe = pipeline.extract(args)
    m = model.extract(args)

    gaussians = load_gaussians(args.ply_path, m.sh_degree)
    bg_color = [1, 1, 1] if m.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cams = load_colmap_views_as_cameras(
        args.colmap_path, h=int(args.render_height), w=int(args.render_width)
    )
    if args.stride > 1:
        cams = cams[:: int(args.stride)]
    if int(args.max_views) > 0:
        cams = cams[: int(args.max_views)]

    os.makedirs(args.out_dir, exist_ok=True)
    frames = []

    print(f"Rendering {len(cams)} COLMAP views at {args.render_width}x{args.render_height}")
    for i, cam in enumerate(cams):
        out = render(cam, gaussians, pipe, background)
        rgb = out["render"]
        if i == 0:
            vis = out.get("visibility_filter", None)
            if vis is not None:
                vis_ratio = float(vis.float().mean().item())
                print(f"[debug] view0 visibility_ratio={vis_ratio:.6f}")
            depth = out.get("depth_3dgs", None)
            if depth is not None:
                dmin = float(depth.min().item())
                dmax = float(depth.max().item())
                print(f"[debug] view0 depth_3dgs min/max = {dmin:.6f}/{dmax:.6f}")
            mean_rgb = float(rgb.mean().item())
            max_rgb = float(rgb.max().item())
            print(f"[debug] view0 rgb mean/max = {mean_rgb:.6f}/{max_rgb:.6f}")

        out_path = os.path.join(args.out_dir, f"{i:05d}_{getattr(cam, 'image_name', 'view')}.png")
        torchvision.utils.save_image(rgb, out_path)
        frames.append((rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"Rendered {i+1}/{len(cams)}")

    if args.video_path:
        import imageio

        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        print(f"Writing video to {args.video_path} (fps={int(args.fps)})")
        imageio.mimsave(args.video_path, frames, fps=int(args.fps))


if __name__ == "__main__":
    main()