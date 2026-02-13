#!/usr/bin/env python3
"""
Generate cameras using the "generate by lens" strategy and render 3DGS views.

This script is intentionally separate from `render_orbit_from_colmap.py`:
  - `render_orbit_from_colmap.py` keeps the original orbit-only behavior.
  - This script implements the lens-based camera generation described by:
      - ROI intrinsic analysis (weighted PCA on 3D Gaussians)
      - Fibonacci manifold sampling on a sphere
      - Diversity-aware selection (FPS in angular space)

Currently this is a geometry-only implementation:
  - It does NOT yet query IP2P attention maps / editability scores.
  - It does NOT yet compute visibility scores from depth maps.
Those can be added later on top of the candidate cameras produced here.
"""

import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.cameras import Simple_Camera
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussiansplatting.utils.general_utils import safe_state

from render_orbit_from_colmap import (
    _parse_center,
    _normalize,
    look_at_c2w,
    fovy_to_fovx,
    load_colmap_prior,
    load_gaussians,
    save_orbit_cameras_to_colmap,
)


def generate_cameras_by_lens(
    gaussians: GaussianModel,
    cam_centers_np: np.ndarray,
    fovy: float,
    render_width: int,
    render_height: int,
    n_candidates: int = 200,
    n_final: int = 20,
    world_up: np.ndarray | None = None,
) -> List[Simple_Camera]:
    """
    Generate a diverse set of cameras using the "generate by lens" principle.

    Geometry-only implementation of the conceptual steps:
      - Step 1: ROI intrinsic analysis via weighted PCA on 3D Gaussians
      - Step 3: Fibonacci sphere sampling on a shell with fixed radius
      - Step 5: Diversity-aware selection using farthest point sampling (FPS)

    Notes:
      - Steps that depend on IP2P attention (editability score) and 3DGS depth
        based visibility are intentionally *not* implemented here.
      - You can later plug in your own scoring function over the returned
        candidate cameras and re-run the FPS selection outside this function.
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        world_up = np.asarray(world_up, dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 1: ROI intrinsic analysis (simplified)
    # ------------------------------------------------------------------
    # We treat *all* Gaussians as ROI Gaussians. If you have a ROI mask,
    # you can pre-filter here before computing PCA.
    xyz = gaussians.get_xyz.detach().cpu().numpy()  # (N, 3)
    if hasattr(gaussians, "get_opacity"):
        alpha = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    else:
        alpha = np.ones(xyz.shape[0], dtype=np.float32)

    alpha = np.clip(alpha, 1e-8, None)
    w = alpha / alpha.sum()

    c_roi = (w[:, None] * xyz).sum(axis=0).astype(np.float32)  # center of ROI
    X = xyz - c_roi[None, :]
    # Weighted covariance: C = sum_i w_i (x_i - c)(x_i - c)^T
    C = (w[:, None] * X).T @ X  # (3, 3)

    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    order = np.argsort(eigvals)[::-1]  # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]  # columns are v1, v2, v3

    v1 = eigvecs[:, 0]  # longest axis
    v2 = eigvecs[:, 1]
    v3 = eigvecs[:, 2]

    # Estimate "front" direction: align eigenvector with mean camera viewing dir
    # Mean camera direction: from cameras towards ROI center.
    cam_dirs = c_roi[None, :] - cam_centers_np  # (N, 3)
    mean_cam_dir = _normalize(cam_dirs.mean(axis=0))

    # Choose eigenvector most aligned with mean camera direction as v_front
    candidates = [v1, v2, v3]
    dots = [abs(float(np.dot(mean_cam_dir, _normalize(v)))) for v in candidates]
    front_idx = int(np.argmax(dots))
    v_front = _normalize(candidates[front_idx])

    # Make a right-handed basis (v_side, v_up_local, v_front)
    v_side = np.cross(world_up, v_front)
    if np.linalg.norm(v_side) < 1e-6:
        # If front ~ world_up, pick a fallback up
        world_up_fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        v_side = np.cross(world_up_fallback, v_front)
    v_side = _normalize(v_side)
    v_up_local = np.cross(v_front, v_side)
    v_up_local = _normalize(v_up_local)

    # Approximate object "size" from PCA eigenvalues
    # (sqrt of largest eigenvalue gives a scale; multiply for safety margin)
    object_size = float(2.0 * np.sqrt(max(eigvals[0], 1e-8)))

    # ------------------------------------------------------------------
    # Step 2: Scale probing (simplified, no IP2P)
    # ------------------------------------------------------------------
    # In the full method, different distances d are probed with IP2P to find
    # the best editability distance d*. Here we simply set:
    #   d* = k * object_size
    k = 1.5
    d_star = k * object_size

    # ------------------------------------------------------------------
    # Step 3: Fibonacci sphere sampling (candidate generation)
    # ------------------------------------------------------------------
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    N = max(int(n_candidates), 1)

    dirs_world: List[np.ndarray] = []
    for i in range(N):
        # Standard Fibonacci sphere in local coordinates
        t = float(i) / float(max(N - 1, 1))
        z = 1.0 - 2.0 * t  # from +1 to -1
        radius_xy = np.sqrt(max(1.0 - z * z, 0.0))
        theta = golden_angle * i
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        # Local dir (x,y,z) where +z is "front"
        dir_local = np.array([x, y, z], dtype=np.float32)
        dir_local = _normalize(dir_local)

        # Map to world using basis [v_side, v_up_local, v_front]
        dir_world = (
            dir_local[0] * v_side
            + dir_local[1] * v_up_local
            + dir_local[2] * v_front
        )
        dir_world = _normalize(dir_world.astype(np.float32))
        dirs_world.append(dir_world)

    dirs_world = np.stack(dirs_world, axis=0)  # (N, 3)

    # Candidate camera centers and canonical scores
    candidate_centers = c_roi[None, :] + d_star * dirs_world  # (N, 3)

    # Canonical score: favour views aligned with front or side axes
    v_front_norm = _normalize(v_front)
    v_side_norm = _normalize(v_side)
    # Camera "viewing" direction is from camera towards object center
    view_dirs = _normalize(c_roi[None, :] - candidate_centers)  # (N, 3)

    front_alignment = np.abs(np.einsum("ij,j->i", view_dirs, v_front_norm))
    side_alignment = np.abs(np.einsum("ij,j->i", view_dirs, v_side_norm))
    S_can = np.maximum(front_alignment, side_alignment)

    # ------------------------------------------------------------------
    # Step 4: (optional) visibility / editability score
    # ------------------------------------------------------------------
    # Here we keep it geometry-only and set:
    #   E = S_can
    # In a full implementation, add S_vis and / or S_edit here.
    E = S_can.copy()

    # ------------------------------------------------------------------
    # Step 5: Diversity-aware final selection (FPS in angle space)
    # ------------------------------------------------------------------
    # 1) Filter top-20% by energy
    M = max(int(0.2 * N), 1)
    top_indices = np.argsort(E)[-M:]  # largest M
    top_dirs = view_dirs[top_indices]  # (M, 3)
    top_centers = candidate_centers[top_indices]

    # 2) Farthest point sampling on the sphere using angular distance
    K = min(int(n_final), M)
    selected_local: List[int] = []

    # Seed with best energy among the filtered set
    first_local = int(np.argmax(E[top_indices]))
    selected_local.append(first_local)

    # Precompute cosine-similarity matrix for efficiency
    sim_mat = np.clip(top_dirs @ top_dirs.T, -1.0, 1.0)  # (M, M)

    # Track for each candidate its maximum similarity to the selected set
    max_sim = sim_mat[first_local].copy()

    for _ in range(1, K):
        # Angular distance ~ arccos(sim), so "farthest" => smallest max_sim
        mask = np.ones(M, dtype=bool)
        mask[selected_local] = False

        remaining_indices = np.where(mask)[0]
        if remaining_indices.size == 0:
            break
        next_local = int(
            remaining_indices[np.argmin(max_sim[remaining_indices])]
        )
        selected_local.append(next_local)

        # Update max_sim with similarities to the newly selected point
        max_sim = np.maximum(max_sim, sim_mat[next_local])

    selected_local = selected_local[:K]
    final_centers = top_centers[selected_local]

    # Build Simple_Camera objects with look-at C2W using 3DGS / COLMAP convention
    h = int(render_height)
    w = int(render_width)
    fovx = fovy_to_fovx(fovy, h=h, w=w)

    cameras: List[Simple_Camera] = []
    for i, P in enumerate(final_centers):
        # P is camera center, look at c_roi
        c2w = look_at_c2w(P.astype(np.float32), c_roi.astype(np.float32), world_up)
        R_c2w = c2w[:3, :3].astype(np.float32)
        cam_center = c2w[:3, 3].astype(np.float32)
        # Simple_Camera expects R such that W2C = [R^T, T].
        R = R_c2w
        T = (-(R_c2w.T) @ cam_center).astype(np.float32)

        cam = Simple_Camera(
            colmap_id=i,
            R=R,
            T=T,
            FoVx=float(fovx),
            FoVy=float(fovy),
            h=h,
            w=w,
            image_name=f"lens_{i:05d}",
            uid=i,
            data_device="cuda",
            qvec=None,
        )
        cameras.append(cam)

    return cameras


def main():
    parser = ArgumentParser(
        description="Generate lens-based cameras using COLMAP + 3DGS and render them."
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument(
        "--colmap_path",
        type=str,
        required=True,
        help="Dataset root containing sparse/0",
    )
    parser.add_argument(
        "--center",
        type=str,
        required=True,
        help="Object center as 'x,y,z' in COLMAP world coords",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Which iteration to render (-1 = latest)",
    )

    parser.add_argument(
        "--n_candidates",
        type=int,
        default=200,
        help="Number of Fibonacci-sphere candidates to sample",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=20,
        help="Number of final views to select (FPS on candidates)",
    )

    parser.add_argument("--render_width", type=int, default=512)
    parser.add_argument("--render_height", type=int, default=512)

    parser.add_argument("--out_dir", type=str, default="output/lens")
    parser.add_argument("--video_path", type=str, default="output/lens.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--save_colmap",
        type=str,
        default=None,
        help=(
            "Save generated cameras to Colmap format at this directory "
            "(creates sparse/0 subdirectory). "
            "Use 'txt' or 'bin' for format, or omit for txt format."
        ),
    )
    parser.add_argument(
        "--colmap_format",
        type=str,
        default="txt",
        choices=["txt", "bin"],
        help="Colmap file format: 'txt' or 'bin' (default: txt)",
    )

    args = get_combined_args(parser)
    safe_state(args.quiet)

    # Load 3D Gaussians
    gaussians, used_iter = load_gaussians(
        args.model_path, args.iteration, model.extract(args).sh_degree
    )
    gaussian_center = (
        gaussians.get_xyz.detach().mean(dim=0).float().cpu().numpy()
    )

    # Parse user-provided center
    C_user = _parse_center(args.center)

    # Load COLMAP camera centers and intrinsics
    cam_centers_np, fovy, colmap_cameras = load_colmap_prior(
        args.colmap_path
    )  # (N,3), rad, dict

    # Diagnostics on center choice (same logic as orbit script)
    d_user = np.linalg.norm(cam_centers_np - C_user[None, :], axis=1)
    d_g = np.linalg.norm(cam_centers_np - gaussian_center[None, :], axis=1)
    med_user = float(np.median(d_user))
    med_g = float(np.median(d_g))
    delta_norm = float(np.linalg.norm(C_user - gaussian_center))
    print(
        f"Center diagnostics: |user-gaussian|={delta_norm:.4f} | "
        f"median_dist_to_user={med_user:.4f} median_dist_to_gaussian={med_g:.4f}"
    )

    # For now, just use Gaussian center as ROI center in PCA;
    # generate_cameras_by_lens internally recomputes a weighted center `c_roi`.
    bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    frames = []
    lens_cameras: List[Simple_Camera] = []

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    print(
        f"Generating lens-based cameras: n_candidates={int(args.n_candidates)}, "
        f"n_views={int(args.n_views)}"
    )
    lens_cameras = generate_cameras_by_lens(
        gaussians,
        cam_centers_np,
        fovy,
        render_width=args.render_width,
        render_height=args.render_height,
        n_candidates=int(args.n_candidates),
        n_final=int(args.n_views),
        world_up=world_up,
    )

    print(f"Rendering {len(lens_cameras)} lens-based views ...")
    for i, cam in enumerate(lens_cameras):
        out = render(cam, gaussians, pipeline.extract(args), background)
        rgb = out["render"]

        out_path = os.path.join(args.out_dir, f"{i:05d}.png")
        torchvision.utils.save_image(rgb, out_path)
        frames.append(
            (rgb.clamp(0.0, 1.0) * 255.0)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        if (i + 1) % 10 == 0:
            print(f"Rendered {i+1}/{len(lens_cameras)}")

    if args.video_path:
        import imageio

        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        print(f"Writing video to {args.video_path} (fps={int(args.fps)})")
        imageio.mimsave(args.video_path, frames, fps=int(args.fps))

    # Save generated cameras to Colmap format if requested
    save_colmap_dir = getattr(args, "save_colmap", None)
    if save_colmap_dir:
        source_cam = list(colmap_cameras.values())[0]  # match source cameras.bin
        save_orbit_cameras_to_colmap(
            lens_cameras,
            save_colmap_dir,
            camera_id=1,
            format=getattr(args, "colmap_format", "txt"),
            source_camera=source_cam,
        )
        print(
            f"\nLens-based cameras saved to Colmap format at: "
            f"{save_colmap_dir}/sparse/0/"
        )
        print(
            "  You can use this path as --colmap_path in DGE for rendering these views."
        )


if __name__ == "__main__":
    main()

