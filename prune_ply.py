#!/usr/bin/env python3
"""
Standalone script: load a Gaussian Splatting .ply, prune by x/y/z percent, save pruned .ply.

Usage:
  python prune_ply.py <ply_path> --output <out.ply> [--z 0.01] [--y 0.4] [--x 3]

  --z  prune bottom N% by z (default 0)
  --y  prune top    N% by y (default 0)
  --x  prune top and bottom N% each by x (default 0)
"""

import argparse
import os
import torch
from torch import nn

from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel


def _prune_gaussians_by_mask(gaussians: GaussianModel, keep_mask: torch.Tensor) -> int:
    """Prune in-place; keep only points where keep_mask is True. Returns number removed."""
    device = gaussians.get_xyz.device
    keep_mask = keep_mask.to(device)
    n_before = keep_mask.shape[0]
    n_remove = n_before - int(keep_mask.sum().item())
    if n_remove == 0:
        return 0
    gaussians._xyz = nn.Parameter(gaussians._xyz[keep_mask].detach().clone().requires_grad_(True))
    gaussians._features_dc = nn.Parameter(gaussians._features_dc[keep_mask].detach().clone().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(gaussians._features_rest[keep_mask].detach().clone().requires_grad_(True))
    gaussians._opacity = nn.Parameter(gaussians._opacity[keep_mask].detach().clone().requires_grad_(True))
    gaussians._scaling = nn.Parameter(gaussians._scaling[keep_mask].detach().clone().requires_grad_(True))
    gaussians._rotation = nn.Parameter(gaussians._rotation[keep_mask].detach().clone().requires_grad_(True))
    if gaussians.max_radii2D.shape[0] == n_before:
        gaussians.max_radii2D = gaussians.max_radii2D[keep_mask].detach().clone()
    if gaussians.xyz_gradient_accum.shape[0] == n_before:
        gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[keep_mask].detach().clone()
    if gaussians.denom.shape[0] == n_before:
        gaussians.denom = gaussians.denom[keep_mask].detach().clone()
    return n_remove


def main():
    parser = argparse.ArgumentParser(
        description="Prune Gaussian .ply by z (bottom %), y (top %), x (top & bottom %), then save."
    )
    parser.add_argument("ply_path", type=str, help="Input .ply path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output pruned .ply path")
    parser.add_argument(
        "--z", "--prune-z-bottom-percent",
        dest="prune_z", type=float, default=0.0,
        help="Prune bottom N%% by z (default: 0)",
    )
    parser.add_argument(
        "--y", "--prune-y-top-percent",
        dest="prune_y", type=float, default=0.0,
        help="Prune top N%% by y (default: 0)",
    )
    parser.add_argument(
        "--x", "--prune-x-both-percent",
        dest="prune_x", type=float, default=0.0,
        help="Prune top and bottom N%% each by x (default: 0)",
    )
    parser.add_argument("--sh-degree", type=int, default=3, help="Spherical harmonics degree (default: 3)")
    args = parser.parse_args()

    ply_path = args.ply_path
    out_path = args.output
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Input PLY not found: {ply_path}")

    gaussians = GaussianModel(args.sh_degree)
    print(f"[Prune] Loading {ply_path}")
    gaussians.load_ply(ply_path)
    n_before = gaussians.get_xyz.shape[0]

    prune_z, prune_y, prune_x = args.prune_z, args.prune_y, args.prune_x
    if prune_z <= 0 and prune_y <= 0 and prune_x <= 0:
        print("[Prune] No pruning (all --z/--y/--x are 0). Saving copy as-is.")
    else:
        xyz = gaussians.get_xyz.detach()
        n_pts = xyz.shape[0]
        device = xyz.device
        keep_mask = torch.ones(n_pts, dtype=torch.bool, device=device)
        if prune_z > 0:
            z = xyz[:, 2]
            k_z = max(0, int(round(n_pts * (prune_z / 100.0))))
            if k_z > 0:
                _, idx_smallest_z = torch.topk(z, k_z, largest=False)
                keep_mask[idx_smallest_z] = False
        if prune_y > 0:
            y = xyz[:, 1]
            k_y = max(0, int(round(n_pts * (prune_y / 100.0))))
            if k_y > 0:
                _, idx_largest_y = torch.topk(y, k_y, largest=True)
                keep_mask[idx_largest_y] = False
        if prune_x > 0:
            x = xyz[:, 0]
            k_x = max(0, int(round(n_pts * (prune_x / 100.0))))
            if k_x > 0:
                _, idx_smallest_x = torch.topk(x, k_x, largest=False)
                _, idx_largest_x = torch.topk(x, k_x, largest=True)
                keep_mask[idx_smallest_x] = False
                keep_mask[idx_largest_x] = False
        n_remove = (~keep_mask).sum().item()
        if n_remove > 0:
            n_removed = _prune_gaussians_by_mask(gaussians, keep_mask)
            msg = []
            if prune_z > 0:
                msg.append(f"z bottom {prune_z}%")
            if prune_y > 0:
                msg.append(f"y top {prune_y}%")
            if prune_x > 0:
                msg.append(f"x top & bottom {prune_x}% each")
            print(f"[Prune] Removed {n_removed} Gaussians ({', '.join(msg)}). Remaining: {gaussians.get_xyz.shape[0]}")
        else:
            print("[Prune] No points to remove (percent rounds to 0).")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    gaussians.save_ply(out_path)
    n_after = gaussians.get_xyz.shape[0]
    print(f"[Prune] Saved pruned PLY to {out_path} ({n_before} -> {n_after} points)")


if __name__ == "__main__":
    main()
