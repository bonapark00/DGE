#!/usr/bin/env python3
"""
Run InstructPix2Pix on a single image with a text prompt.
Usage:
  python run_instruct_pix2pix.py --image path/to/image.png --prompt "Make him wear blue earrings"
  python run_instruct_pix2pix.py -i load/lego_bulldozer.jpg -p "Turn it into a snow scene" -o out.png
"""

# Apply huggingface_hub compatibility patch before diffusers
import hf_hub_patch  # noqa: E402

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline


def _dge_like_resize_dims(h: int, w: int) -> tuple[int, int]:
    """
    Match DGE's resizing rule:
    factor = 512 / max(W, H)
    factor = ceil(min(W, H) * factor / 64) * 64 / min(W, H)
    then round both dims down to multiples of 64.
    Returns (new_h, new_w).
    """
    factor = 512.0 / float(max(w, h))
    factor = math.ceil((min(w, h) * factor) / 64.0) * 64.0 / float(min(w, h))
    new_w = int((w * factor) // 64) * 64
    new_h = int((h * factor) // 64) * 64
    # safety: avoid 0-dim if extremely small inputs
    new_w = max(new_w, 64)
    new_h = max(new_h, 64)
    return new_h, new_w


def main():
    parser = argparse.ArgumentParser(description="Edit a single image with InstructPix2Pix")
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Edit instruction (e.g. 'Make him wear blue earrings on his ears')",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save edited image. Default: input_edited.png",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="HuggingFace model id (default: timbrooks/instruct-pix2pix)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for conditioning (default: 7.5)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps (default: 20)",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Image conditioning strength (default: 1.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.output is None:
        out_path = image_path.parent / f"{image_path.stem}_edited{image_path.suffix}"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f"Loading InstructPix2Pix from {args.model} ...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(args.device)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized_h, resized_w = _dge_like_resize_dims(orig_h, orig_w)
    if (resized_w, resized_h) != (orig_w, orig_h):
        print(f"Resizing input to DGE-like size: {(orig_w, orig_h)} -> {(resized_w, resized_h)}")
        image_for_ip2p = image.resize((resized_w, resized_h), resample=Image.BICUBIC)
    else:
        image_for_ip2p = image

    print(f"Editing with prompt: {args.prompt}")
    result = pipe(
        prompt=args.prompt,
        image=image_for_ip2p,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
    )
    out_image = result.images[0]
    # Match DGE behavior: resize edited output back to original resolution
    if out_image.size != (orig_w, orig_h):
        out_image = out_image.resize((orig_w, orig_h), resample=Image.BICUBIC)

    if not out_path.suffix or out_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        out_path = out_path.with_suffix(".png")
    out_image.save(str(out_path))
    print(f"Saved edited image to {out_path}")


if __name__ == "__main__":
    main()
