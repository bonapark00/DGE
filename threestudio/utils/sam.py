import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor

from lang_sam import LangSAM

# from threestudio.utils.typing import *


class LangSAMTextSegmentor(torch.nn.Module):
    def __init__(self, sam_type="sam2.1_hiera_large"):
        super().__init__()
        self.model = LangSAM(sam_type)

        self.to_pil_image = ToPILImage(mode="RGB")
        self.to_tensor = ToTensor()

    def forward(self, images, prompt: str):
        images = rearrange(images, "b h w c -> b c h w")
        masks = []
        for image in images:
            try:
                # breakpoint()
                image = self.to_pil_image(image.clamp(0.0, 1.0))
                # Add explicit error handling for lang_sam predict
                results = self.model.predict([image], [prompt])
                if results is None or len(results) == 0:
                    print(f"Invalid result from lang_sam for prompt '{prompt}'")
                    print(f"Using full mask as fallback")
                    masks.append(torch.ones_like(images[0, 0:1]))
                    continue
                # handle list of dicts as per installed lang_sam
                first = results[0]
                mask = first.get("masks", None)
                # breakpoint()
                if mask is None:
                    print(f"No masks returned, using full mask")
                    masks.append(torch.ones_like(images[0, 0:1]))
                elif getattr(mask, "ndim", 0) == 3:
                    # mask may be numpy array [K, H, W]
                    if isinstance(mask, torch.Tensor):
                        m = mask[0:1].to(torch.float32)
                    else:
                        import numpy as np  # local import to avoid global dep if unnecessary
                        m = torch.from_numpy(mask[0:1]).to(torch.float32)
                    m = m.to(images.device)
                    masks.append(m)
                else:
                    print(f"None {prompt} Detected (ndim={getattr(mask, 'ndim', None)}), using full mask")
                    masks.append(torch.ones_like(images[0, 0:1]))
            except RuntimeError as e:
                error_msg = str(e)
                print(f"RuntimeError in segmentation with prompt '{prompt}': {error_msg}")
                print(f"Using full mask as fallback (entire image will be edited)")
                masks.append(torch.ones_like(images[0, 0:1]))
            except Exception as e:
                print(f"Error in segmentation with prompt '{prompt}': {e}")
                print(f"Using full mask as fallback (entire image will be edited)")
                masks.append(torch.ones_like(images[0, 0:1]))

        return torch.stack(masks, dim=0)


if __name__ == "__main__":
    model = LangSAMTextSegmentor()

    image = Image.open("load/lego_bulldozer.jpg")
    prompt = "a lego bulldozer"

    image = ToTensor()(image)

    image = image.unsqueeze(0)

    mask = model(image, prompt)

    breakpoint()
