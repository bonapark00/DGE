"""
Compatibility patch for huggingface_hub to fix diffusers import issues.
This patches the missing cached_download function with hf_hub_download.
"""

import huggingface_hub

# Create compatibility alias for cached_download
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

print("Applied huggingface_hub compatibility patch")
