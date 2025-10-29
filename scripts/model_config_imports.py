# In scripts/model_config_imports.py

"""File containing imports for SupResDiffGAN models only."""

import os
import sys

# Add the parent directory to the path to find the SupResDiffGAN package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
# from diffusers import AutoencoderKL, AutoencoderTiny # Remove Tiny import
from diffusers import AutoencoderKL # Import only KL

# SupResDiffGAN imports only
try:
    from SupResDiffGAN.modules.Diffusion import Diffusion as Diffusion_supresdiffgan
    from SupResDiffGAN.modules.Discriminator import Discriminator as Discriminator_supresdiffgan
    from SupResDiffGAN.modules.FeatureExtractor import FeatureExtractor as FeatureExtractor_supresdiffgan
    from SupResDiffGAN.modules.UNet import UNet as UNet_supresdiffgan
    from SupResDiffGAN.modules.VggLoss import VGGLoss as VGGLoss_supresdiffgan
    from SupResDiffGAN.SupResDiffGAN import SupResDiffGAN
    from SupResDiffGAN.SupResDiffGAN_simple_gan import SupResDiffGAN_simple_gan
    from SupResDiffGAN.SupResDiffGAN_without_adv import SupResDiffGAN_without_adv
    print("Successfully imported SupResDiffGAN components.")
except ImportError as e:
    print(f"Error importing SupResDiffGAN components: {e}")
    # Consider adding fallbacks or raising the error if these are critical
    # Example: raise ImportError("Could not import core SupResDiffGAN modules.") from e