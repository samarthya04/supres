# ======================================================================
# SupResDiffGAN Imports (Fixed for DDP and AMP stability)
# ======================================================================
from SupResDiffGAN.modules.Diffusion import Diffusion as Diffusion_supresdiffgan
from SupResDiffGAN.modules.Discriminator import (
    Discriminator as Discriminator_supresdiffgan,
)
from SupResDiffGAN.modules.FeatureExtractor import (
    FeatureExtractor as FeatureExtractor_supresdiffgan,  # ✅ DDP-safe version
)
from SupResDiffGAN.modules.UNet import UNet as UNet_supresdiffgan
from SupResDiffGAN.modules.VggLoss import VGGLoss as VGGLoss_supresdiffgan

# ⚙️ AMP-safe & DDP-compatible SupResDiffGAN versions
from SupResDiffGAN.SupResDiffGAN_amp import SupResDiffGAN as SupResDiffGAN  # ✅ Recommended
from SupResDiffGAN.SupResDiffGAN_simple_gan import SupResDiffGAN_simple_gan
from SupResDiffGAN.SupResDiffGAN_without_adv import SupResDiffGAN_without_adv
