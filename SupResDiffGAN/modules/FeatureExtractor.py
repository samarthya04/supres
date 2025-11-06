import torch
import torch.nn as nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device) -> None:
        """Initialize the FeatureExtractor.

        Uses pretrained VGG19 on ImageNet to extract multi-level features.
        DDP-safe version: moves buffers dynamically to the input device.
        """
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights="IMAGENET1K_V1").features

        # Store the feature layers for perceptual loss
        self.layers = nn.ModuleDict({
            "conv1": nn.Sequential(*list(vgg19_model.children())[:2]),
            "conv2": nn.Sequential(*list(vgg19_model.children())[2:7]),
            "conv3": nn.Sequential(*list(vgg19_model.children())[7:12]),
            "conv4": nn.Sequential(*list(vgg19_model.children())[12:21]),
            "conv5": nn.Sequential(*list(vgg19_model.children())[21:35]),
        })

        # Register buffers instead of regular tensors (DDP-safe)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.criterion = nn.L1Loss()
        self.feature_weights = {
            "conv1": 0.1,
            "conv2": 0.1,
            "conv3": 1.0,
            "conv4": 1.0,
            "conv5": 1.0,
        }

        # Move everything to the correct initial device
        self.to(device)

    def forward(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        """Compute the perceptual (VGG-based) loss between SR and HR images."""
        device = sr_img.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        # Normalize from [-1, 1] → [0, 1]
        sr_img = (sr_img + 1) / 2
        hr_img = (hr_img + 1) / 2

        # Standardize with mean/std
        sr_img = (sr_img - self.mean) / self.std
        hr_img = (hr_img - self.mean) / self.std

        perceptual_loss = 0.0

        # Extract multi-level features and compute weighted L1 differences
        with torch.no_grad():
            for name, layer in self.layers.items():
                sr_img = layer(sr_img)
                hr_img = layer(hr_img)
                perceptual_loss += self.feature_weights[name] * self.criterion(sr_img, hr_img)

        return perceptual_loss
