# In scripts/model_config.py

from .model_config_imports import *
# Import AutoencoderKL instead of AutoencoderTiny
from diffusers import AutoencoderKL
import torch # Ensure torch is imported
import os # Ensure os is imported


def model_selection(cfg, device):
    """Select and initialize SupResDiffGAN model variants based on the configuration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    device : torch.device
        The device on which the model will be loaded (e.g., 'cuda' or 'cpu').

    Returns
    -------
    torch.nn.Module
        The initialized SupResDiffGAN model.

    Raises
    ------
    ValueError
        If the specified model is not a supported SupResDiffGAN variant.
    """

    if cfg.model.name == "SupResDiffGAN":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN, use_discriminator=True
        )

    elif cfg.model.name == "SupResDiffGAN_without_adv":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_without_adv, use_discriminator=False
        )

    elif cfg.model.name == "SupResDiffGAN_simple_gan":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_simple_gan, use_discriminator=True
        )

    else:
        raise ValueError(
            f"Model '{cfg.model.name}' not found. "
            f"Supported models: SupResDiffGAN, SupResDiffGAN_without_adv, SupResDiffGAN_simple_gan"
        )


def initialize_supresdiffgan(cfg, device, model_class, use_discriminator=True):
    """Helper function to initialize SupResDiffGAN variants.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    device : torch.device
        The device on which the model will be loaded (e.g., 'cuda' or 'cpu').
    model_class : class
        The class of the model to initialize (e.g., SupResDiffGAN, SupResDiffGAN_without_adv).
    use_discriminator : bool, optional
        Whether to include the discriminator in the model initialization.

    Returns
    -------
    torch.nn.Module
        The initialized model.
    """
    if cfg.autoencoder == "VAE":
        # Recommendation 3: Use AutoencoderKL
        model_id = "stabilityai/stable-diffusion-2-1" # Model ID for AutoencoderKL
        autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        # AutoencoderKL output has 4 channels. UNet in_channels needs to be 8 (4+4).
        # Discriminator in_channels needs to be 6 (pixel space) or 8 (latent space).
        print(f"Loaded AutoencoderKL from {model_id}")

    else:
        raise ValueError(f"Unsupported autoencoder type: {cfg.autoencoder}")


    discriminator = None # Initialize as None
    if use_discriminator:
        discriminator = Discriminator_supresdiffgan(
             # Config value (6) assumes pixel-space D. If latent-space, should be 8.
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels, # Now includes 512
        )
        print("Initialized Discriminator.")


    # Ensure UNet is correctly sized based on config
    unet = UNet_supresdiffgan(cfg.unet) # Should now get [64, 96, 128, 512]
    print(f"Initialized UNet with channels: {cfg.unet}")


    diffusion = Diffusion_supresdiffgan(
        timesteps=cfg.diffusion.timesteps, # Should now get 1000
        beta_type=cfg.diffusion.beta_type,
        posterior_type=cfg.diffusion.posterior_type,
    )
    print(f"Initialized Diffusion with {cfg.diffusion.timesteps} timesteps.")


    # Initialize vgg_loss based on cfg.use_perceptual_loss
    vgg_loss = None # Default to None
    if cfg.get('use_perceptual_loss', False): # Use .get for safety
        if cfg.get('feature_extractor', False): # Use .get for safety
            vgg_loss = FeatureExtractor_supresdiffgan(device)
            print("Initialized FeatureExtractor for perceptual loss.")
        else:
            vgg_loss = VGGLoss_supresdiffgan(device)
            print("Initialized VGGLoss for perceptual loss.")
    else:
        print("Perceptual loss is disabled.")


    # Prepare arguments for model initialization or loading
    # Filter kwargs based on the specific model class constructor signature
    import inspect
    sig = inspect.signature(model_class.__init__)
    valid_kwargs = sig.parameters.keys()

    model_kwargs = {
        'ae': autoencoder,
        'unet': unet,
        'diffusion': diffusion,
        'learning_rate': cfg.model.lr,
        'alfa_perceptual': cfg.model.alfa_perceptual,
        'vgg_loss': vgg_loss
    }

    # Add discriminator and alfa_adv only if the model expects them and use_discriminator is True
    if use_discriminator:
        if 'discriminator' in valid_kwargs:
             model_kwargs['discriminator'] = discriminator
        if 'alfa_adv' in valid_kwargs:
             model_kwargs['alfa_adv'] = cfg.model.alfa_adv
    # If the model expects alfa_adv even without discriminator (like SupResDiffGAN_without_adv placeholder), add it
    elif 'alfa_adv' in valid_kwargs:
         model_kwargs['alfa_adv'] = 0.0 # Pass 0.0 or the config value if needed


    model_to_load = cfg.model.get('load_model', None) # Use .get for safety
    resume_checkpoint = cfg.trainer.get('resume_from_checkpoint', None) # Check for resume path

    # Determine the path to load from (resume takes precedence if specified)
    load_path = resume_checkpoint if resume_checkpoint else model_to_load

    if load_path is not None:
        print(f"Attempting to load model weights from: {load_path}")
        _, ext = os.path.splitext(load_path)
        if ext == ".pth":
            # Load raw state dict
            print(f"Loading state dict from .pth file: {load_path}")
            model = model_class(**model_kwargs) # Initialize model first
            try:
                state_dict = torch.load(load_path, map_location=device)
                # Adjust keys if necessary (e.g., remove 'model.' prefix)
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False) # Use strict=False initially
                print("Successfully loaded state dict from .pth file.")
            except Exception as e:
                print(f"Warning: Failed to load state dict from {load_path}. Error: {e}. Initializing fresh model.")
                model = model_class(**model_kwargs)


        elif ext == ".ckpt":
            # Load from PyTorch Lightning checkpoint
            print(f"Loading from Lightning checkpoint: {load_path}")
            try:
                # Pass necessary construction args again for PL loading
                model = model_class.load_from_checkpoint(
                    load_path,
                    map_location=device,
                    strict=False, # Use strict=False initially
                    **model_kwargs
                )
                print(f"Successfully loaded model from checkpoint: {load_path}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {load_path}. Error: {e}. Initializing fresh model.")
                # Fallback to initializing a new model if loading fails
                model = model_class(**model_kwargs)
        else:
            raise ValueError(f"Unsupported file extension for loading: {ext}")

    else:
        # Initialize a new model from scratch
        print("Initializing new model from scratch.")
        model = model_class(**model_kwargs)

    return model