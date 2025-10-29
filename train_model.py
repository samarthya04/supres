import hydra
import torch
import os
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from scripts.data_loader import train_val_test_loader
from scripts.exceptions import (
    EvaluateFreshInitializedModelException,
    UnknownModeException,
)
from scripts.model_config import model_selection
from scripts.utilis import model_path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Main function to train or test the model based on the provided configuration.

    This function initializes the model, data loaders, logger, and trainer,
    and performs training, testing, or both based on the specified mode.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing all settings for model training and testing.

    Returns
    -------
    None

    Raises
    ------
    EvaluateFreshInitializedModelException
        If no pre-trained model is specified in the configuration when in "test" mode.
    UnknownModeException
        If an unsupported mode is specified in the configuration.
    """
    torch.set_float32_matmul_precision('medium')
    final_model_path = model_path(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=final_model_path.split("/")[-1],
        config=config_dict,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    model = model_selection(cfg=cfg, device=device)
    train_loader, val_loader, test_loader = train_val_test_loader(cfg=cfg)

    # --- FIX IS HERE ---
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,  # val/LPIPS
        dirpath=cfg.checkpoint.dirpath,
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{val/LPIPS:.3f}}",  # Use val/LPIPS
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode,
        save_last=cfg.checkpoint.save_last  # <-- THIS LINE IS ADDED
    )
    # --- END FIX ---

    model = model.to(device)

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        accelerator=cfg.trainer.accelerator,  # Use GPU
        devices=cfg.trainer.devices,  # Number of GPUs
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        precision=cfg.trainer.precision,  # 16-bit precision
        # strategy=DDPStrategy(find_unused_parameters=True),  # Comment out
    )

    ckpt_path = cfg.trainer.get("resume_from_checkpoint")

    if cfg.mode == "train":
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        
        # --- UPDATED LOGIC ---
        # Load the best checkpoint and save its state_dict
        best_ckpt_path = checkpoint_callback.best_model_path
        
        if not best_ckpt_path:
             print("Warning: No best checkpoint path found. Saving the LAST model.")
             best_ckpt_path = trainer.checkpoint_callback.last_model_path

        if best_ckpt_path:
            print(f"Fit complete. Loading model state_dict from: {best_ckpt_path}")
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt['state_dict'])
            print(f"Saving model state_dict to: {final_model_path}.pth")
            torch.save(model.state_dict(), f"{final_model_path}.pth")
        else:
            print("Warning: No best OR last checkpoint found. Saving the model in memory.")
            torch.save(model.state_dict(), f"{final_model_path}.pth")

    elif cfg.mode == "train-test":
        # 1. Train the model
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

        # 2. Get the path to the best checkpoint
        best_ckpt_path = checkpoint_callback.best_model_path
        
        # --- UPDATED LOGIC ---
        if not best_ckpt_path:
            print("Warning: No best checkpoint path found. Testing/Saving the LAST model.")
            # Fallback to the last model path if it exists
            best_ckpt_path = trainer.checkpoint_callback.last_model_path
        
        if best_ckpt_path:
            # 3. Load the best/last checkpoint's state_dict into your model object
            print(f"Fit complete. Loading model state_dict from: {best_ckpt_path}")
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt['state_dict'])
        else:
            print("Warning: No checkpoint found. Testing/Saving the LAST model in memory.")

        # 4. Now 'model' IS your best/last model
        print("Adjusting model for testing...")
        model = adjust_model_for_testing(cfg, model)
        
        # 5. Test the loaded model
        print("Running test on the loaded model...")
        trainer.test(model, test_loader) 

        # 6. Save the loaded model's state_dict
        print(f"Saving loaded model's state_dict to: {final_model_path}.pth")
        torch.save(model.state_dict(), f"{final_model_path}.pth")
        # --- END UPDATED LOGIC ---

    elif cfg.mode == "test":
        if cfg.model.load_model is None:
            raise EvaluateFreshInitializedModelException()

        # --- UPDATED LOGIC ---
        # You must load the weights into the model *before* testing
        print(f"Loading model for testing from: {cfg.model.load_model}")
        ckpt = torch.load(cfg.model.load_model, map_location=device)
        
        # Check if checkpoint is from Lightning (has 'state_dict') or raw weights
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        print("Adjusting model for testing...")
        model = adjust_model_for_testing(cfg, model)
        
        print("Running test...")
        trainer.test(model, test_loader)
        # --- END UPDATED LOGIC ---

    else:
        raise UnknownModeException()


def adjust_model_for_testing(cfg, model) -> object:
    """Adjust the model configuration for testing.

    This function sets the validation timesteps and posterior type
    for the diffusion process based on the provided configuration.

    Parameters
    ----------
    cfg : object
        The configuration object containing the validation settings.
    model : object
        The model object to be adjusted.

    Returns
    -------
    object
        The adjusted model object.
    """
    # List of supported SupResDiffGAN models that may require adjustments
    models_with_diffusion = {
        "SupResDiffGAN",
        "SupResDiffGAN_without_adv", 
        "SupResDiffGAN_simple_gan",
    }

    # Check if the model requires adjustments of diffusion parameters
    if cfg.model.name in models_with_diffusion:
        if cfg.diffusion.validation_timesteps is not None:
            model.diffusion.set_timesteps(cfg.diffusion.validation_timesteps)

        if cfg.diffusion.validation_posterior_type is not None:
            model.diffusion.set_posterior_type(cfg.diffusion.validation_posterior_type)

    return model


if __name__ == "__main__":
    main()