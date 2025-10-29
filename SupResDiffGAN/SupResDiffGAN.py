import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import wandb # Keep if you use wandb logging
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time

# --- Corrected LPIPS Import and Assignment ---
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LPIPSMetric = LearnedPerceptualImagePatchSimilarity
    print("Successfully imported torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity")
except ImportError:
    print("Warning: Could not import LearnedPerceptualImagePatchSimilarity. LPIPS will be disabled.")
    # Define a dummy module that returns 0 if import fails
    class LPIPSDummy(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, img1, img2): return torch.tensor(0.0, device=img1.device)
    LPIPSMetric = LPIPSDummy
# --- End Corrected LPIPS Import ---


# Helper function for re-normalization
def normalize_tensor_01(tensor):
    """Normalize tensor from [-1, 1] to [0, 1]"""
    # Clamp to ensure input is truly in [-1, 1] before normalization
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return (tensor + 1.0) / 2.0

class SupResDiffGAN(pl.LightningModule):
    """SupResDiffGAN class for Super-Resolution Diffusion Generative Adversarial Network.
    Parameters are loaded via Hydra config.
    """
    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 1e-3,
        alfa_adv: float = 1e-2,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['ae', 'discriminator', 'unet', 'diffusion', 'vgg_loss'])

        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion

        self.vgg_loss = vgg_loss
        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Initialize LPIPS, normalize=False as input will be manually normalized to [0, 1]
        try:
             self.lpips = LPIPSMetric(net_type="alex", normalize=False)
             print(f"Initialized LPIPSMetric: {type(self.lpips)}")
        except Exception as e:
             print(f"Error initializing LPIPSMetric: {e}. Using dummy LPIPS.")
             class LPIPSDummy(nn.Module):
                 def __init__(self, *args, **kwargs): super().__init__()
                 def forward(self, img1, img2): return torch.tensor(0.0, device=img1.device)
             self.lpips = LPIPSDummy()


        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        # Freeze Autoencoder
        for param in self.ae.parameters():
            param.requires_grad = False
        self.ae.eval()

        self.automatic_optimization = False

        self.test_step_outputs = []
        self.ema_weight = 0.97
        self.ema_mean = 0.5
        self.s = 0

        # Placeholders for logging
        self.last_g_content_loss = torch.tensor(0.0)
        self.last_g_perceptual_loss = torch.tensor(0.0)
        self.last_g_adv_loss = torch.tensor(0.0)
        self.last_d_acc = 0.5


    def setup(self, stage=None):
         """Called by PyTorch Lightning. Moves components to the correct device."""
         if hasattr(self, 'device'):
             print(f"Moving LPIPS and VGG Loss (if applicable) to device: {self.device}")
             if hasattr(self.lpips, 'to'): self.lpips.to(self.device)
             if self.vgg_loss and hasattr(self.vgg_loss, 'to'):
                 self.vgg_loss.to(self.device)
         else:
             print("Warning: self.device not available during setup.")


    def _get_scaled_latents(self, encoder_output):
        """Extracts and scales latents from AutoencoderKL output."""
        if hasattr(encoder_output, 'latent_dist'):
            latents = encoder_output.latent_dist.sample()
        elif hasattr(encoder_output, 'latents'):
             latents = encoder_output.latents
        else:
             latents = encoder_output
        scaling_factor = getattr(self.ae.config, "scaling_factor", 1.0)
        return latents * scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        if not hasattr(self, 'device'):
             current_device = next(self.parameters()).device
        else:
             current_device = self.device
        x = x.to(current_device)

        with torch.no_grad():
            self.ae.eval()
            encoder_output = self.ae.encode(x)
            x_lat = self._get_scaled_latents(encoder_output).detach()

        x_sampled_lat = self.diffusion.sample(self.generator, x_lat.to(current_device), x_lat.shape)
        x_sampled_lat = x_sampled_lat.to(current_device)

        with torch.no_grad():
            scaling_factor = getattr(self.ae.config, "scaling_factor", 1.0)
            x_out = self.ae.decode(x_sampled_lat / scaling_factor).sample
        x_out = torch.clamp(x_out, -1, 1)
        return x_out


    def training_step(
        self, batch: dict, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        lr_img = batch["lr"].to(self.device)
        hr_img = batch["hr"].to(self.device)
        optimizer_g, optimizer_d = self.optimizers()

        with torch.no_grad():
            self.ae.eval()
            lr_lat = self._get_scaled_latents(self.ae.encode(lr_img)).detach()
            x0_lat = self._get_scaled_latents(self.ae.encode(hr_img)).detach()

        timesteps = torch.randint(
            0, self.diffusion.timesteps, (x0_lat.shape[0],),
            device=self.device, dtype=torch.long,
        )
        x_t = self.diffusion.forward(x0_lat, timesteps).to(self.device)

        alpha_bars_device = self.diffusion.alpha_bars_torch.to(timesteps.device)
        alfa_bars = alpha_bars_device[timesteps]

        x_gen_0_lat = self.generator(lr_lat.to(self.device), x_t.to(self.device), alfa_bars.to(self.device))
        x_gen_0_lat = x_gen_0_lat.to(self.device)

        with torch.no_grad():
            s_tensor = torch.tensor(self.s, device=self.device, dtype=torch.long).expand(
                x0_lat.shape[0]
            )
            x_s_lat = self.diffusion.forward(x0_lat, s_tensor).to(self.device)
            x_gen_s_lat = self.diffusion.forward(x_gen_0_lat.detach(), s_tensor).to(self.device)

            scaling_factor = getattr(self.ae.config, "scaling_factor", 1.0)
            hr_s_img = torch.clamp(self.ae.decode(x_s_lat / scaling_factor).sample, -1, 1)
            sr_s_img = torch.clamp(self.ae.decode(x_gen_s_lat / scaling_factor).sample, -1, 1)


        # Alternate Training
        if batch_idx % 2 == 0: # Train Generator
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            scaling_factor = getattr(self.ae.config, "scaling_factor", 1.0) # Get scaling factor again
            # Re-calculate noisy generated latent *with* gradients
            x_gen_s_lat_for_g = self.diffusion.forward(x_gen_0_lat, s_tensor).to(self.device)
            # Re-decode images *with* gradients for generator loss calculation
            sr_img_for_g = torch.clamp(self.ae.decode(x_gen_0_lat / scaling_factor).sample, -1, 1)
            sr_s_img_for_g = torch.clamp(self.ae.decode(x_gen_s_lat_for_g / scaling_factor).sample, -1, 1)

            g_loss = self.generator_loss(
                x0_lat, x_gen_0_lat, hr_img, sr_img_for_g, hr_s_img, sr_s_img_for_g
            )
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

            # Logging
            self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("train/g_content_loss", self.last_g_content_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            if self.vgg_loss is not None:
                self.log("train/g_perceptual_loss", self.last_g_perceptual_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("train/g_adv_loss", self.last_g_adv_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            return {"loss": g_loss}
        else: # Train Discriminator
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            d_loss = self.discriminator_loss(hr_s_img, sr_s_img) # sr_s_img is already detached
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            # Logging
            self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("train/ema_noise_step", float(self.s), on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("train/ema_accuracy", self.last_d_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            return {"loss": d_loss}


    def validation_step(
        self, batch: dict, batch_idx: int
    ) -> None:
        lr_img = batch["lr"].to(self.device)
        hr_img = batch["hr"].to(self.device)
        padding_info = {"lr": batch["padding_data_lr"], "hr": batch["padding_data_hr"]}

        sr_img = self(lr_img)

        # Visualization
        if batch_idx < 1:
            print(f"Generating validation images for Epoch {self.current_epoch}, Batch {batch_idx}...")
            try:
                title = f"Validation Epoch {self.current_epoch} - Batch {batch_idx}"
                per_image_metrics = []
                num_images_to_plot = min(3, lr_img.shape[0])

                with torch.no_grad():
                    for i in range(num_images_to_plot):
                        hr_img_i_cpu = hr_img[i:i+1].cpu()
                        sr_img_i_cpu = sr_img[i:i+1].cpu()
                        hr_img_np = (hr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
                        sr_img_np = (sr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2

                        hr_h, hr_w = padding_info["hr"][i][1], padding_info["hr"][i][0]
                        hr_img_np_crop = np.clip(hr_img_np[:hr_h, :hr_w, :], 0, 1)
                        sr_img_np_crop = np.clip(sr_img_np[:hr_h, :hr_w, :], 0, 1)

                        psnr = peak_signal_noise_ratio(hr_img_np_crop, sr_img_np_crop, data_range=1.0)
                        ssim_win_size = min(7, hr_img_np_crop.shape[0], hr_img_np_crop.shape[1])
                        if ssim_win_size < 7 : print(f"Warning: SSIM window size reduced to {ssim_win_size}")
                        if ssim_win_size % 2 == 0: ssim_win_size -= 1
                        if ssim_win_size < 3: ssim_win_size = 3
                        ssim = structural_similarity(hr_img_np_crop, sr_img_np_crop, channel_axis=-1, data_range=1.0, win_size=ssim_win_size)

                        lpips_val = self.lpips(
                            normalize_tensor_01(hr_img[i:i+1]),
                            normalize_tensor_01(sr_img[i:i+1])
                        ).item()
                        per_image_metrics.append((psnr, ssim, lpips_val))

                img_array = self.plot_images_with_metrics(
                    hr_img[:num_images_to_plot].cpu(), lr_img[:num_images_to_plot].cpu(), sr_img[:num_images_to_plot].cpu(),
                    {k: v[:num_images_to_plot] for k, v in padding_info.items()},
                    title, per_image_metrics
                )
                if self.logger and self.logger.experiment:
                     self.logger.experiment.log({
                         f"Validation Images Epoch {self.current_epoch}": wandb.Image(img_array, caption=f"Epoch {self.current_epoch} Batch {batch_idx}")
                     })
                else:
                    print("WandB logger not available.")
                print("Successfully processed validation images.")
            except Exception as e:
                print(f"Visualization or Metric error in validation: {str(e)}")

        # Metrics Calculation
        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        batch_lpips = []
        with torch.no_grad():
            for i in range(lr_img.shape[0]):
                hr_img_i_cpu = hr_img[i:i+1].cpu()
                sr_img_i_cpu = sr_img[i:i+1].cpu()
                hr_img_np = (hr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
                sr_img_np = (sr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2

                hr_h, hr_w = padding_info["hr"][i][1], padding_info["hr"][i][0]
                hr_img_np_crop = np.clip(hr_img_np[:hr_h, :hr_w, :], 0, 1)
                sr_img_np_crop = np.clip(sr_img_np[:hr_h, :hr_w, :], 0, 1)

                metrics["PSNR"].append(peak_signal_noise_ratio(hr_img_np_crop, sr_img_np_crop, data_range=1.0))
                ssim_win_size = min(7, hr_img_np_crop.shape[0], hr_img_np_crop.shape[1])
                if ssim_win_size % 2 == 0: ssim_win_size -= 1
                if ssim_win_size < 3: ssim_win_size = 3
                metrics["SSIM"].append(structural_similarity(hr_img_np_crop, sr_img_np_crop, channel_axis=-1, data_range=1.0, win_size=ssim_win_size))
                metrics["MSE"].append(np.mean((hr_img_np_crop - sr_img_np_crop) ** 2))

                lpips_val_single = self.lpips(
                     normalize_tensor_01(hr_img[i:i+1]),
                     normalize_tensor_01(sr_img[i:i+1])
                ).item()
                batch_lpips.append(lpips_val_single)

        self.log_dict({
            "val/PSNR": np.mean(metrics["PSNR"]), "val/SSIM": np.mean(metrics["SSIM"]),
            "val/MSE": np.mean(metrics["MSE"]), "val/LPIPS": np.mean(batch_lpips)
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def plot_images_with_metrics( self, hr_img, lr_img, sr_img, padding_info, title, per_image_metrics):
        num_images = hr_img.shape[0]
        fig, axs = plt.subplots(3, num_images, figsize=(num_images * 3, 9), dpi=100)
        if num_images == 1: axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
        fig.patch.set_facecolor("#1e1e1e")

        for i in range(num_images):
            if i >= len(per_image_metrics): continue
            psnr, ssim, lpips_val = per_image_metrics[i]

            sr_img_plot = (sr_img[i].float().numpy().transpose(1, 2, 0) + 1) / 2
            sr_h, sr_w = padding_info["hr"][i][1], padding_info["hr"][i][0]
            sr_img_plot = np.clip(sr_img_plot[:sr_h, :sr_w, :], 0, 1)

            hr_img_true = (hr_img[i].float().numpy().transpose(1, 2, 0) + 1) / 2
            hr_img_true = np.clip(hr_img_true[:sr_h, :sr_w, :], 0, 1)

            lr_img_true = (lr_img[i].float().numpy().transpose(1, 2, 0) + 1) / 2
            lr_h, lr_w = padding_info["lr"][i][1], padding_info["lr"][i][0]
            lr_img_true = np.clip(lr_img_true[:lr_h, :lr_w, :], 0, 1)

            for row_idx, (img_data, row_title) in enumerate([
                (hr_img_true, "HR Ground Truth"),
                (lr_img_true, "LR Input"),
                (sr_img_plot, f"SR Predicted\nPSNR: {psnr:.2f}\nSSIM: {ssim:.3f}\nLPIPS: {lpips_val:.3f}")
            ]):
                axs[row_idx, i].imshow(img_data)
                title_color = "white" if row_idx < 2 else "limegreen"
                title_fontsize = 9 if row_idx < 2 else 7
                title_pad = 5 if row_idx < 2 else 2
                axs[row_idx, i].set_title(row_title, fontsize=title_fontsize, color=title_color, pad=title_pad)
                axs[row_idx, i].set_xticks([]); axs[row_idx, i].set_yticks([])
                axs[row_idx, i].set_facecolor("#2e2e2e")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


    def test_step(
        self, batch: dict, batch_idx: int
    ) -> dict:
        lr_img = batch["lr"].to(self.device)
        hr_img = batch["hr"].to(self.device)
        padding_info = {"lr": batch["padding_data_lr"], "hr": batch["padding_data_hr"]}

        start_time = time.perf_counter()
        with torch.no_grad():
             sr_img = self(lr_img)
        elapsed_time = time.perf_counter() - start_time

        # Visualization
        if batch_idx == 0:
            per_image_metrics = []
            num_images_to_plot = min(3, lr_img.shape[0])
            with torch.no_grad():
                for i in range(num_images_to_plot):
                    hr_img_i_cpu = hr_img[i:i+1].cpu()
                    sr_img_i_cpu = sr_img[i:i+1].cpu()
                    hr_img_np = (hr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
                    sr_img_np = (sr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
                    hr_h, hr_w = padding_info["hr"][i][1], padding_info["hr"][i][0]
                    hr_img_np_crop = np.clip(hr_img_np[:hr_h, :hr_w, :], 0, 1)
                    sr_img_np_crop = np.clip(sr_img_np[:hr_h, :hr_w, :], 0, 1)

                    psnr = peak_signal_noise_ratio(hr_img_np_crop, sr_img_np_crop, data_range=1.0)
                    ssim_win_size = min(7, hr_img_np_crop.shape[0], hr_img_np_crop.shape[1])
                    if ssim_win_size % 2 == 0: ssim_win_size -=1
                    if ssim_win_size < 3: ssim_win_size = 3
                    ssim = structural_similarity(hr_img_np_crop, sr_img_np_crop, channel_axis=-1, data_range=1.0, win_size=ssim_win_size)
                    lpips_val = self.lpips(
                        normalize_tensor_01(hr_img[i:i+1]),
                        normalize_tensor_01(sr_img[i:i+1])
                    ).item()
                    per_image_metrics.append((psnr, ssim, lpips_val))

            current_timesteps = self.diffusion.timesteps
            # Use validation_posterior_type if testing after training, else use config default
            current_posterior = getattr(self.hparams, 'validation_posterior_type', self.diffusion.posterior_type)

            title = f"Test Images: Timesteps {current_timesteps}, Posterior {current_posterior}"

            img_array = self.plot_images_with_metrics(
                 hr_img[:num_images_to_plot].cpu(), lr_img[:num_images_to_plot].cpu(), sr_img[:num_images_to_plot].cpu(),
                 {k: v[:num_images_to_plot] for k, v in padding_info.items()},
                 title, per_image_metrics
            )
            save_dir = "outputs/test_images"
            os.makedirs(save_dir, exist_ok=True)
            from PIL import Image
            img_pil = Image.fromarray(img_array)
            save_path = os.path.join(save_dir, f"test_results_timesteps_{current_timesteps}_posterior_{current_posterior}.png")
            img_pil.save(save_path)
            print(f"Saved test image to {save_path}")
            try:
                if self.logger and self.logger.experiment:
                     self.logger.experiment.log(
                         {"test_images": wandb.Image(img_pil, caption=f"Test Results {current_timesteps} steps ({current_posterior})")}
                     )
                else:
                    print("WandB logger not available.")
            except Exception as log_e:
                 print(f"WandB logging failed for test image: {log_e}")


        # Metrics Calculation
        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        batch_lpips = []
        with torch.no_grad():
            for i in range(lr_img.shape[0]):
                hr_img_i_cpu = hr_img[i:i+1].cpu()
                sr_img_i_cpu = sr_img[i:i+1].cpu()
                hr_img_np = (hr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
                sr_img_np = (sr_img_i_cpu.numpy().squeeze().transpose(1, 2, 0) + 1) / 2
                hr_h, hr_w = padding_info["hr"][i][1], padding_info["hr"][i][0]
                hr_img_np_crop = np.clip(hr_img_np[:hr_h, :hr_w, :], 0, 1)
                sr_img_np_crop = np.clip(sr_img_np[:hr_h, :hr_w, :], 0, 1)

                metrics["PSNR"].append(peak_signal_noise_ratio(hr_img_np_crop, sr_img_np_crop, data_range=1.0))
                ssim_win_size = min(7, hr_img_np_crop.shape[0], hr_img_np_crop.shape[1])
                if ssim_win_size % 2 == 0: ssim_win_size -=1
                if ssim_win_size < 3: ssim_win_size = 3
                metrics["SSIM"].append(structural_similarity(hr_img_np_crop, sr_img_np_crop, channel_axis=-1, data_range=1.0, win_size=ssim_win_size))
                metrics["MSE"].append(np.mean((hr_img_np_crop - sr_img_np_crop) ** 2))
                lpips_val_single = self.lpips(
                     normalize_tensor_01(hr_img[i:i+1]),
                     normalize_tensor_01(sr_img[i:i+1])
                ).item()
                batch_lpips.append(lpips_val_single)

        result = {
            "PSNR": np.mean(metrics["PSNR"]), "SSIM": np.mean(metrics["SSIM"]),
            "MSE": np.mean(metrics["MSE"]), "LPIPS": np.mean(batch_lpips),
            "time": elapsed_time,
        }
        self.test_step_outputs.append(result)
        return result


    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            print("No test outputs recorded for this epoch.")
            return

        avg_psnr = np.mean([x.get("PSNR", np.nan) for x in self.test_step_outputs])
        avg_ssim = np.mean([x.get("SSIM", np.nan) for x in self.test_step_outputs])
        avg_mse = np.mean([x.get("MSE", np.nan) for x in self.test_step_outputs])
        avg_lpips = np.mean([x.get("LPIPS", np.nan) for x in self.test_step_outputs])
        avg_time = np.mean([x.get("time", np.nan) for x in self.test_step_outputs])

        self.log_dict({
            "test/PSNR_epoch": avg_psnr, "test/SSIM_epoch": avg_ssim,
            "test/MSE_epoch": avg_mse, "test/LPIPS_epoch": avg_lpips,
            "test/time_per_batch_epoch": avg_time,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        print(f"\n--- Test Epoch Summary ---")
        print(f"Avg PSNR: {avg_psnr:.4f}")
        print(f"Avg SSIM: {avg_ssim:.4f}")
        print(f"Avg MSE: {avg_mse:.6f}")
        print(f"Avg LPIPS: {avg_lpips:.4f}")
        print(f"Avg Time/Batch: {avg_time:.4f}s")
        print(f"--------------------------\n")

        self.test_step_outputs.clear()

    # --- UPDATED discriminator_loss ---
    def discriminator_loss(
        self, hr_s_img: torch.Tensor, sr_s_img: torch.Tensor # Expects pixel-space images
    ) -> torch.Tensor:
        """Computes the discriminator loss using BCEWithLogitsLoss."""
        hr_s_img = hr_s_img.to(self.device)
        sr_s_img = sr_s_img.to(self.device) # Is already detached from training_step

        # --- Concatenation Logic (Inspired by dawir7, adapted) ---
        # Randomly decide the order: 0 -> (real, fake), 1 -> (fake, real)
        batch_size = hr_s_img.shape[0]
        perm = torch.randperm(batch_size, device=self.device)
        # Simple split for demonstration (can use torch.randint(0, 2, ...) as well)
        split_idx = batch_size // 2
        order_flags = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        order_flags[perm[split_idx:]] = 1.0 # ~Half will have order (fake, real)

        # Reshape flags for broadcasting
        order_flags_expanded = order_flags.view(-1, 1, 1, 1)

        # Create concatenated input based on random order
        first_img = torch.where(order_flags_expanded == 0, hr_s_img, sr_s_img)
        second_img = torch.where(order_flags_expanded == 0, sr_s_img, hr_s_img)
        discriminator_input = torch.cat([first_img, second_img], dim=1) # Concatenate along channel dim

        # Target labels depend on the order: 0.0 for (real, fake), 1.0 for (fake, real)
        # Note: Previous implementations might use 1.0 for real and 0.0 for fake *directly*.
        # This concatenation approach is different. Let's assume the target label
        # corresponds to whether the *second* image in the pair is the real one.
        # So, target = 0.0 if order is (real, fake), target = 1.0 if order is (fake, real).
        # This matches `order_flags`.
        target_labels = order_flags.view(-1, 1, 1, 1) # Reshape targets to match output

        # --- End Concatenation Logic ---

        # Get discriminator prediction (logits)
        logits = self.discriminator(discriminator_input) # Input should now be 6 channels

        # Calculate loss
        d_loss = self.adversarial_loss(logits, target_labels)

        # Calculate accuracy for EMA noise step adjustment
        with torch.no_grad():
             # Accuracy is based on whether the prediction matches the `order_flag`
            predicted_order = (torch.sigmoid(logits) > 0.5).float()
            current_accuracy = (predicted_order == target_labels).float().mean().item()
            self.last_d_acc = current_accuracy # Store for logging
            self.calculate_ema_noise_step(current_accuracy) # Update EMA noise step

        return d_loss

    # --- UPDATED generator_loss ---
    def generator_loss(
        self,
        x0_lat: torch.Tensor,       # Ground truth latent
        x_gen_0_lat: torch.Tensor, # Generated latent
        hr_img: torch.Tensor,       # Ground truth image (pixel space)
        sr_img: torch.Tensor,       # Generated image (pixel space, requires grad)
        hr_s_img: torch.Tensor,     # Noisy ground truth image (pixel space)
        sr_s_img: torch.Tensor,     # Noisy generated image (pixel space, requires grad)
    ) -> torch.Tensor:
        """Computes the generator loss."""
        x0_lat = x0_lat.to(self.device)
        x_gen_0_lat = x_gen_0_lat.to(self.device)
        hr_img = hr_img.to(self.device)
        sr_img = sr_img.to(self.device) # Requires grad
        hr_s_img = hr_s_img.to(self.device) # Used for concatenation input to D
        sr_s_img = sr_s_img.to(self.device) # Requires grad, used for concatenation input to D

        # 1. Content Loss (Latent Space MSE)
        content_loss = self.content_loss(x_gen_0_lat, x0_lat)

        # 2. Perceptual Loss (Pixel Space VGG/FeatureExtractor)
        perceptual_loss = torch.tensor(0.0).to(self.device)
        if self.vgg_loss is not None:
            perceptual_loss = self.vgg_loss(sr_img, hr_img) # Use grad-enabled sr_img

        # 3. Adversarial Loss
        # Create the concatenated input for the discriminator, just like in discriminator_loss
        # The generator *wants* the discriminator to misclassify the fake pairs.
        # If D expects target 0 for (real, fake) and 1 for (fake, real),
        # G wants D to predict 1 for (real, fake) and 0 for (fake, real).
        # This is equivalent to using the *opposite* target labels.

        batch_size = hr_s_img.shape[0]
        # Use the same random permutation logic for consistency (though randomness makes it less critical)
        perm = torch.randperm(batch_size, device=self.device)
        split_idx = batch_size // 2
        order_flags = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        order_flags[perm[split_idx:]] = 1.0
        order_flags_expanded = order_flags.view(-1, 1, 1, 1)

        # Create concatenated input *using grad-enabled sr_s_img*
        first_img_g = torch.where(order_flags_expanded == 0, hr_s_img, sr_s_img) # Use grad-enabled sr_s_img here
        second_img_g = torch.where(order_flags_expanded == 0, sr_s_img, hr_s_img) # And here
        discriminator_input_g = torch.cat([first_img_g, second_img_g], dim=1)

        # Target labels for the generator are the *opposite* of what the discriminator uses
        target_labels_g = (1.0 - order_flags).view(-1, 1, 1, 1)

        # Get discriminator logits for the fake pair
        fake_logits = self.discriminator(discriminator_input_g)

        # Calculate adversarial loss for the generator
        adversarial_loss = self.adversarial_loss(fake_logits, target_labels_g)


        # Combine losses
        g_loss = (
            content_loss
            + self.alfa_perceptual * perceptual_loss
            + self.alfa_adv * adversarial_loss
        )

        # Store detached components for logging
        self.last_g_content_loss = content_loss.detach()
        self.last_g_perceptual_loss = perceptual_loss.detach() if self.vgg_loss is not None else torch.tensor(0.0)
        self.last_g_adv_loss = adversarial_loss.detach()

        return g_loss


    def configure_optimizers( self ) -> list[torch.optim.Optimizer]:
        opt_g = torch.optim.Adam( self.generator.parameters(), lr=self.lr, betas=self.betas )
        opt_d = torch.optim.Adam( self.discriminator.parameters(), lr=self.lr, betas=self.betas )
        return [opt_g, opt_d]


    def calculate_ema_noise_step(self, current_accuracy: float) -> None:
         self.ema_mean = current_accuracy * (1 - self.ema_weight) + self.ema_mean * self.ema_weight
         target_s_float = (self.ema_mean - 0.5) * 2 * self.diffusion.timesteps
         self.s = int( torch.clamp( torch.tensor(target_s_float), 0, self.diffusion.timesteps - 1, ).item() )