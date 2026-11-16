from dataset_abl_planar import*

import torch
torch.cuda.empty_cache()
from denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch import* # Unet, GaussianDiffusion
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from utils import *
try:
    from plotting_utils import *          # optional: per-epoch monitoring figures (not part of this release)
except ImportError:
    def _no_figure(*args, **kwargs):
        return None
    plot_contour_abl = _no_figure
import argparse
import wandb  # Optional: for logging
from datetime import datetime
import gc
gc.collect()
torch.cuda.empty_cache()
import shutil
import torch.cuda.amp as amp  # Import AMP module

def save_samples(samples, epoch, save_dir):
    np.save(f'{save_dir}/samples_epoch_{epoch}.npy', samples.cpu().numpy())

def create_directories(base_dir):
    dirs = {
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'samples': os.path.join(base_dir, 'samples'),
        'logs': os.path.join(base_dir, 'logs'),
        'figures': os.path.join(base_dir, 'Figures/')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def train(config, model, diffusion, dataloader, device, save_dirs):
    lr = config.training.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()  # Initialize GradScaler

    best_loss = float('inf')

    log_file = open(os.path.join(save_dirs['logs'], 'training_log.txt'), 'w')
    wandb_log  = config.wandb.wandb_log
    project_Name  = config.wandb.project_Name
    wandb_log_step  = config.wandb.wandb_log_step
    wandb_key  = config.wandb.wandb_key


    run_Name = 'try'
    if wandb_log:
        wandb.login(key=wandb_key)
        settings=wandb.Settings(disable_job_creation=True)  # To disable error messages
        wandb.init(project=project_Name, name=run_Name, settings=settings)

    try:
        for epoch in range(config.training.num_epochs):
            model.train()
            total_loss = 0

            #for batch_idx, data in enumerate(dataloader):
            for batch_idx, (images , smooth_images ,real_lr_images,z0,Ug) in enumerate(dataloader):

                images = images.to(device)  # HR images
                smooth_images = smooth_images.to(device)  # LR images
                real_lr_images = real_lr_images.to(device)  # Alternative LR images
                z0=z0.to(device).float()
                Ug=Ug.to(device).float()


                physics_cond = torch.stack([z0,Ug],dim=1)


                optimizer.zero_grad()

                # Enable mixed precision
                with amp.autocast():
                 #   physics_cond = torch.zeros(16, 2, device=device)  # [B, 2]
                  #  physics_cond[:, 0] = 10.0  # Add wind speeds
                  #  physics_cond[:, 1] = 5.05    # Add roughness values


                    loss_paired , LR_maintain_loss_paired = diffusion(images, smooth_images, paired=True,physics_cond=physics_cond)

                    if config.training.unpaired_weight > 0:
                        loss_unpaired,LR_maintain_loss_unpaired = diffusion(images, real_lr_images, paired=False,physics_cond=physics_cond)
                    else:
                        loss_unpaired = torch.tensor([0.0], device=device)
                        LR_maintain_loss_unpaired = torch.tensor([0.0], device=device)


                    #'''
                    loss = (config.training.paired_weight * loss_paired + config.training.unpaired_weight * loss_unpaired
                            + config.training.LR_maintain_weight_paired*LR_maintain_loss_paired
                            + config.training.LR_maintain_weight_unpaired*LR_maintain_loss_unpaired)
                    #'''

                # Use GradScaler for stable mixed precision training
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()  # Update the scaler

                total_loss += loss.item()

                if wandb_log and batch_idx % wandb_log_step == 0:
                    wandb.log({"Paired loss": loss_paired.cpu().detach().numpy()})
                    wandb.log({"Unpaired loss": loss_unpaired.cpu().detach().numpy()})
                    wandb.log({"LR maintain loss paired": LR_maintain_loss_paired.cpu().detach().numpy()})
                    wandb.log({"LR maintain loss unpaired": LR_maintain_loss_unpaired.cpu().detach().numpy()})

                if batch_idx % 10 == 0:
                    msg = f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}'
                    print(msg)
                    log_file.write(msg + '\n')
                    log_file.flush()

            avg_loss = total_loss / len(dataloader)
            msg = f'Epoch {epoch} Average Loss: {avg_loss:.4f}'
            print(msg)
            log_file.write(msg + '\n')

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(save_dirs['checkpoints'], 'best_model.pt'))

            if (epoch + 1) % config.save.plot_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(save_dirs['checkpoints'], f'checkpoint_epoch_{epoch}.pt'))

                model.eval()

                with torch.no_grad():
                    batch_size = smooth_images.shape[0]
                    samples = diffusion.sample(batch_size=batch_size, cond=smooth_images,physics_cond=physics_cond)

                    batch_size = real_lr_images.shape[0]
                    samples_pred = diffusion.sample(batch_size=batch_size, cond=real_lr_images,physics_cond=physics_cond)

                    save_name = save_dirs['figures'] + rf'cont_{epoch}.png'
                    plot_contour_abl(smooth_images, images, samples, real_lr_images, samples_pred, save_name)

                    save_name = save_dirs['figures']  + rf'dist_{epoch}.png'
                    #plot_distribution(smooth_images, images, samples, real_lr_images, samples_pred, save_name)

                    save_name = save_dirs['figures']  + rf'spectrum_{epoch}.png'
                   # plot_spectrum(smooth_images, images, samples, real_lr_images, samples_pred, save_name)

    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dirs['checkpoints'], 'interrupted_model.pt'))

    finally:
        log_file.close()


def create_model(
    image_size,
    channels=3,
    dim=64,
    dim_mults=(1, 2, 4, 8),
    timesteps = 1000,
    objective = 'pred_v',
):
    model = ConditionalUnet(
        dim=dim,
        channels=channels,
        dim_mults=dim_mults,
        with_conditioning=True
    )

    diffusion = ConditionalGaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        objective = objective,
    )

    return model, diffusion


def main(config_path):
   # size = [64, 64]
    config = load_config(config_path)
    size = config.data.size


    batch_size = config.training.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    base_directory =  config.save.directory

    now = datetime.now()  # Get current date and time
    date_time_str = now.strftime("%Y_%m_%d %H_%M_%S")  # Format as string
    base_directory_run = base_directory + '/'+date_time_str
    save_dirs = create_directories(base_directory_run)

    shutil.copy(config_path, base_directory_run  +'/config.yaml')


    model, diffusion = create_model(
        image_size=(size[0], size[1]),
        channels=config.model.channels,
        dim=config.model.dim,
        dim_mults=config.model.dim_mults,
        timesteps=config.model.timesteps,
        objective = config.model.objective
    )
    model = model.to(device)
    diffusion = diffusion.to(device)

    if config.problem != 'abl':
        raise ValueError(f"unsupported problem '{config.problem}'; this release only contains the ABL case")
    dataset = abl_Dataset(config)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.training.num_workers)

    print("Starting training...")

    train(config, model, diffusion, dataloader, device, save_dirs)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Diffusion Model for Texture Enhancement')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
