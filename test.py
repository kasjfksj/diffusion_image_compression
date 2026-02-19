from uqdm_sd import load_checkpoint_SD, load_data, load_data_from_folder
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import resize, to_pil_image
import torch
from uqdm import load_checkpoint
import numpy as np
import os
import gc
# Reduce fragmentation by using expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Setup


# Load models
model_sd = load_checkpoint_SD('checkpoints/uqdm-small', use_sd=True)


# # Load data
# train_iter, eval_iter = load_data('ImageNet64', model_sd.config.data)
train_iter, eval_iter = load_data_from_folder()
image = next(iter(eval_iter))
image = resize(image, size=(512,512))
# import torch.nn.functional as F
print(image.shape)  # Should be [B, C, 256, 256]
# Compress/decompress with SD (all stages)
compressed_sd = model_sd.compress(image)
recons_sd = model_sd.decompress(compressed_sd, image.shape, recon_method='ancestral')
bits_sd = [len(b) * 8 for b in compressed_sd]
bpps_sd = np.cumsum(bits_sd) / np.prod(image.shape) * 3
def compute_psnr(original, reconstructed):
    """
    original, reconstructed: tensors of shape [B, C, H, W], values in [0, 1]
    Returns mean PSNR in dB across the batch.
    """
    mse = torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])  # per image
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean().item()

# Compute PSNR for each progressive stage
psnr_sd = []
for stage in recons_sd:
    # Normalize both to [0, 1] if they aren't already
    orig_norm = (image - image.min()) / (image.max() - image.min())
    recon_norm = (stage - stage.min()) / (stage.max() - stage.min())
    psnr_sd.append(compute_psnr(orig_norm, recon_norm))

print(f"PSNR per stage (dB): {np.round(psnr_sd, 3)}")
# After deleting first model

# model_no_sd = load_checkpoint('checkpoints/uqdm-small')
# compressed_no_sd = model_no_sd.compress(image)
# recons_no_sd = model_no_sd.decompress(compressed_no_sd, image.shape)
# bits_no_sd = [len(b) * 8 for b in compressed_no_sd]
# bpps_no_sd = np.cumsum(bits_no_sd) / np.prod(image.shape) * 3

# print(f"SD stages: {len(recons_sd)}, No-SD stages: {len(recons_no_sd)}")
print(f"BPPs SD: {np.round(bpps_sd, 3)}")
# print(f"BPPs No-SD: {np.round(bpps_no_sd, 3)}")

# Create comparison grid: original + all SD stages + all no-SD stages
num_images = min(2, image.shape[0])
max_stages = max(len(recons_sd),0)
ncols = 1 + max_stages * 2  # original + SD stages + no-SD stages

rows = []
for i in range(num_images):
    rows.append(image[i])
    for stage in recons_sd:
        rows.append(stage[i])
    
    # # # No-SD stages
    # for stage in recons_no_sd:
    #     rows.append(stage[i])

grid = make_grid(torch.stack(rows), nrow=ncols, normalize=True, padding=2)
save_image(grid, 'sd_comparison_all_stages.png')

print(f'Saved to sd_comparison_all_stages_sd_scheduler.png')
# print(f'Layout: Original | SD stages ({len(recons_sd)}) | No-SD stages ({len(recons_no_sd)})')
print(f'Layout: Original | SD stages ({len(recons_sd)}) ')