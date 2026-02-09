from uqdm import load_checkpoint, load_data
from torchvision.utils import save_image, make_grid
import torch
import numpy as np

# Setup
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load models
model_sd = load_checkpoint('checkpoints/uqdm-small', use_sd=True)
model_no_sd = load_checkpoint('checkpoints/uqdm-small', use_sd=False)

# Load data
train_iter, eval_iter = load_data('ImageNet64', model_sd.config.data)
image = next(iter(eval_iter))

print(f"Image shape: {image.shape}, range: [{image.min():.2f}, {image.max():.2f}]")

# Compress/decompress with SD (all stages)
compressed_sd = model_sd.compress(image)
recons_sd = model_sd.decompress(compressed_sd, image.shape)
bits_sd = [len(b) * 8 for b in compressed_sd]
bpps_sd = np.cumsum(bits_sd) / np.prod(image.shape) * 3

# Compress/decompress without SD (all stages)
compressed_no_sd = model_no_sd.compress(image)
recons_no_sd = model_no_sd.decompress(compressed_no_sd, image.shape)
bits_no_sd = [len(b) * 8 for b in compressed_no_sd]
bpps_no_sd = np.cumsum(bits_no_sd) / np.prod(image.shape) * 3

print(f"SD stages: {len(recons_sd)}, No-SD stages: {len(recons_no_sd)}")
print(f"BPPs SD: {np.round(bpps_sd, 3)}")
print(f"BPPs No-SD: {np.round(bpps_no_sd, 3)}")

# Create comparison grid: original + all SD stages + all no-SD stages
num_images = min(2, image.shape[0])
max_stages = max(len(recons_sd), len(recons_no_sd))
ncols = 1 + max_stages * 2  # original + SD stages + no-SD stages

rows = []
for i in range(num_images):
    # Original
    rows.append(image[i])
    
    # SD stages
    for stage in recons_sd:
        rows.append(stage[i])
    
    # No-SD stages
    for stage in recons_no_sd:
        rows.append(stage[i])

grid = make_grid(torch.stack(rows), nrow=ncols, normalize=True, padding=2)
save_image(grid, 'sd_comparison_all_stages.png')

print(f'Saved to sd_comparison_all_stages.png')
print(f'Layout: Original | SD stages ({len(recons_sd)}) | No-SD stages ({len(recons_no_sd)})')