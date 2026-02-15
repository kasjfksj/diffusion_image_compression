from uqdm_sd import load_checkpoint_SD, load_data, load_data_from_folder
from torchvision.utils import save_image, make_grid
import torch
from uqdm import load_checkpoint
import numpy as np
import os
import gc
# Reduce fragmentation by using expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Setup
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load models
model_sd = load_checkpoint_SD('checkpoints/uqdm-small', use_sd=True)


# # Load data
train_iter, eval_iter = load_data_from_folder()

image = next(iter(eval_iter))
import torch.nn.functional as F
print(image.shape)  # Should be [B, C, 256, 256]
# Compress/decompress with SD (all stages)
image = torch.nn.functional.interpolate(image.float(), size=(256, 256), mode='bilinear', align_corners=False)

compressed_sd = model_sd.compress(image)
recons_sd = model_sd.decompress(compressed_sd, image.shape)
bits_sd = [len(b) * 8 for b in compressed_sd]
bpps_sd = np.cumsum(bits_sd) / np.prod(image.shape) * 3
# After deleting first model

# Create comparison grid: original + all SD stages + all no-SD stages
num_images = min(2, image.shape[0])
max_stages = max(len(recons_sd),0)
ncols = 1 + max_stages * 2  # original + SD stages + no-SD stages
print(f"BPPs SD: {np.round(bpps_sd, 3)}")

rows = []
for i in range(num_images):
    rows.append(image[i])
    for stage in recons_sd:
        rows.append(stage[i])
    


grid = make_grid(torch.stack(rows), nrow=ncols, normalize=True, padding=2)
save_image(grid, 'sd_comparison_all_stages_apple.png')

