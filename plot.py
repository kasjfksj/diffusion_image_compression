import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from uqdm_sd import load_checkpoint_SD, load_data_from_folder,load_data
from vdm import load_checkpoint_SD_VDM

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# ── STEP 1: Evaluate UQDM ─────────────────────────────────────────────────
if not os.path.exists('uqdm_results.npz'):
    model = load_checkpoint_SD('checkpoints/uqdm-small')
    _, eval_iter = load_data('ImageNet64', model.config.data)
    print("Evaluating UQDM model...")
    bpps, psnrs = model.evaluate(eval_iter, n_batches=10, seed=seed)
    np.savez('uqdm_results.npz', bpps=np.array(bpps), psnrs=np.array(psnrs))
    del model
    torch.cuda.empty_cache()
    print("UQDM results saved.")

# ── STEP 2: Evaluate SD ────────────────────────────────────────────────────
if not os.path.exists('sd_results.npz'):
    model_2 = load_checkpoint_SD_VDM('checkpoints/uqdm-small')
    _, eval_iter = load_data('ImageNet64', model_2.config.data)
    print("Evaluating SD model...")
    bpps, psnrs = model_2.evaluate(eval_iter, n_batches=10, seed=seed)
    np.savez('sd_results.npz', bpps=np.array(bpps), psnrs=np.array(psnrs))
    del model_2
    torch.cuda.empty_cache()
    print("SD results saved.")

# ── STEP 3: Plot ───────────────────────────────────────────────────────────
uqdm = np.load('uqdm_results.npz')
sd = np.load('sd_results.npz')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(uqdm['bpps'], uqdm['psnrs'], '-', label='UQDM', linewidth=2, color='orange')
ax.plot(sd['bpps'], sd['psnrs'], '-', label='SD', linewidth=2, color='blue')
ax.set_xlabel('Rate (BPP)', fontsize=13)
ax.set_ylabel('PSNR (dB)', fontsize=13)
ax.set_title('UQDM vs SD: Rate-Distortion Curve (64*64)', fontsize=14)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uqdm_sd_comparison_64*64.png', dpi=150, bbox_inches='tight')
print("Saved plot to uqdm_sd_comparison_64*64.png")