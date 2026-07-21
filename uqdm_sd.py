import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.distributions import constraints, TransformedDistribution, SigmoidTransform, AffineTransform
from torch.distributions import Normal, Uniform
from torch.distributions.kl import kl_divergence
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from tensorflow_compression.python.ops import gen_ops
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
import os
from pathlib import Path
from contextlib import contextmanager
import zipfile
from tqdm import tqdm
from itertools import islice
from ml_collections import ConfigDict
import math
import time
import lpips   
from torch.utils.data import Subset, random_split
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms.functional as TF
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

DATASET_PATH = {
    'ImageNet64': 'data/imagenet64/',
}

DDIM_SCHEDULE = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761,
                 741, 721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521,
                 501, 481, 461, 441, 421, 401, 381, 361, 341, 321, 301, 281,
                 261, 241, 221, 201, 181, 161, 141, 121, 101,  81,  61,  41,
                  21,  10,   5,   0]
def softplus_inverse(x):
    """Helper which computes the inverse of `tf.nn.softplus`."""
    import math
    import numpy as np
    return math.log(np.expm1(x))
class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.device = device

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W] float in [0, 255]
        returns: [B, 2048] inception features
        """
        images = images.clamp(0, 255).to(torch.uint8).to(self.device)  # uint8 as torchmetrics expects
        self.fid_metric.inception.eval()
        feats = self.fid_metric.inception(images)
        return feats.cpu()
def save_features(
    features: torch.Tensor,
    save_dir: str,
    split: str,           # e.g. 'real' or 'fake'
    timestep: int,        # e.g. ts_s value
    shard_id: int = 0,    # increment per batch to avoid overwriting
):
    """
    Save a [B, 2048] feature tensor to:
    {save_dir}/t{timestep:04d}/{split}_shard{shard_id:05d}.pt
    """
    folder = os.path.join(save_dir, f"t{timestep:04d}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{split}_shard{shard_id:05d}.pt")
    torch.save(features, path)
    return path


def list_timesteps(save_dir: str) -> list[int]:
    """Return all timesteps that have saved features under save_dir."""
    dirs = [d for d in os.listdir(save_dir) if d.startswith("t") and os.path.isdir(os.path.join(save_dir, d))]
    return sorted(int(d[1:]) for d in dirs)
def compute_fid_from_features(
    real_feats: torch.Tensor,   # [N, 2048]
    fake_feats: torch.Tensor,   # [M, 2048]
) -> float:
    """
    Compute FID directly from pre-extracted feature tensors,
    bypassing any image re-processing.
    """
    def _stats(f):
        mu  = f.mean(dim=0).numpy()          # [2048]
        sig = np.cov(f.numpy(), rowvar=False) # [2048, 2048]
        return mu, sig

    mu_r, sig_r = _stats(real_feats.float())
    mu_f, sig_f = _stats(fake_feats.float())

    diff = mu_r - mu_f
    # Symmetric matrix square root via eigen-decomposition (more stable than scipy sqrtm)
    vals_r, vecs_r = np.linalg.eigh(sig_r)
    vals_f, vecs_f = np.linalg.eigh(sig_f)
    sqrt_r = vecs_r @ np.diag(np.sqrt(np.maximum(vals_r, 0))) @ vecs_r.T
    covmean = sqrt_r @ sig_f @ sqrt_r
    vals_c, vecs_c = np.linalg.eigh(covmean)
    sqrt_cov = vecs_c @ np.diag(np.sqrt(np.maximum(vals_c, 0))) @ vecs_c.T

    fid = float(diff @ diff + np.trace(sig_r + sig_f - 2 * sqrt_cov))
    return fid


def compute_fid_all_timesteps(
    save_dir: str,
    min_samples: int = 2048,
    device: str = "cuda",
) -> dict[int, float]:
    """
    Load stored features for every timestep under save_dir and compute FID.
    Returns {timestep -> fid_score}.
    """
    results = {}
    for ts in list_timesteps(save_dir):
        try:
            real_feats = load_features(save_dir, split="real", timestep=ts)
            fake_feats = load_features(save_dir, split="fake", timestep=ts)
        except FileNotFoundError as e:
            print(f"[FID] Skipping t={ts}: {e}")
            continue

        n_real, n_fake = real_feats.shape[0], fake_feats.shape[0]
        if min(n_real, n_fake) < min_samples:
            print(f"[FID] Skipping t={ts}: only {n_real} real / {n_fake} fake samples")
            continue

        fid = compute_fid_from_features(real_feats, fake_feats)
        results[ts] = fid
        print(f"[FID] t={ts:4d} | real={n_real} fake={n_fake} | FID={fid:.3f}")

    return results
def load_features(
    save_dir: str,
    split: str,
    timestep: int,
) -> torch.Tensor:
    """
    Load and concatenate all shards for a given timestep and split.
    Returns [N, 2048].
    """
    folder = os.path.join(save_dir, f"t{timestep:04d}")
    shards = sorted(
        f for f in os.listdir(folder)
        if f.startswith(split) and f.endswith(".pt")
    )
    if not shards:
        raise FileNotFoundError(f"No shards found for split='{split}' at {folder}")
    return torch.cat([torch.load(os.path.join(folder, s)) for s in shards], dim=0)



class SD15ScoreNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mcfg = config.model
        self.SOFTPLUS_INV1 = softplus_inverse(1.0)

        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        ).cuda()
        self.sd_scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )

        # Scale head takes the penultimate UNet latent (320 channels from conv_in path)
        # SD1.5 UNet final conv_out goes from 320 -> 4, so we hook before it
        penultimate_channels = self.unet.conv_out.in_channels  # typically 320
        self.scale_head = nn.Sequential(
            nn.Conv2d(penultimate_channels, 128, kernel_size=1),  # index 0 — was 128, not 64
            nn.SiLU(),                                             # index 1
            nn.Conv2d(128, 64, kernel_size=1),                    # index 2 — was 128→64, not 64→4
            nn.SiLU(),                                             # index 3
            nn.Conv2d(64, 4, kernel_size=1),                      # index 4 — the "unexpected" layer
        ).cuda()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        # # Zero-init the last layer for stable training start
        nn.init.kaiming_normal_(self.scale_head[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.scale_head[0].bias)
        nn.init.xavier_normal_(self.scale_head[-1].weight, gain=0.01)  # small but non-zero
        nn.init.zeros_(self.scale_head[-1].bias)
        self.unet.enable_gradient_checkpointing()
        self._penultimate_latent = None
        self._register_penultimate_hook()

    def _register_penultimate_hook(self):
        """Hook the layer immediately before conv_out to capture its output."""
        def hook_fn(module, input, output):
            # input[0] is the tensor fed into conv_out — the penultimate latent
            self._penultimate_latent = input[0]

        self.unet.conv_out.register_forward_hook(hook_fn)

    def softplus_init1(self, x):
        return torch.nn.functional.softplus(x + self.SOFTPLUS_INV1)

    def forward(self, z, g_t):
        g_t = g_t.expand(z.shape[0])

        alpha2_target = torch.sigmoid(-g_t)
        alphas_cumprod = self.sd_scheduler.alphas_cumprod.to(z.device)
        diffs = (alphas_cumprod.unsqueeze(0) - alpha2_target.unsqueeze(1)).abs()
        timesteps = diffs.argmin(dim=1).long()
        null_cond = torch.zeros(z.shape[0], 77, 768, device=z.device, dtype=z.dtype)

        # UNet forward — no grad for the UNet weights, but hook captures the
        # penultimate latent so scale_head can still receive gradients through it
        eps_hat = self.unet(z, timesteps, encoder_hidden_states=null_cond).sample

        # _penultimate_latent was captured inside the no_grad block, so we must
        # re-enable grad for the scale_head branch
        penultimate = self._penultimate_latent

        pred_scale_factors = self.softplus_init1(self.scale_head(penultimate))

        return eps_hat, pred_scale_factors
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Code from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    which is modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py
    and partially based on https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


"""
Data and Checkpoint Loading
"""


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class ToIntTensor:
    def __init__(self, resolution=64):
        self.resolution = resolution

    def __call__(self, image):
        image = torch.as_tensor(
            image.reshape(3, self.resolution, self.resolution),
            dtype=torch.uint8
        )
        return image


class NPZLoader(Dataset):
    """
    Load from a batched numpy dataset.
    Keeps one data batch loaded in memory, so load idx sequentially for fast sampling
    """

    def __init__(self, path, train=True, transform=None, remove_duplicates=True):
        self.path = path
        if train:
            self.files = list(Path(path).glob('*train*.npz'))
        else:
            self.files = list(Path(path).glob('*val*.npz'))
        self.batch_lens = [self.npz_len(f) for f in self.files]
        self.anchors = np.cumsum([0] + self.batch_lens)
        self.removed_idxs = [[] for _ in range(len(self.files))]
        self.transform = transform
        self.cache_fid = None
        self.cache_npy = None

    # https://stackoverflow.com/questions/68224572/how-to-determine-the-shape-size-of-npz-file
    @staticmethod
    def npz_len(npz):
        """
        Takes a path to an .npz file, which is a Zip archive of .npy files and returns the batch size of stored data,
        i.e. of the first .npy found
        """
        with zipfile.ZipFile(npz) as archive:
            for name in archive.namelist():
                if not name.endswith('.npy'):
                    continue
                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                return shape[0]

    def load_npy(self, fid):
        if not fid == self.cache_fid:
            self.cache_fid = fid
            self.cache_npy = np.load(str(self.files[fid]))['data']
            self.cache_npy = np.delete(self.cache_npy, self.removed_idxs[fid], axis=0)
        return self.cache_npy

    def __len__(self):
        return self.anchors[-1]

    def __getitem__(self, idx):
        fid = np.argmax(idx < self.anchors) - 1
        idx = idx - self.anchors[fid]
        numpy_array = self.load_npy(fid)[idx]
        if self.transform is not None:
            torch_array = self.transform(numpy_array)
        return torch_array


def load_data_from_folder(folder_path='data_1/', resolution=512):
    """
    Load images from a folder and split into train (90%) and eval (10%) sets.
    Returns infinitely looping training iterator and finite eval iterator.
    """
    class ImageFolderFlat(Dataset):
        def __init__(self, folder_path):
            self.folder_path = Path(folder_path)
            self.transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).byte()),  # convert to [0, 255] uint8
            ])
            self.image_paths = (
                list(self.folder_path.glob('*.jpg')) +
                list(self.folder_path.glob('*.png')) +
                list(self.folder_path.glob('*.jpeg')) +
                list(self.folder_path.glob('*.JPEG'))
            )

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

    full_dataset = ImageFolderFlat(folder_path)

    total_size = len(full_dataset)
    train_size = 1
    eval_size = total_size - train_size
    train_data, eval_data = random_split(full_dataset, [train_size, eval_size])

    train_iter = DataLoader(train_data, batch_size=1, shuffle=True,
                            pin_memory=True, num_workers=0)
    eval_iter = DataLoader(eval_data, batch_size=32, shuffle=False,
                           pin_memory=True, num_workers=0)

    train_iter = cycle(train_iter)

    return train_iter, eval_iter


def load_data(dataspec, cfg):
    """
    Load datasets, with finite eval set and infinitely looping training set
    """
    if not dataspec in DATASET_PATH.keys():
        raise ValueError('Unknown dataset. Add dataspec to load_data() or use one of \n%s' % list(DATASET_PATH.keys()))

    if dataspec in ['ImageNet64']:
        train_data, eval_data = [NPZLoader(DATASET_PATH[dataspec], train=mode, transform=ToIntTensor()) for mode in
                                 [True, False]]

    # Limit to only 5 data points
    train_data = Subset(train_data, range(min(5, len(train_data))))
    eval_data = Subset(eval_data, range(min(5, len(eval_data))))
    train_iter, eval_iter = [DataLoader(d, batch_size=cfg.batch_size, shuffle=cfg.get('shuffle', False),
                                        pin_memory=cfg.get('pin_memory', True), num_workers=cfg.get('num_workers', 1))
                             for d in [train_data, eval_data]]
    train_iter = cycle(train_iter)

    return train_iter, eval_iter


def load_checkpoint_SD(config_path=None, ckpt_path=None):
    """
    Load model from checkpoint.

    Input:
    ------
    config_path: path to a folder containing config.json. If None, uses default config.
    ckpt_path:   path to a .pt file containing scale_head weights. If None, uses random init.
    """
    if config_path is not None:
        with open(os.path.join(config_path, 'config.json'), 'r') as f:
            config = ConfigDict(json.load(f))
    else:
        config = ConfigDict()  # default config

    model = UQDM_SD(config).to(device)

    # Only load weights if an explicit ckpt_path is given
    if ckpt_path is not None:
        model.load(ckpt_path)

    return model


"""
UQDM: Diffusion model, Distributions, Entropy Coding, UQDM
"""

@contextmanager
def local_seed(seed, i=0):
    # Allow for local randomness, use hashing to get unique local seeds for subsequent draws
    if seed is None:
        yield
    else:
        with torch.random.fork_rng():
            local_seed = hash((seed, i)) % (2 ** 32)
            torch.manual_seed(local_seed)
            yield


class LogisticDistribution(TransformedDistribution):
    """
    Creates a logistic distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the affine transform of a standard logistic distribution.
    Patterned after https://github.com/pytorch/pytorch/blob/main/torch/distributions/logistic_normal.py

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution

    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        base_dist = Uniform(torch.tensor(0, dtype=loc.dtype, device=loc.device),
                            torch.tensor(1, dtype=loc.dtype, device=loc.device))
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        super().__init__(
            base_dist, transforms, validate_args=validate_args
        )

    @property
    def mean(self):
        return self.loc

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticDistribution, _instance)
        return super().expand(batch_shape, _instance=new)

    def cdf(self, x):
        # Should be numerically more stable than the default.
        return torch.sigmoid((x - self.loc) / self.scale)

    @staticmethod
    def log_sigmoid(x):
        # A numerically more stable implementation of torch.log(torch.sigmoid(x)).
        # c.f. https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_sigmoid.html#jax.nn.log_sigmoid
        return -torch.nn.functional.softplus(-x)

    def log_cdf(self, x):
        standardized = (x - self.loc) / self.scale
        return self.log_sigmoid(standardized)

    def log_survival_function(self, x):
        standardized = (x - self.loc) / self.scale
        return self.log_sigmoid(- standardized)


class NormalDistribution(torch.distributions.Normal):
    """
    Overrides the Normal distribution to add a numerically more stable log_cdf
    """

    def log_cdf(self, x):
        x = (x - self.loc) / self.scale
        # more stable, for float32 ported from JAX, using log(1-x) ~= -x, x >> 1
        # for small x
        x_l = torch.clip(x, max=-10)
        log_scale = -0.5 * x_l ** 2 - torch.log(-x_l) - 0.5 * np.log(2. * np.pi)
        # asymptotic series
        even_sum = torch.zeros_like(x)
        odd_sum = torch.zeros_like(x)
        x_2n = x_l ** 2
        for n in range(1, 3 + 1):
            y = np.prod(np.arange(2 * n - 1, 1, -2)) / x_2n
            if n % 2:
                odd_sum += y
            else:
                even_sum += y
            x_2n *= x_l ** 2
        x_lower = log_scale + torch.log(1 + even_sum - odd_sum)
        return torch.where(
            x > 5, -torch.special.ndtr(-x),
            torch.where(x > -10, torch.special.ndtr(torch.clip(x, min=-10)).log(), x_lower))

    def log_survival_function(self, x):
        raise NotImplementedError


class UniformNoisyDistribution(torch.distributions.Distribution):
    """
    Add uniform noise U[-delta/2, +delta/2] to a distribution.
    Adapted from https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/distributions/uniform_noise.py
    Also see https://pytorch.org/docs/stable/_modules/torch/distributions/distribution.html
    """

    arg_constraints = {}

    def __init__(self, base_dist, delta):
        super().__init__()
        self.base_dist = base_dist
        self.delta = delta  # delta is the noise width.
        self.half = delta / 2.
        self.log_delta = torch.log(delta)

    def sample(self, sample_shape=torch.Size([])):
        x = self.base_dist.sample(sample_shape)
        x += self.delta * torch.rand(x.shape, dtype=x.dtype, device=x.device) - self.half
        return x

    @property
    def mean(self):
        return self.base_dist.mean

    def discretize(self, u, tail_mass=2 ** -8):
        """
        Turn the continuous distribution into a discrete one by discretizing to the grid u + k * delta.
        Returns the pmf of k = round((x -  p_mean) / delta + u) as this is used for UQ, ignoring outlier values in the tails.
        """
        # For quantiles: Because p(x) = (G(x+d/2) - G(x-d/2))/d,
        # P(X <= x) = 1/d int_{x-d/2}^{x+d/2} G(u) du <= G(x+d/2) or >= G(x-d/2) which might be tighter for small d
        # P(X <= G^-1(a) - d/2) <= a, P(K <= (G^-1(a) - p_mean)/d - 1/2 - p_mean/d + u) <= a
        L = torch.floor((self.base_dist.icdf(tail_mass / 2) - self.base_dist.mean).min() / self.delta - 0.5)
        R = torch.ceil((self.base_dist.icdf(1 - tail_mass / 2) - self.base_dist.mean).max() / self.delta + 0.5)
        x = (torch.arange(L, R + 1, device=u.device).reshape(-1, *4*[1]) - u) * self.delta + self.base_dist.mean
        # Assume pdf is locally linear then ln(p(x+-d/2)) = ln(p(x)*d) = ln(p(x)) + ln(d)
        logits = self.log_prob(x) + torch.log(self.delta)
        return OverflowCategorical(logits=logits, L=L, R=R)

    def log_prob(self, y):
        if not hasattr(self.base_dist, "log_cdf"):
            raise NotImplementedError(
                "`log_prob()` is not implemented unless the base distribution implements `log_cdf()`.")
        try:
            return self._log_prob_with_logsf_and_logcdf(y)
        except NotImplementedError:
            return self._log_prob_with_logcdf(y)

    @staticmethod
    def _logsum_expbig_minus_expsmall(big, small):
        # Numerically stable evaluation of log(exp(big) - exp(small)).
        # https://github.com/tensorflow/compression/blob/a41fc70fc092bc6b72d5075deec34cbb47ef9077/tensorflow_compression/python/distributions/uniform_noise.py#L33
        return torch.where(
            torch.isinf(big), big, torch.log1p(-torch.exp(small - big)) + big
        )

    def _log_prob_with_logcdf(self, y):
        return self._logsum_expbig_minus_expsmall(
            self.base_dist.log_cdf(y + self.half), self.base_dist.log_cdf(y - self.half)) - self.log_delta

    def _log_prob_with_logsf_and_logcdf(self, y):
        """Compute log_prob(y) using log survival_function and cdf together."""
        # There are two options that would be equal if we had infinite precision:
        # Log[ sf(y - .5) - sf(y + .5) ]
        #   = Log[ exp{logsf(y - .5)} - exp{logsf(y + .5)} ]
        # Log[ cdf(y + .5) - cdf(y - .5) ]
        #   = Log[ exp{logcdf(y + .5)} - exp{logcdf(y - .5)} ]
        h = self.half
        base = self.base_dist
        logsf_y_plus = base.log_survival_function(y + h)
        logsf_y_minus = base.log_survival_function(y - h)
        logcdf_y_plus = base.log_cdf(y + h)
        logcdf_y_minus = base.log_cdf(y - h)

        # Important:  Here we use select in a way such that no input is inf, this
        # prevents the troublesome case where the output of select can be finite,
        # but the output of grad(select) will be NaN.

        # In either case, we are doing Log[ exp{big} - exp{small} ]
        # We want to use the sf items precisely when we are on the right side of the
        # median, which occurs when logsf_y < logcdf_y.
        condition = logsf_y_plus < logcdf_y_plus
        big = torch.where(condition, logsf_y_minus, logcdf_y_plus)
        small = torch.where(condition, logsf_y_plus, logcdf_y_minus)
        return self._logsum_expbig_minus_expsmall(big, small) - self.log_delta


class OverflowCategorical(torch.distributions.Categorical):
    """
    Discrete distribution over [L, L+1, ..., R-1, R] with LaPlace-based tail_masses for values <L and >R.
    """

    def __init__(self, logits, L, R):
        self.L = L
        self.R = R
        # stable version of log(1 - sum_i exp(logp_i))
        self.overflow = torch.log(torch.clip(- torch.expm1(torch.logsumexp(logits, dim=0)), min=0))
        super().__init__(logits=torch.movedim(torch.cat([logits, self.overflow[None]], dim=0), 0, -1))


class EntropyModel:
    """
    Entropy codec for discrete data based on Arithmetic Coding / Range Coding.
    Adapted from https://github.com/tensorflow/compression.
    For learned backward variances every symbol has a unique coding prior that requires a unique cdf table,
    which is computed in parallel here.
    """

    def __init__(self, prior, range_coder_precision=16):
        """

        Inputs:
        -------
        prior     - [Categorical or OverflowCategorical] prior model over integers (optionally with allocated tail mass
                    which will be encoded via Elias gamma code embedded into the range coder).
        range_coder_precision - precision passed to the range coding op, how accurately prior is quantized.
        """
        super().__init__()
        self.prior = prior
        self.prior_shape = self.prior.probs.shape[:-1]
        self.precision = range_coder_precision

        # Build quantization tables
        # Build quantization tables — stay on CPU throughout
        total = 2 ** self.precision
        probs = self.prior.probs.reshape(-1, self.prior.probs.shape[-1]).cpu().float()
        quantized_pdf = torch.round(probs * total).to(torch.int32)
        quantized_pdf = torch.clip(quantized_pdf, min=1)

        # Normalize: reduce overflowing rows
        while True:
            sums = quantized_pdf.sum(dim=-1)
            mask = sums > total
            if not mask.any():
                break
            penalty = probs[mask] * (torch.log2(1 + 1 / (quantized_pdf[mask] - 1)))
            idx = penalty.nan_to_num(torch.inf).argmin(dim=-1)
            excess = sums[mask] - total
            quantized_pdf[mask, idx] -= excess.clamp(max=quantized_pdf[mask, idx] - 1)  # keep min 1
        # Normalize: increase underflowing rows

        while True:
            sums = quantized_pdf.sum(dim=-1)
            mask = sums < total
            if not mask.any():
                break
            penalty = probs[mask] * (torch.log2(1 + 1 / quantized_pdf[mask]))
            idx = penalty.argmax(dim=-1)
            deficit = total - sums[mask]
            quantized_pdf[mask, idx] += deficit

        quantized_cdf = torch.cumsum(quantized_pdf, dim=-1)

        # Keep everything on CPU — compress/decompress call .cpu() anyway
        self.quantized_cdf = torch.cat([
            -self.precision * torch.ones((quantized_pdf.shape[0], 1), dtype=torch.float32),
            torch.zeros((quantized_pdf.shape[0], 1), dtype=torch.float32),
            quantized_cdf.float()
        ], dim=-1).reshape(-1)

        self.indexes = torch.arange(quantized_pdf.shape[0], dtype=torch.int32)
        self.offsets = self.prior.L if type(self.prior) is OverflowCategorical else 0

    def compress(self, x):
        """
        Compresses a floating-point tensor to a bit string with the discretized prior.
        """

        x = (x - self.offsets).to(torch.int32).reshape(-1).cpu()
        codec = gen_ops.create_range_encoder([], self.quantized_cdf.cpu())
        codec = gen_ops.entropy_encode_index(codec, self.indexes.cpu(), x)
        bits = gen_ops.entropy_encode_finalize(codec).numpy()

        
        return bits

    def decompress(self, bits):
        """
        Decompresses a tensor from bit strings. This requires knowledge of the image shape,
        which for arbitrary images sizes needs to be sent as side-information.
        """
        bits = tf.convert_to_tensor(bits, dtype=tf.string)
        codec = gen_ops.create_range_decoder(bits, self.quantized_cdf.cpu())
        codec, x = gen_ops.entropy_decode_index(codec, self.indexes.cpu(), self.indexes.shape, tf.int32)
        # sanity = gen_ops.entropy_decode_finalize(codec)
        x = torch.from_numpy(x.numpy()).reshape(self.prior_shape).to(device).to(torch.float32) + self.offsets
        return x


def decode_and_save(z_t, eps_hat, alpha_t, sigma_t, vae, vae_scale_factor=0.18215):
    # Compute clean latent
    x_latent = (z_t - sigma_t * eps_hat) / alpha_t

    # Decode through VAE
    with torch.no_grad():
        x_pixel = vae.decode(x_latent / vae_scale_factor).sample  # [-1, 1]

    # Convert to image
    x_pixel = (x_pixel.clamp(-1, 1) + 1) / 2  # [0, 1]
    x_pixel = (x_pixel * 255).byte()
    x_pixel = x_pixel[0].permute(1, 2, 0).cpu().numpy()  # first in batch

    alpha_str = f"{alpha_t.flatten()[0].item():.3f}"
    Image.fromarray(x_pixel).save(f"image_{alpha_str}.png")


CHECKPOINT_DIR = 'checkpoints/uqdm-sd'


class Diffusion_SD(torch.nn.Module):
    """
    Progressive Compression with Gaussian Diffusion in LATENT SPACE.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.step = 0
        self.denoised = None

        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = 0.18215

        self.sd_scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.register_buffer('alphas_cumprod', self.sd_scheduler.alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.sd_scheduler.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.sd_scheduler.alphas_cumprod))

        # Score network with frozen UNet backbone
        self.score_net = SD15ScoreNet(self.config)

        # Optimizer and EMA scoped to trainable LoRA + conv_out params only — frozen UNet is never updated
        scale_params = [p for p in self.score_net.parameters() if p.requires_grad]
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.optimizer = torch.optim.Adam(
            scale_params,
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.get('weight_decay', 0.0),
        )
        self.ema = ExponentialMovingAverage(
            scale_params,
            decay=self.config.optim.get('ema_decay', 0.9999),
        )
        self.feat_extractor = InceptionFeatureExtractor(device="cuda")
        self._fid_shard_counters = {}

    def gamma(self, t):
        """Log-SNR at integer SD timestep t in [0, 999]."""
        alpha_bar = self.alphas_cumprod[t]

        return torch.log((1.0 - alpha_bar) / alpha_bar)

    def sigma2(self, t):
        return torch.sigmoid(self.gamma(t))

    def sigma(self, t):
        return torch.sqrt(self.sigma2(t))

    def alpha(self, t):
        return torch.sqrt(torch.sigmoid(-self.gamma(t)))
    def compute_lpips(self, x_hat, x_raw):
        """x_hat, x_raw: [B, 3, H, W] in [0, 255]"""
        x_hat_n = (x_hat.float().to(device) / 127.5) - 1.0  # [0,255] -> [-1,1]
        x_raw_n = (x_raw.float().to(device) / 127.5) - 1.0
        with torch.no_grad():
            return self.lpips_fn(x_hat_n, x_raw_n).squeeze(-1).squeeze(-1).squeeze(-1).cpu()  # [B]
    
    def q_t(self, x_latent, t=1):
        # q(z_t | x_latent) = N(alpha_t * x_latent, sigma^2_t)
        # Now x_latent is in latent space [B, 4, H/8, W/8]
        return Normal(loc=self.alpha(t) * x_latent, scale=self.sigma(t))

    def p_1(self):
        # p(z_1) = N(0, 1) - still works in latent space
        return Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    # These remain unchanged
    def p_s_t(self, p_loc, p_scale, t, s):

        # print(p_scale[0][0][0])
        if self.config.model.prior_type == 'logistic':
            base_dist = LogisticDistribution(loc=p_loc, scale=p_scale * np.sqrt(3. / np.pi ** 2))
        elif self.config.model.prior_type in ('gaussian', 'normal'):
            base_dist = NormalDistribution(loc=p_loc, scale=p_scale)
        else:
            try:
                base_dist = getattr(torch.distributions, self.config.model.prior_type)
            except AttributeError:
                raise ValueError(f"Unknown prior type {self.config.model.prior_type}")
        return base_dist

    def q_s_t(self, q_loc, q_scale):
        return NormalDistribution(loc=q_loc, scale=q_scale)

    def relative_entropy_coding(self, q, p, compress_mode=None):
        raise NotImplementedError
    def get_s_t_params(self, z_t, t, s, x=None, clip_denoised=True, cache_denoised=False, deterministic=False):
        """
        Compute the (location, scale) parameters of either q(z_s | z_t, x)
        or the reverse process distribution p(z_s | z_t) = q(z_s | z_t, x=x_hat) for the given z_t and times t, s.

        Inputs:
        -------
        x              - if not None compute the parameters of q(z_t | z, x) instead p(z_s | z_t)
        clip_denoised  - if True, will clip the denoised prediction x_hat(z_t) to [-1, 1];
                         this might be used to draw better samples.
        cache_denoised - keep the denoised prediction in memory for later use
        deterministic  - if True, compute the mean needed for flow-based sampling instead, removing less noise overall
        """
        gamma_t, gamma_s = self.gamma(t), self.gamma(s)
        alpha_t, alpha_s = self.alpha(t), self.alpha(s)
        sigma_t, sigma_s = self.sigma(t), self.sigma(s)
        # expm1 = 1 - alpha_t^2 / alpha_s^2 * sigma_s^2 / sigma_t^2 = sigma_t|s^2 / sigma_t^2
        expm1_term = - torch.special.expm1(gamma_s - gamma_t)


        if x is None:
            if self.config.model.get('learned_prior_scale'):

                eps_hat, pred_scale_factors = self.score_net(z_t, gamma_t)
            else:
                eps_hat,_ = self.score_net(z_t, gamma_t)
            # Compute denoised prediction only if necessary
            if clip_denoised or cache_denoised:
                x = (z_t - sigma_t * eps_hat) / alpha_t  # c.f. VDM eq (30)

            if cache_denoised:
                self.denoised = x

            # Variance of q(z_s | z_t, x)
            scale = sigma_s * torch.sqrt(expm1_term)
            # Additional modifications for p(z_s | z_t)
            if self.config.model.get('base_prior_scale', 'forward_kernel') == 'forward_kernel':
                # use sigma_t|s^2, the variance of q(z_t | z_s) instead
                scale = sigma_t * torch.sqrt(expm1_term)
            if self.config.model.get('learned_prior_scale'):
                scale = scale * pred_scale_factors
        else:
            scale = sigma_s * torch.sqrt(expm1_term)

        # Mean of q(z_s | z_t, x)
        if x is not None:
            if deterministic:

                loc = sigma_s / sigma_t * z_t - (alpha_t * sigma_s / sigma_t - alpha_s) * x
            else:
                loc = alpha_s * ((1 - expm1_term) / alpha_t * z_t + expm1_term * x)
        else:
            if deterministic:

                loc = alpha_s / alpha_t * z_t + (sigma_s - alpha_s / alpha_t * sigma_t) * eps_hat
            else:
                loc = alpha_s / alpha_t * (z_t - sigma_t * expm1_term * eps_hat)

        return loc, scale
    def _get_t_emb(self, t, dev=None) -> torch.Tensor:
        """Shared sinusoidal embedding used by both schedule and delta heads."""
        if dev is None:
            dev = next(self.schedule_head.parameters()).device
        t_flat = torch.as_tensor(t, dtype=torch.float32, device=dev).view(-1)
        dim  = self.config.get('t_emb_dim', 256)
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=dev) / half
        )
        args = t_flat[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
 
    def transmit_q_s_t(self, x_latent, z_t, t, s, compress_mode=None, cache_denoised=False,x_raw=None):
        """Now x_latent is in latent space"""
        p_loc, p_scale = self.get_s_t_params(z_t, t, s, cache_denoised=cache_denoised)
        q_loc, q_scale = self.get_s_t_params(z_t, t, s, x=x_latent)

        p_s_t = self.p_s_t(p_loc, p_scale, t, s)
        q_s_t = self.q_s_t(q_loc, q_scale)
                        # In transmit_q_s_t i=0:

        z_s, rate = self.relative_entropy_coding(q_s_t, p_s_t, compress_mode=compress_mode)

        return z_s, rate

    def transmit_image(self, z_0_latent, x_raw, compress_mode=None):
        """
        z_0_latent: final latent [B, 4, H/8, W/8]
        x_raw: original pixel image [B, 3, H, W] for comparison
        """
        if compress_mode in ['encode', 'decode']:

            p = torch.distributions.Categorical(logits=self.log_probs_x_z0(z_0_latent=z_0_latent))

        if compress_mode == 'decode':

            x_raw = self.entropy_decode(self.compress_bits.pop(0), p)
        elif compress_mode == 'encode':

            self.compress_bits += [self.entropy_encode(x_raw, p)]

        return x_raw


    

    @torch.no_grad()
    def sample(self, init_z=None, shape=None, times=None, deterministic=False,
               clip_samples=False, decode_method='argmax', return_hist=False):
        """
        Perform ancestral / flow-based sampling.

        Inputs:
        -------
        init_z        - latent state [B, C, H, W]
        shape         - if no init_z is given specify the shape of z instead
        times         - (optional) provide a custom (e.g. partial) sequence of steps
        deterministic - use flow-based sampling instead of ancestral sampling
        clip_samples  - clip latents to [-1, 1]
        decode_method - 'argmax' or 'sample'
        return_hist   - if set return full history of latent states
        """
        if init_z is None:
            assert shape is not None
            p_1 = self.p_1()
            z = p_1.sample(shape)
        else:
            z = init_z
        if return_hist:
            samples = [z]
        if times is None:
            times = torch.linspace(1.0, 0.0, self.config.model.n_timesteps + 1, device=device)

        # for i in trange(len(times) - 1, desc="sampling"):
        for i in range(len(times) - 1):

            t, s = times[i], times[i + 1]
            p_loc, p_scale = self.get_s_t_params(z, t, s, clip_denoised=clip_samples, deterministic=deterministic)
            if deterministic:
                z = p_loc
            else:
                z = self.p_s_t(p_loc, p_scale, t, s).sample()
            if return_hist:
                samples.append(z)
        x_raw = self.decode_p_x_z_0(z_0_latent=z, method=decode_method)

        if return_hist:
            return x_raw, samples + [x_raw]
        else:
            return x_raw
    def forward(self, x_raw, z_1=None, recon_method=None, compress_mode=None, seed=None, timestep_path=None):
        x_pixel = 2 * ((x_raw.float() + .5) / self.config.model.vocab_size) - 1
        with torch.no_grad():
            x_latent = self.vae.encode(x_pixel).latent_dist.sample() * self.vae_scale_factor
            x_latent = x_latent.detach()

        rescale_pixel_to_bpd = 1. / (np.prod(x_raw.shape[1:]) * np.log(2.))

        if timestep_path is not None:
            sd_timesteps = torch.tensor(timestep_path, dtype=torch.long)
        else:
            sd_timesteps = self.sd_scheduler.timesteps
        total_steps = len(sd_timesteps) - 1

        # ── 1. PRIOR LOSS ────────────────────────────────────────────────────────
        if z_1 is None and not torch.is_inference_mode_enabled():
            q_1        = self.q_t(x_latent, t=sd_timesteps[0].item())
            p_1        = self.p_1()
            with local_seed(seed, i=0):
                z_1    = q_1.sample()
            loss_prior = kl_divergence(q_1, p_1).sum(dim=[1, 2, 3])
        else:
            if z_1 is None:
                with local_seed(seed, i=0):
                    z_1 = self.p_1().sample(x_latent.shape)
            loss_prior = torch.zeros(x_latent.shape[0], device=device)

        # ── 2. DIFFUSION LOOP ────────────────────────────────────────────────────
        z_s       = z_1
        rate_s    = loss_prior
        loss_diff = 0.
        metrics   = []

        for i in range(total_steps):
            z_t    = z_s
            rate_t = rate_s
            ts_t   = sd_timesteps[i].item()
            ts_s   = sd_timesteps[i + 1].item()

            with local_seed(seed, i=i + 1):
                z_s, rate_s = self.transmit_q_s_t(
                    x_latent, z_t, ts_t, ts_s,
                    compress_mode=compress_mode,
                    cache_denoised=(recon_method == 'denoise'),
                    x_raw=x_raw,
                )
            loss_diff += rate_s

            # collect progressive reconstructions during decompress
            if recon_method is not None and torch.is_inference_mode_enabled():
                # Truncate schedule to steps at or below ts_start, then prepend ts_start
                truncated = [t for t in DDIM_SCHEDULE if t <= ts_s]
                denoise_steps = torch.tensor(
                    [ts_s] + truncated if (not truncated or truncated[0] != ts_s) else truncated,
                    dtype=torch.long, device=z_s.device
                )


                x_hat_t = self.denoise_z_t(z_s, recon_method, times=denoise_steps)

                metrics += [{
                    'prog_bpds':   (loss_diff * rescale_pixel_to_bpd / np.log(2)).cpu(),
                    'prog_x_hats': x_hat_t.detach().cpu(),
                    'prog_mses':   torch.mean((x_hat_t - x_raw).float()**2, dim=[1,2,3]).cpu(),
                    'prog_lpips':  self.compute_lpips(x_hat_t, x_raw),
                }]

        z_0_latent = z_s

        # ── 3. FINAL RECONSTRUCTION ──────────────────────────────────────────────
        log_probs  = self.log_probs_x_z0(z_0_latent=z_0_latent, x_raw=x_raw)
        loss_recon = -log_probs.sum(dim=[1, 2, 3])

        x_raw = self.transmit_image(z_0_latent, x_raw, compress_mode=compress_mode)

        if recon_method is not None:
            
            metrics += [{
                'prog_bpds':   (loss_recon * rescale_pixel_to_bpd / np.log(2)).cpu(),
                'prog_x_hats': x_raw.cpu(),
                'prog_mses':   torch.zeros(x_raw.shape[0]),
                'prog_lpips':  torch.zeros(x_raw.shape[0]),
            }]
            metrics = default_collate(metrics)
        else:
            metrics = {}

        # ── 4. AGGREGATE ─────────────────────────────────────────────────────────
        bpd_latent = loss_prior.mean() * rescale_pixel_to_bpd
        bpd_diff   = loss_diff.mean()  * rescale_pixel_to_bpd
        bpd_recon  = loss_recon.mean() * rescale_pixel_to_bpd
        loss       = bpd_latent + bpd_diff + bpd_recon

        metrics.update({
            'bpd':        loss,
            'bpd_latent': bpd_latent,
            'bpd_recon':  bpd_recon,
            'bpd_diff':   bpd_diff,
        })
        return loss, metrics

    def entropy_encode(self, k, p):
        """
        Encode integer array k to bits using a prior / coding distribution p.
        We might want to quantize scale for determinism and added stability across multiple machines.
        """
        # When using a scalar prior it would be better to quantize u as in tfc.UniversalBatchedEntropyModel
        # assert self.config.model.learned_prior_scale

        em = EntropyModel(p)

        bitstring = em.compress(k)

        return bitstring

    def entropy_decode(self, bits, p):
        """
        Decode integer array from bits using the prior p.
        """
        # assert self.config.model.learned_prior_scale
        em = EntropyModel(p)
        k = em.decompress(bits)
        return k


    @torch.inference_mode()
    def compress(self, image, timestep_path=None):
        self.compress_bits = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        self.forward(image.to(device), compress_mode='encode', seed=0, timestep_path=timestep_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compress_time = time.time() - start_time

        return self.compress_bits, compress_time

    @torch.inference_mode()
    def decompress(self, bits, image_shape, recon_method='denoise', timestep_path=None):
        # consume the bits for each step, return the intermediate reconstructions for each step
        self.compress_bits = bits.copy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        _, metrics = self.forward(torch.zeros(image_shape, device=device), compress_mode='decode',
                                recon_method=recon_method, seed=0, timestep_path=timestep_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decompress_time = time.time() - start_time

        return metrics['prog_x_hats'],decompress_time

    def extract_and_save_features(
        self,
        x_hat: torch.Tensor,   # [B, 3, H, W] in [0, 255]
        x_raw: torch.Tensor,   # [B, 3, H, W] in [0, 255]
        timestep: int,
        save_dir: str,
    ):
        """Drop-in call inside the forward() diffusion loop."""
        for split, imgs in [("real", x_raw), ("fake", x_hat)]:
            key = (split, timestep)
            shard_id = self._fid_shard_counters.get(key, 0)

            feats = self.feat_extractor.extract(imgs)  # [B, 2048]
            save_features(feats, save_dir, split=split, timestep=timestep, shard_id=shard_id)

            self._fid_shard_counters[key] = shard_id + 1
    def log_probs_x_z0(self, z_0_latent, x_raw=None):
        """
        Decode z_0_latent and compute pixel probabilities using a simple
        discretized Gaussian in pixel space.
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            z_0_pixel = self.vae.decode(z_0_latent / self.vae_scale_factor).sample
            # z_0_pixel is in [-1, 1], convert to [0, vocab_size-1]
            z_0_pixel = ((z_0_pixel + 1) / 2 * self.config.model.vocab_size).clamp(0, self.config.model.vocab_size - 1)

        x_vals = torch.arange(self.config.model.vocab_size, device=z_0_pixel.device).float()
        x_vals = x_vals.reshape([1] * z_0_pixel.ndim + [-1])

        # simple discretized Gaussian centered at VAE output
        sigma = 1.0  # pixel-space noise std; tune this
        z = z_0_pixel.unsqueeze(-1)
        logits = -0.5 * ((z - x_vals) / sigma) ** 2

        logprobs = torch.log_softmax(logits, dim=-1)

        if x_raw is None:
            return logprobs
        else:
            log_probs = torch.gather(
                logprobs, dim=-1, index=x_raw.long().unsqueeze(-1)
            ).squeeze(-1)
            return log_probs

    def decode_p_x_z_0(self, z_0_latent, method='argmax'):
        """Decode latent to pixels via VAE decoder."""
        with torch.no_grad():
            z_0_pixel = self.vae.decode(z_0_latent / self.vae_scale_factor).sample
        # z_0_pixel is in [-1, 1]; convert to [0, vocab_size-1] uint8-range integers
        x_raw = ((z_0_pixel + 1) / 2 * self.config.model.vocab_size).clamp(0, self.config.model.vocab_size - 1).round().long()
        return x_raw

    def denoise_z_t(self, z_t, recon_method, times=None):
        """z_t is in latent space"""
        if recon_method == 'ancestral':
            x_hat_t = self.sample(
                times=times, init_z=z_t,
                clip_samples=False, decode_method='argmax', return_hist=False
            )
        elif recon_method == 'flow_based':

            x_hat_t = self.sample(
                times=times, init_z=z_t, deterministic=True,
                clip_samples=False, decode_method='argmax', return_hist=False
            )
        elif recon_method == 'denoise':
            assert self.denoised is not None
            x_hat_t = self.decode_p_x_z_0(z_0_latent=self.denoised, method='argmax')
            self.denoised = None
        else:
            raise ValueError(f"Unknown progressive reconstruction method {recon_method}")

        return x_hat_t

    @staticmethod
    def get_noise_schedule(config):
        # gamma is the negative log-snr as in VDM eq (3)

        gamma_min, gamma_max, schedule = [getattr(config.model, k) for k in
                                          ['gamma_min', 'gamma_max', 'noise_schedule']]
        assert gamma_max > gamma_min, "SNR should be decreasing in time"
        if schedule == "fixed_linear":
            gamma = Diffusion_SD.FixedLinearSchedule(gamma_min, gamma_max)
        elif schedule == "learned_linear":
            gamma = Diffusion_SD.LearnedLinearSchedule(gamma_min, gamma_max, config.model.get('fix_gamma_max'))
        else:
            raise ValueError('Unknown noise schedule %s' % schedule)
        return gamma

    class FixedLinearSchedule(torch.nn.Module):
        def __init__(self, gamma_min, gamma_max):
            super().__init__()
            self.gamma_min = gamma_min
            self.gamma_max = gamma_max

        def forward(self, t):
            return self.gamma_min + (self.gamma_max - self.gamma_min) * t

    class LearnedLinearSchedule(torch.nn.Module):
        def __init__(self, gamma_min, gamma_max, fix_gamma_max=False):
            super().__init__()
            self.fix_gamma_max = fix_gamma_max
            if fix_gamma_max:
                self.gamma_max = torch.tensor(gamma_max)
            else:
                self.b = torch.nn.Parameter(torch.tensor(gamma_min))
            self.w = torch.nn.Parameter(torch.tensor(gamma_max - gamma_min))

        def forward(self, t):
            w = self.w.abs()
            if self.fix_gamma_max:
                return w * (t - 1.) + self.gamma_max
            else:
                return self.b + w * t

    def save(self):
        checkpoint = {
            'step': self.step,
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'scale_head': self.score_net.scale_head.state_dict()
        }
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f'ckpt_{self.step:07d}.pt')
        torch.save(checkpoint, path)
        print(f'Saved checkpoint → {path}')


    def load(self, ckpt_path=None):
        self.score_net = SD15ScoreNet(self.config)


        trainable_params = (
            list(self.score_net.scale_head.parameters())
        )
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.get('weight_decay', 0.0),
        )
        self.ema = ExponentialMovingAverage(
            trainable_params,
            decay=self.config.optim.get('ema_decay', 0.9999),
        )

        if ckpt_path is not None:
            cp = torch.load(ckpt_path, map_location=device, weights_only=False)

            self.score_net.scale_head.load_state_dict(cp['scale_head'])
            if 'optimizer' in cp:
                self.optimizer.load_state_dict(cp['optimizer'])
            if 'ema' in cp:
                self.ema.load_state_dict(cp['ema'])
            if 'step' in cp:
                self.step = cp['step']
            print(f'Loaded checkpoint from {ckpt_path}')
        else:
            print('No checkpoint provided — starting fresh.')
    def trainer(self, train_iter, eval_iter=None, timestep_path=None):
        """
        Train UQDM-style SD1.5 model for a specified number of steps on a train set.
        Hyperparameters are set via self.config.training, self.config.eval, and self.config.optim.
        Only LoRA adapter weights and the widened conv_out (eps/scale split head)
        are trained; the frozen SD1.5 base weights are never touched.
        """
        # Pull the trainable set directly off the optimizer's own param groups,
        # rather than recomputing it separately — guarantees this always matches
        # whatever __init__/load() actually configured the optimizer with, even
        # if that set changes later (e.g. two-param-group LR split).
        trainable_params = [
            p for group in self.optimizer.param_groups for p in group['params']
        ]

        if self.step >= self.config.training.n_steps:
            print('Skipping training, increase training.n_steps if more steps are desired.')

        from tqdm import tqdm

        pbar = tqdm(total=self.config.training.n_steps, initial=self.step, desc='Training')
        while self.step < self.config.training.n_steps:

            # ── Parameter update ──────────────────────────────────────────────────
            batch = next(train_iter).to(device)
            self.optimizer.zero_grad()
            self.score_net.unet.train()  # enables dropout/groupnorm train-mode for LoRA path
            loss, metrics = self(batch, timestep_path=timestep_path)
            loss.backward()
            if self.config.optim.warmup > 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config.optim.lr * np.minimum(
                        self.step / self.config.optim.warmup, 1.0
                    )
            if self.config.optim.grad_clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=self.config.optim.grad_clip_norm
                )
            self.optimizer.step()

            self.step += 1
            self.ema.update(trainable_params)

            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

            last = self.step == self.config.training.n_steps

            # ── Print loss every 100 steps ──────────────────────────────────────────
            if self.step % 100 == 0 or last:
                print(f"[step {self.step}] loss: {loss.item():.6f}")

            if self.step % self.config.training.log_metrics_every_steps == 0 or last:
                self.save()
                print(metrics)
            # ── Checkpoint + train metrics ────────────────────────────────────────


            # ── Validation metrics ────────────────────────────────────────────────
            if eval_iter is not None and (
                self.step % self.config.training.eval_every_steps == 0 or last
            ):
                n_batches = self.config.training.eval_steps_to_run
                res = []
                for batch in tqdm(
                    islice(eval_iter, n_batches),
                    total=n_batches or len(eval_iter),
                    desc='Evaluating on test set',
                    leave=False,
                ):
                    batch = batch.to(device)
                    with torch.inference_mode():
                        self.ema.store(trainable_params)
                        self.ema.copy_to(trainable_params)
                        self.score_net.unet.eval()  # disables dropout in resnet blocks for stable eval
                        _, ths_metrics = self(batch)
                        self.ema.restore(trainable_params)
                    res += [ths_metrics]
                res = default_collate(res)
                print({k: v.mean().item() for k, v in res.items()})
        pbar.close()

    @staticmethod
    def mse_to_psnr(mse, max_val):
        with np.errstate(divide='ignore'):
            return -10 * (np.log10(mse) - 2 * np.log10(max_val))

    @torch.inference_mode()
    def evaluate(self, eval_iter, n_batches=None, seed=None, timestep_path=None):
        """
        Evaluate rate-distortion on the test set.

        Inputs:
        -------
        n_batches     - (optionally) give a number of batches to evaluate
        timestep_path - (optionally) specify a custom diffusion path,
                        e.g. [999, 138, 0]. Default (None) uses all consecutive timesteps.
        """

        res = []
        for X in tqdm(islice(eval_iter, n_batches), total=n_batches or len(eval_iter), desc='Evaluating UQDM_SD'):
            print('Evaluating batch %s...' % len(res))
            X = X.to(device)
            ths_res = {}

            recon_method = 'denoise'

            loss, metrics = self(X, recon_method=recon_method, seed=seed, timestep_path=timestep_path)
            print(f"bpd_diff (internal): {metrics['bpd_diff'].item():.4f}")
            print(f"bpd_latent:      []    {metrics['bpd_latent'].item():.4f}")
            print(f"bpd_recon:           {metrics['bpd_recon'].item():.4f}")
            print(f"bpps[-1] before *3:  {np.cumsum(metrics['prog_bpds'].mean(dim=1))[-1]:.4f}")
            print(f"bpps[-1] after *3:   {3*np.cumsum(metrics['prog_bpds'].mean(dim=1))[-1]:.4f}")

            bpds = np.cumsum(metrics['prog_bpds'].mean(dim=1))

            psnrs = self.mse_to_psnr(metrics['prog_mses'].mean(dim=1), max_val=255.)
            lpips_ = metrics['prog_lpips'].mean(dim=1).numpy()
            ths_res[recon_method] = dict(bpds=bpds, psnrs=psnrs, lpips=lpips_)
            res += [ths_res]
        res = default_collate(res)

        for recon_method in res.keys():
            bpps = np.round(3 * res[recon_method]['bpds'].mean(axis=0).numpy(), 4)
            psnrs = np.round(res[recon_method]['psnrs'].mean(axis=0).numpy(), 4)
            lpips_ = np.round(    res[recon_method]['lpips'].mean(axis=0).numpy(),  4)
            print('Reconstructions via: %s\nbpps:  %s\npsnrs: %s\nlpips: %s\n' % (recon_method, bpps, psnrs, lpips_))
        return bpps, psnrs, lpips_


class UQDM_SD(Diffusion_SD):
    """
    Making Progressive Compression tractable with Universal Quantization.
    """

    def __init__(self, config):
        """
        See Diffusion_SD.__init__ for hyperparameters.
        """
        super().__init__(config)
        self.compress_bits = None

    def p_s_t(self, p_loc, p_scale, t, s):
        # p(z_s | z_t) is a convolution of g_t and U(+- d_t), d_t = sqrt(12) * sigma_s * sqrt(exmp1term)
        delta_t = self.sigma(s) * torch.sqrt(- 12 * torch.special.expm1(self.gamma(s) - self.gamma(t)))
        base_dist = super().p_s_t(p_loc, p_scale, t, s)
        return UniformNoisyDistribution(base_dist, delta_t)

    def q_s_t(self, q_loc, q_scale):
        # q(z_s | z_t, x) = U(q_loc +- sqrt(3) * q_scale)
        return Uniform(low=q_loc - np.sqrt(3) * q_scale, high=q_loc + np.sqrt(3) * q_scale)

    def relative_entropy_coding(self, q, p, compress_mode=None):
        # Transmit sample z_s ~ q(z_s | z_t, x)
        if not torch.is_inference_mode_enabled():
            z_s = q.sample()
        else:
            # Apply universal quantization
            # shared U(-0.5, 0.5), seeds have already been set in self.forward
            u = torch.rand(q.mean.shape, device=q.mean.device) - 0.5

            if compress_mode in ['encode', 'decode']:
                p_discrete = p.discretize(u)
            if compress_mode == 'decode':
                # consume bits
                quantized = self.entropy_decode(self.compress_bits.pop(0), p_discrete)
            else:
                # Add dither U(-delta/2, delta/2)
                # Transmit residual q - p for greater numerical stability
                quantized = torch.round((q.mean - p.mean + p.delta * u) / p.delta)
                if compress_mode == 'encode':
                    # accumulate bits
                    self.compress_bits += [self.entropy_encode(quantized, p_discrete)]
            # Subtract the same (pseudo-random) dither using shared randomness
            z_s = quantized * p.delta + p.mean - p.delta * u

        # Evaluate z_s under log (posterior/prior) to get MC estimate of KL.
        rate = - p.log_prob(z_s) - torch.log(p.delta)
        rate = torch.sum(rate, dim=[1, 2, 3])

        return z_s, rate

    def _compute_rate_expected(self, q, p, n_samples=10):
        rates = torch.stack([
            torch.sum(-p.log_prob(q.sample()), dim=[1, 2, 3])
            for _ in range(n_samples)
        ])
        return rates.mean(0)

    @torch.no_grad()
    def compute_cost_matrix_uqdm(self, data_iter, seed=None, timestep_stride=100,
                                num_images=1, cache_path=None):
        """
        Build a BPP cost matrix for UQDM using relative entropy coding rate.
        Entry [i, j] = mean coding cost (bpp) of transmitting z_{ts[j]} from z_{ts[i]}.
        Caches eps and scale at each timestep, then manually computes q and p params.
        """
        # --- Load from cache if available ---
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached UQDM cost matrix from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            cost_matrix = data['cost_matrix']
            timesteps = data['timesteps'].tolist()
            print(f"Loaded cost matrix shape {cost_matrix.shape}")
            return cost_matrix, timesteps

        # --- Select timesteps ---
        all_ts = self.sd_scheduler.timesteps
        timesteps = all_ts[::timestep_stride].tolist()
        if all_ts[-1].item() not in timesteps:
            timesteps.append(all_ts[-1].item())
        T = len(timesteps)

        print(f"Computing UQDM BPP matrix for {T} timesteps (stride={timestep_stride})...")
        print(f"Total transitions: {T * (T - 1) // 2}, averaging over {num_images} images")

        cost_matrix_accum = np.zeros((T, T))

        start_time = time.time()

        for img_idx in range(num_images):
            x_raw = next(data_iter).to(device)
            print(f"\n[Image {img_idx + 1}/{num_images}] Encoding to latent space...")

            x_pixel = 2 * ((x_raw.float() + .5) / self.config.model.vocab_size) - 1
            x_latent = self.vae.encode(x_pixel).latent_dist.sample() * self.vae_scale_factor
            x_latent = x_latent.detach()
            print(x_raw.shape, x_latent.shape)
            rescale_pixel_to_bpd = 1. / (np.prod(x_raw.shape[1:]) * np.log(2.))

            # --- Single shared noise for consistent z_t ---
            with local_seed(seed, i=img_idx * 10_000):
                base_eps = torch.randn_like(x_latent)

            # --- Cache z_t, eps_hat, pred_scale_factors for all timesteps ---
            z_cache     = {}
            eps_cache   = {}
            scale_cache = {}
            psnr_accum = np.zeros(T)
            print(f"  Caching z_t and score predictions for {T} timesteps...")
            for idx, ts_val in enumerate(timesteps):
                ts_tensor = torch.tensor(ts_val, device=x_latent.device)
                alpha_t   = self.alpha(ts_tensor)
                sigma_t   = self.sigma(ts_tensor)

                z_t = alpha_t * x_latent + sigma_t * base_eps
                z_cache[ts_val] = z_t.detach()
                gamma_t = self.gamma(ts_tensor)

                if self.config.model.get('learned_prior_scale'):
                    eps_hat, pred_scale_factors = self.score_net(z_t, gamma_t)
                    scale_cache[ts_val] = pred_scale_factors.detach()
                else:
                    eps_hat, _ = self.score_net(z_t, gamma_t)
                    scale_cache[ts_val] = None
                eps_cache[ts_val] = eps_hat.detach()
                x_hat = (z_t - sigma_t * eps_hat) / alpha_t
                x_hat = x_hat.clamp(-4.0, 4.0)
                psnr_accum[idx] += compute_psnr(x_hat, x_latent)

            # --- Fill cost matrix ---
            cost_matrix_img = np.zeros((T, T))
            for i, ts_t in enumerate(timesteps):
                z_t     = z_cache[ts_t]
                eps_t   = eps_cache[ts_t]
                scale_t = scale_cache[ts_t]
                print(i)
                for j in range(i + 1, T):
                    ts_s = timesteps[j]

                    p_loc, p_scale, q_loc, q_scale = self._get_params_from_cache(
                        z_t, ts_t, ts_s, x_latent, eps_t, scale_t
                    )

                    p = self.p_s_t(p_loc, p_scale, ts_t, ts_s)
                    q = self.q_s_t(q_loc, q_scale)

                    rate = self._compute_rate_expected(q, p)
                    cost_matrix_img[i, j] = rate.mean().item() * rescale_pixel_to_bpd * 3
            cost_matrix_accum += cost_matrix_img

        # --- Final average ---
        cost_matrix = cost_matrix_accum / num_images
        psnr_per_timestep = psnr_accum / num_images
        return cost_matrix, timesteps, psnr_per_timestep

    def _get_params_from_cache(self, z_t, ts_t, ts_s, x_latent, eps_hat, pred_scale_factors):
        t_tensor   = torch.tensor(ts_t, device=z_t.device)
        s_tensor   = torch.tensor(ts_s, device=z_t.device)
        gamma_t    = self.gamma(t_tensor)
        gamma_s    = self.gamma(s_tensor)
        alpha_t    = self.alpha(t_tensor)
        alpha_s    = self.alpha(s_tensor)
        sigma_t    = self.sigma(t_tensor)
        sigma_s    = self.sigma(s_tensor)
        expm1_term = (-torch.special.expm1(gamma_s - gamma_t))

        # ── Prior ──────────────────────────────────────────────────────────────
        p_scale = sigma_s * torch.sqrt(expm1_term)
        if self.config.model.get('base_prior_scale', 'forward_kernel') == 'forward_kernel':
            p_scale = sigma_t * torch.sqrt(expm1_term)
        if self.config.model.get('learned_prior_scale') and pred_scale_factors is not None:
            p_scale = p_scale * pred_scale_factors
        p_loc = alpha_s / alpha_t * (z_t - sigma_t * expm1_term * eps_hat)

        # ── Posterior: use TRUE x_latent directly, matching get_s_t_params ───
        q_scale = sigma_s * torch.sqrt(expm1_term)
        q_loc   = alpha_s * ((1 - expm1_term) / alpha_t * z_t + expm1_term * x_latent)

        return p_loc, p_scale, q_loc, q_scale


def compute_psnr(pred, target, max_val=None):
    """
    Compute PSNR between pred and target tensors/arrays.

    Inputs:
    -------
    pred, target - tensors or numpy arrays
    max_val      - maximum pixel/value range. If None, uses target's max abs value
                   (useful for latents); pass 255.0 for uint8 images.
    """
    if hasattr(pred, 'cpu'):
        pred = pred.detach().cpu().numpy()
    if hasattr(target, 'cpu'):
        target = target.detach().cpu().numpy()
    if max_val is None:
        max_val = np.abs(target).max()

    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def find_optimal_path_dp(bpp_matrix, timesteps):
    """
    Find optimal transmission path via DP using raw KL-based bpp.

    bpp_matrix: [T, T] upper-triangular numpy array of raw KL bpp costs
                output of compute_cost_matrix()
    timesteps:  list of SD integer timesteps corresponding to matrix indices

    Returns:
        optimal_path_indices: list of matrix indices in the optimal path
        timestep_path:        list of SD timesteps in the optimal path
        bpp_breakdown:        dict of cost diagnostics
    """
    T = len(bpp_matrix)
    INF = float('inf')

    # dp[j] = (min_cost_to_reach_j, predecessor_index)
    dp = [(INF, -1)] * T
    dp[0] = (0.0, -1)

    start_time = time.time()
    for j in range(1, T):
        for i in range(j):
            raw_bpp = bpp_matrix[i, j]
            if raw_bpp <= 0:
                print(f"  Warning: non-positive bpp at [{i},{j}] = {raw_bpp:.6f}, skipping")
                continue
            cost = dp[i][0] + raw_bpp
            if cost < dp[j][0]:
                dp[j] = (cost, i)

    optimal_total_bpp = dp[T - 1][0]

    # Traceback
    optimal_path = []
    current = T - 1
    while current != -1:
        optimal_path.append(current)
        current = dp[current][1]
    optimal_path.reverse()

    timestep_path = [timesteps[i] for i in optimal_path]

    # Consecutive baseline: sum of every adjacent step
    consecutive_bpp = sum(bpp_matrix[i, i + 1] for i in range(T - 1))

    bpp_breakdown = {
        'optimal_total_bpp':   optimal_total_bpp,
        'consecutive_bpp':     consecutive_bpp,
        'num_transitions':     len(optimal_path) - 1,
        'saving_bpp':          consecutive_bpp - optimal_total_bpp,
    }

    print(f"DP done in {time.time() - start_time:.3f}s")
    print(f"  Optimal path bpp    : {optimal_total_bpp:.6f}  ({len(optimal_path)-1} transitions)")
    print(f"  Consecutive bpp     : {consecutive_bpp:.6f}  ({T-1} transitions)")
    print(f"  Saving              : {consecutive_bpp - optimal_total_bpp:.6f} bpp")
    print(f"  Timestep path       : {timestep_path}")

    return optimal_path, timestep_path, bpp_breakdown


# ---------------------------------------------------------------------------
# FID evaluation (clean-fid based)
# ---------------------------------------------------------------------------

# pip install clean-fid



def _get_feat_model(device):
    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)
    return feat_model


@torch.inference_mode()
def _img_to_acts(feat_model, img_uint8_bchw: torch.Tensor, device) -> np.ndarray:
    import torch.nn.functional as F
    x = img_uint8_bchw.float().to(device) / 255.0
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    x = x * 2.0 - 1.0
    acts = feat_model(x).cpu().numpy()
    return acts


def _compute_stats(all_acts: list) -> tuple:
    acts = np.concatenate(all_acts, axis=0)   # (N, 2048)
    print(f"  Computing stats from {acts.shape[0]} images, activation dim {acts.shape[1]}")
    mu = np.mean(acts, axis=0)                # (2048,)
    if acts.shape[0] == 1:
        sigma = np.zeros((acts.shape[1], acts.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(acts, rowvar=False)    # (2048, 2048)
        # Regularize to avoid singular matrix with small sample sizes
        sigma += np.eye(sigma.shape[0]) * 1e-6
    return mu, sigma


def compute_real_stats(eval_iter_fn, feat_model, device, cache_path: str):
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached real stats from {cache}")
        d = np.load(cache)
        return d["mu"], d["sigma"]

    print("Computing real image stats (one-time)...")
    all_acts = []

    for batch in tqdm(eval_iter_fn()):
        img = batch[0] if isinstance(batch, (tuple, list)) else batch
        if img.ndim == 3:
            img = img.unsqueeze(0)
        for i in range(img.shape[0]):
            single = img[i].unsqueeze(0)
            acts = _img_to_acts(feat_model, single, device)
            all_acts.append(acts)

    mu, sigma = _compute_stats(all_acts)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma)
    print(f"Saved real stats → {cache_path}")
    return mu, sigma


def run_stage(model, eval_iter_fn, feat_model, timestep_path, device):
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    n_steps = len(timestep_path)
    n_recons = n_steps + 1  # diffusion steps + lossless pixel step

    all_acts_per_step   = [[] for _ in range(n_recons)]
    total_bpp_per_step  = [0.0] * n_recons
    total_psnr_per_step = [0.0] * n_recons
    total_ssim_per_step = [0.0] * n_recons
    n = 0

    for batch in tqdm(eval_iter_fn()):
        img = batch[0] if isinstance(batch, (tuple, list)) else batch
        if img.ndim == 3:
            img = img.unsqueeze(0)
        image = img.to(device)

        with torch.inference_mode():
            compressed = model.compress(image, timestep_path=timestep_path)

        bits = [len(b) * 8 for b in compressed]
        cumulative_bits = np.cumsum(bits)
        n_pixels = np.prod(image.shape[-2:])
        bpp_per_step = cumulative_bits / n_pixels * 3

        for step_idx in range(n_recons):
            src_idx = min(step_idx, len(bpp_per_step) - 1)
            total_bpp_per_step[step_idx] += bpp_per_step[src_idx]

        with torch.inference_mode():
            reconstructions = model.decompress(
                compressed, image.shape,
                recon_method="denoise",
                timestep_path=timestep_path,
            )

        img_f = image.float()
        for step_idx, x_hat in enumerate(reconstructions):
            x_hat_f  = x_hat.float().clamp(0, 255).to(device)
            x_hat_u8 = x_hat_f.byte().cpu()

            psnr_val = psnr(x_hat_f, img_f, data_range=255.0).item()
            total_psnr_per_step[step_idx] += psnr_val
            total_ssim_per_step[step_idx] += ssim(x_hat_f, img_f, data_range=255.0).item()

            acts = _img_to_acts(feat_model, x_hat_u8, device)
            all_acts_per_step[step_idx].append(acts)

        n += 1

    mean_bpp_per_step  = [total_bpp_per_step[i]  / max(n, 1) for i in range(n_recons)]
    mean_psnr_per_step = [total_psnr_per_step[i] / max(n, 1) for i in range(n_recons)]
    mean_ssim_per_step = [total_ssim_per_step[i] / max(n, 1) for i in range(n_recons)]
    stats_per_step     = [_compute_stats(all_acts_per_step[i]) for i in range(n_recons)]

    # extend timesteps to label the extra lossless pixel step (repeats final t)
    actual_timesteps    = list(timestep_path)
    extended_timesteps  = actual_timesteps + [actual_timesteps[-1]]

    return stats_per_step, mean_bpp_per_step, mean_psnr_per_step, mean_ssim_per_step, n, extended_timesteps


def run_compression_sweep(model, eval_iter_fn, timestep_paths: dict,
                           out_root="/tmp/fid_sweep"):
    device = next(model.parameters()).device
    n_eval = len(eval_iter_fn().dataset)
    print(f"Total eval images: {n_eval}")
    if n_eval < 2000:
        print(f"WARNING: only {n_eval} eval images — FID will run but is not meaningful.")

    feat_model = _get_feat_model(device)

    real_stats_path = Path(out_root) / "real_stats.npz"
    if n_eval < 2000 and real_stats_path.exists():
        print("Small dataset — deleting cached real stats.")
        real_stats_path.unlink()

    mu_real, sigma_real = compute_real_stats(
        eval_iter_fn, feat_model, device, real_stats_path
    )

    results = {}
    checkpoint_path = Path(out_root) / "results_checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        print(f"Resumed from checkpoint: {list(results.keys())} already done")

    for label, timestep_path in timestep_paths.items():
        if label in results:
            print(f"Skipping {label} (already in checkpoint)")
            continue

        print(f"\n=== {label} ===")
        stats_per_step, mean_bpp_per_step, mean_psnr_per_step, mean_ssim_per_step, n, extended_timesteps = run_stage(
            model, eval_iter_fn, feat_model, timestep_path, device
        )

        step_results = []
        for step_idx, (mu_recon, sigma_recon) in enumerate(stats_per_step):
            t        = extended_timesteps[step_idx]
            bpp      = mean_bpp_per_step[step_idx]
            psnr_val = mean_psnr_per_step[step_idx]
            ssim_val = mean_ssim_per_step[step_idx]

            is_lossless = not np.isfinite(psnr_val)
            if is_lossless:
                print(f"  t={t:4d} [lossless]  bpp={bpp:.4f}  PSNR=inf  SSIM={ssim_val:.4f}")
            else:
                fid = fid_from_stats(mu_real, sigma_real, mu_recon, sigma_recon, n)
                step_results.append({
                    "timestep": int(t),
                    "bpp":      bpp,
                    "fid":      fid,
                    "psnr":     psnr_val,
                    "ssim":     ssim_val,
                    "lossless": False,
                })
                print(f"  t={t:4d}  bpp={bpp:.4f}  FID={fid:.2f}  "
                      f"PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

            # always record lossless step separately so it appears in output
            if is_lossless:
                step_results.append({
                    "timestep": int(t),
                    "bpp":      bpp,
                    "fid":      None,   # FID undefined for lossless
                    "psnr":     float("inf"),
                    "ssim":     float(ssim_val),
                    "lossless": True,
                })

        if not step_results:
            print(f"  WARNING: all steps skipped for {label}")
            continue

        # summary uses last non-lossless step for fid/psnr, but last step for bpp
        lossy_results = [s for s in step_results if not s["lossless"]]
        final = lossy_results[-1] if lossy_results else step_results[-1]

        results[label] = {
            "bpp":          step_results[-1]["bpp"],   # full rate including lossless
            "fid":          final["fid"],
            "psnr":         final["psnr"],
            "ssim":         final["ssim"],
            "fid_per_step": step_results,
        }
        print(f"  final: bpp={step_results[-1]['bpp']:.4f}  FID={final['fid']:.2f}")
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def fid_from_stats(mu_real, sigma_real, mu_recon, sigma_recon, n_images) -> float:
    if n_images < 2000:
        print(f"  WARNING: FID computed from only {n_images} images — "
              f"not statistically meaningful, debugging only")
    print(f"  mu_real: {mu_real.shape}, sigma_real: {sigma_real.shape}")
    print(f"  mu_recon: {mu_recon.shape}, sigma_recon: {sigma_recon.shape}")
    return float(frechet_distance(mu_real, sigma_real, mu_recon, sigma_recon))


def plot_fid_vs_rate(fid_results: dict, baselines: dict = None, plot_per_step: bool = True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_yscale("log")

    if plot_per_step:
        for label, info in fid_results.items():
            if "fid_per_step" in info:
                steps = info["fid_per_step"]
                bpps  = [s["bpp"] for s in steps]
                fids  = [s["fid"] for s in steps]
                ax.plot(bpps, fids, "o-", label=label)
        ax.set_xlabel("Rate (bpp)", fontsize=13)
    else:
        labels = list(fid_results.keys())
        bpps   = [fid_results[l]["bpp"] for l in labels]
        fids   = [fid_results[l]["fid"] for l in labels]
        order  = np.argsort(bpps)
        ax.plot(np.array(bpps)[order], np.array(fids)[order],
                "o-", color="orange", label="UQDM", linewidth=2)
        ax.set_xlabel("Rate (bpp)", fontsize=13)

    ax.set_ylabel("FID", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fid_vs_rate.png", dpi=150)
    plt.show()
def resize_and_crop(image, size=512):
    """
    Resize shortest edge to size, then center crop to size x size.
    image: [B, C, H, W] in [0, 255] uint8
    """
    B = image.shape[0]
    result = []
    for b in range(B):
        img = image[b]  # [C, H, W]
        # Resize shortest edge to 512
        h, w = img.shape[-2], img.shape[-1]
        scale = size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = TF.resize(img, [new_h, new_w], antialias=True)
        # Center crop to 512x512
        img = TF.center_crop(img, [size, size])
        result.append(img)
    return torch.stack(result) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--config_path", default="checkpoints/uqdm-small")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default="reconstructions")
    parser.add_argument("--eval_steps", type=int, default=999)
    parser.add_argument("--recon_method",
                        choices=["ancestral", "flow_based", "denoise"],
                        default="ancestral")
    parser.add_argument("--data", default="kodak/")
    parser.add_argument("--fid_data", default=None,
                    help="Path to separate dataset for FID (e.g. COCO val). "
                         "If None, FID is skipped.")
    parser.add_argument("--fid_save_dir", default="fid_features")
    parser.add_argument("--fid_num_images", type=int, default=5000)
    args = parser.parse_args()


    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    model = load_checkpoint_SD(config_path=args.config_path, ckpt_path=args.ckpt_path)
    train_iter, eval_iter = load_data_from_folder(args.data, resolution=512)
    seed = 0

    if args.mode == 'train':
        timesteps =  [999,246,214,176,130,67,1,0]
        model.trainer(train_iter, eval_iter,timesteps = timesteps)

    elif args.mode == 'eval':
        # os.makedirs(args.save_dir, exist_ok=True)

        timesteps = [700, 312, 110, 0]

        # ── 1. RD metrics loop (Kodak / small set) ────────────────────────────────
        all_bpps, all_psnrs, all_lpips = [], [], []
        all_compress_times, all_decompress_times = [], []
        num_images_to_plot = 24

        for i, image in enumerate(train_iter):
            compressed, compress_time = model.compress(image, timestep_path=timesteps)
            bits = [len(b) * 8 for b in compressed]
            reconstructions, decompress_time = model.decompress(
                compressed, image.shape, recon_method='flow_based', timestep_path=timesteps
            )
            assert (reconstructions[-1] == image).all()

            psnrs = [compute_psnr(image, recon) for recon in reconstructions]
            lpips_scores = [
                model.compute_lpips(recon, image).item() for recon in reconstructions
            ]
            bpps = [len(b) * 8 / (image.shape[-1] * image.shape[-2]) for b in compressed]

            all_bpps.append(bpps)
            all_psnrs.append(psnrs)
            all_lpips.append(lpips_scores)
            all_compress_times.append(compress_time)
            all_decompress_times.append(decompress_time)

            print(f'Image {i}: bpps={np.round(bpps, 4)}, PSNR={np.round(psnrs, 4)}, '
                f'LPIPS={np.round(lpips_scores, 4)}')

            if i + 1 >= num_images_to_plot:
                break

        all_bpps  = np.stack(all_bpps)
        all_psnrs = np.stack(all_psnrs)
        all_lpips = np.stack(all_lpips)
        mean_bpps  = np.mean(all_bpps,  axis=0)
        mean_psnrs = np.mean(all_psnrs, axis=0)
        mean_lpips = np.mean(all_lpips, axis=0)

        print("Mean BPP per step:",   np.round(mean_bpps,  4))
        print("Mean PSNR per step:",  np.round(mean_psnrs, 4))
        print("Mean LPIPS per step:", np.round(mean_lpips, 4))

        np.savetxt(os.path.join(args.save_dir, 'mean_bpps.txt'),  mean_bpps,  fmt='%.6f', header='bpp')
        np.savetxt(os.path.join(args.save_dir, 'mean_psnrs.txt'), mean_psnrs, fmt='%.6f', header='psnr')
        np.savetxt(os.path.join(args.save_dir, 'mean_lpips.txt'), mean_lpips, fmt='%.6f', header='lpips')

        # ── 2. FID loop — only runs if --fid_data is provided ────────────────────
        if args.fid_data is not None:
            print(f"\n{'='*60}")
            print(f"FID extraction on: {args.fid_data}  ({args.fid_num_images} images)")
            print(f"Saving features to: {args.fid_save_dir}")
            print(f"{'='*60}\n")

            os.makedirs(args.fid_save_dir, exist_ok=True)
            _, fid_iter = load_data_from_folder(args.fid_data, resolution=512)
            model._fid_shard_counters = {}

            for i, image in enumerate(fid_iter):
                if i >= args.fid_num_images:
                    break
                image = resize_and_crop(image, size=512)
                compressed, _ = model.compress(image, timestep_path=timesteps)
                reconstructions, _ = model.decompress(
                    compressed, image.shape,
                    recon_method='flow_based',
                    timestep_path=timesteps,
                )


                for recon, ts in zip(reconstructions, timesteps):
                    model.extract_and_save_features(
                        x_hat=recon,
                        x_raw=image,
                        timestep=ts,
                        save_dir=args.fid_save_dir,
                    )

                if i % 50 == 0:
                    print(f"[FID] {i+1}/{args.fid_num_images} images processed...")

            # ── 3. Compute + report FID aligned with RD metrics ───────────────────
            fid_scores = compute_fid_all_timesteps(args.fid_save_dir, min_samples=100)

            print(f"\n{'='*60}")
            print(f"{'Timestep':>10} | {'BPP':>8} | {'PSNR':>8} | {'LPIPS':>8} | {'FID':>8}")
            print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            for step_idx, ts in enumerate(timesteps):
                print(
                    f"{ts:>10} | {mean_bpps[step_idx]:>8.4f} | "
                    f"{mean_psnrs[step_idx]:>8.4f} | "
                    f"{mean_lpips[step_idx]:>8.4f} | "
                    f"{fid_scores.get(ts, float('nan')):>8.3f}"
                )
            print(f"{'='*60}\n")

            np.savetxt(
                os.path.join(args.save_dir, 'fid_results.txt'),
                np.column_stack([
                    timesteps[1:],
                    mean_bpps,
                    mean_psnrs,
                    mean_lpips,
                    [fid_scores.get(ts, float('nan')) for ts in timesteps[1:]],
                ]),
                header='timestep bpp psnr lpips fid', fmt='%.6f'
            )
            print(f"[FID] Saved to {os.path.join(args.save_dir, 'fid_results.txt')}")