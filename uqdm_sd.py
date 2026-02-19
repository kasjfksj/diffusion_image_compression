from networkx import config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.distributions import constraints, TransformedDistribution, SigmoidTransform, AffineTransform
from torch.distributions import Normal, Uniform
from torch.distributions.kl import kl_divergence
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, DDPMScheduler
# For compression to bits only
from tensorflow_compression.python.ops import gen_ops
import tensorflow as tf

from itertools import islice
from ml_collections import ConfigDict
import numpy as np
import json
import os
from pathlib import Path
from contextlib import contextmanager
import zipfile
from tqdm import tqdm

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

DATASET_PATH = {
    'ImageNet64': 'data/imagenet64/',
}


def softplus_inverse(x):
    """Helper which computes the inverse of `tf.nn.softplus`."""
    import math
    import numpy as np
    return math.log(np.expm1(x))
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
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        # Extra head to predict scale factors, matching VDM_Net's approach
        self.scale_head = nn.Conv2d(4, 4, kernel_size=1).cuda()
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def softplus_init1(self, x):
        return torch.nn.functional.softplus(x + self.SOFTPLUS_INV1)

    def forward(self, z, g_t):
        g_t = g_t.expand(z.shape[0])

        alpha2_target = torch.sigmoid(-g_t)
        alphas_cumprod = self.sd_scheduler.alphas_cumprod.to(z.device)
        diffs = (alphas_cumprod.unsqueeze(0) - alpha2_target.unsqueeze(1)).abs()
        timesteps = diffs.argmin(dim=1).long()

        null_cond = torch.zeros(z.shape[0], 77, 768, device=z.device, dtype=z.dtype)
        with torch.no_grad():
            eps_hat = self.unet(z, timesteps, encoder_hidden_states=null_cond).sample

        pred_scale_factors = self.softplus_init1(self.scale_head(eps_hat.detach()))
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
    # for IMAGENET64
    def __call__(self, image):
        image = torch.as_tensor(image.reshape(3, 64, 64), dtype=torch.uint8)
        return image

from torch.utils.data import Subset

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
        # if not train and remove_duplicates:
        #     removed = np.load(os.path.join(path, 'val_data.npz'))
        #     self.removed_idxs = [
        #         removed[(removed >= self.anchors[i]) & (removed < self.anchors[i + 1])] - self.anchors[i] for i in
        #         range(len(self.files))]
        #     self.anchors -= np.cumsum([0] + [np.size(r) for r in self.removed_idxs])
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
        # return sum(self.batch_lens)
        return self.anchors[-1]

    def __getitem__(self, idx):
        fid = np.argmax(idx < self.anchors) - 1
        idx = idx - self.anchors[fid]
        numpy_array = self.load_npy(fid)[idx]
        if self.transform is not None:
            torch_array = self.transform(numpy_array)
        return torch_array

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.data import Subset
def load_data_from_folder():
    """
    Load images from data/ folder and split into train (90%) and eval (10%) sets
    Returns infinitely looping training iterator and finite eval iterator
    """
    class ImageFolderFlat(Dataset):
        def __init__(self, folder_path, transform=None):
            self.folder_path = Path(folder_path)
            self.transform = transform
            self.image_paths = list(self.folder_path.glob('*.jpg')) + \
                              list(self.folder_path.glob('*.png')) + \
                              list(self.folder_path.glob('*.jpeg'))
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long())
    ])
    
    full_dataset = ImageFolderFlat('data_1/', transform=transform)
    
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    eval_size = total_size - train_size
    
    train_data, eval_data = random_split(full_dataset, [train_size, eval_size])
    
    train_iter = DataLoader(train_data, batch_size=32, shuffle=True,
                           pin_memory=True, num_workers=1)
    eval_iter = DataLoader(eval_data, batch_size=32, shuffle=False,
                          pin_memory=True, num_workers=1)
    
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
    # elif:   # Add more datasets here
    
    # Limit to only 5 data points
    train_data = Subset(train_data, range(min(5, len(train_data))))
    eval_data = Subset(eval_data, range(min(5, len(eval_data))))
    # print("asdfs",len(train_data))
    train_iter, eval_iter = [DataLoader(d, batch_size=cfg.batch_size, shuffle=cfg.get('shuffle', False),
                                        pin_memory=cfg.get('pin_memory', True), num_workers=cfg.get('num_workers', 1))
                             for d in [train_data, eval_data]]
    train_iter = cycle(train_iter)

    return train_iter, eval_iter


def load_checkpoint_SD(path, use_sd=False):
    """
    Load model from checkpoint.

    Input:
    ------
    path: path to a folder containing hyperparameters as config.json and parameters as checkpoint.pt
    """
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = ConfigDict(json.load(f))

    model = UQDM_SD(config).to(device)
    cp_path = config.get('restore_ckpt', None)
    if cp_path is not None:
        model.load(os.path.join(path, cp_path), use_sd=use_sd)

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
    # arg_constraints = {"delta": torch.distributions.constraints.nonnegative}

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
        # return torch.log(self.base_dist.cdf(y + self.half) - self.base_dist.cdf(y - self.half)) - self.log_delta
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
        total = 2 ** self.precision
        probs = self.prior.probs.reshape(-1, self.prior.probs.shape[-1])
        quantized_pdf = torch.round(probs * total).to(torch.int32)
        quantized_pdf = torch.clip(quantized_pdf, min=1)

        # Normalize pdf so that sum pmf_i = 2 ** precision
        while True:
            mask = quantized_pdf.sum(dim=-1) > total
            if not mask.any():
                break
            # m * (log2(v) - log2(v-1))
            penalty = probs[mask] * (torch.log2(1 + 1 / (quantized_pdf[mask] - 1)))
            # inf if v = 1 as intended but handle nan if also pmf = 0
            idx = penalty.nan_to_num(torch.inf).argmin(dim=-1)
            quantized_pdf[mask, idx] -= 1
        while True:
            mask = quantized_pdf.sum(axis=-1) < total
            if not mask.any():
                break
            # m * (log2(v+1) - log2(v))
            penalty = probs[mask] * (torch.log2(1 + 1 / quantized_pdf[mask]))
            idx = penalty.argmax(dim=-1)
            quantized_pdf[mask, idx] += 1

        quantized_cdf = torch.cumsum(quantized_pdf, dim=-1)
        self.quantized_cdf = torch.cat([
            - self.precision * torch.ones((quantized_pdf.shape[0], 1), device=device),
            torch.zeros((quantized_pdf.shape[0], 1), device=device),
            quantized_cdf
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
class Diffusion_SD(torch.nn.Module):
    """
    Progressive Compression with Gaussian Diffusion in LATENT SPACE.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.step = 0
        self.denoised = None
        
        # Load VAE for encoding/decoding
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae"
        )
        

        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = 0.18215
        self.sd_scheduler = DDPMScheduler.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="scheduler"
        )

        # Keep VDM gamma schedule for diffusion math
        self.register_buffer('alphas_cumprod', self.sd_scheduler.alphas_cumprod)  # shape (1000,)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.sd_scheduler.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.sd_scheduler.alphas_cumprod))

        # Keep VDM gamma only for timestep indexing
        self.gamma = self.get_noise_schedule(config)  # VDM schedule



    def t_to_timestep(self, t):
        if isinstance(t, torch.Tensor):
            t_val = t.flatten()[0].item()
        else:
            t_val = float(t)
        return int(t_val * 999) 

    

    def sigma2(self, t):
        return torch.sigmoid(self.gamma(t))

    def sigma(self, t):
        return torch.sqrt(self.sigma2(t))

    def alpha(self, t):
        return torch.sqrt(torch.sigmoid(-self.gamma(t)))

    def q_t(self, x_latent, t=1):
        # q(z_t | x_latent) = N(alpha_t * x_latent, sigma^2_t)
        # Now x_latent is in latent space [B, 4, H/8, W/8]
        return Normal(loc=self.alpha(t) * x_latent, scale=self.sigma(t))

    def p_1(self):
        # p(z_1) = N(0, 1) - still works in latent space
        return Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    # These remain unchanged
    def p_s_t(self, p_loc, p_scale, t, s):
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

    def get_s_t_params(self, z_t, t, s, x_latent=None, clip_denoised=True, cache_denoised=False, deterministic=False, x_raw_debug=None):
        """
        Now works in LATENT SPACE.
        z_t: [B, 4, H/8, W/8] noisy latent
        x_latent: [B, 4, H/8, W/8] clean latent (if provided)
        """
        gamma_t, gamma_s = self.gamma(t), self.gamma(s)
        alpha_t, alpha_s = self.alpha(t), self.alpha(s)
        sigma_t, sigma_s = self.sigma(t), self.sigma(s)
        expm1_term = (-torch.special.expm1(gamma_s - gamma_t))
        null_cond = torch.zeros(z_t.shape[0], 77, 768, device=z_t.device, dtype=z_t.dtype)
        
        if x_latent is None:
            # Predict noise using score network
            if self.config.model.get('learned_prior_scale'):

                eps_hat, pred_scale_factors = self.score_net(z_t, gamma_t)


            else:
                eps_hat = self.score_net(z_t, gamma_t, encoder_hidden_states=null_cond)
            
            # Compute denoised prediction in LATENT space
            if clip_denoised or cache_denoised:
                x_latent = (z_t - sigma_t * eps_hat) / alpha_t  # Still in latent space

                decode_and_save(z_t, eps_hat, alpha_t, sigma_t, self.vae, self.vae_scale_factor)

            if clip_denoised:
                # Clip in latent space (less aggressive than [-1,1])
                x_latent.clamp_(-4.0, 4.0)  # Latents can have larger range
            
            if cache_denoised:
                self.denoised = x_latent
         
            scale = sigma_s * torch.sqrt(expm1_term)

            if self.config.model.get('base_prior_scale', 'forward_kernel') == 'forward_kernel':
                scale = sigma_t * torch.sqrt(expm1_term)
            if self.config.model.get('learned_prior_scale'):
                scale = scale * pred_scale_factors
        else:
            scale = sigma_s * torch.sqrt(expm1_term)

        # Mean computation - same formulas, different space
        if x_latent is not None:
            if deterministic:
                loc = sigma_s / sigma_t * z_t - (alpha_t * sigma_s / sigma_t - alpha_s) * x_latent
            else:
                loc = alpha_s * ((1 - expm1_term) / alpha_t * z_t + expm1_term * x_latent)
        else:
            if deterministic:
                loc = alpha_s / alpha_t * z_t + (sigma_s - alpha_s / alpha_t * sigma_t) * eps_hat
            else:
                loc = alpha_s / alpha_t * (z_t - sigma_t * expm1_term * eps_hat)

        return loc, scale

    def transmit_q_s_t(self, x_latent, z_t, t, s, compress_mode=None, cache_denoised=False,x_raw=None):
        """Now x_latent is in latent space"""
        p_loc, p_scale = self.get_s_t_params(z_t, t, s, cache_denoised=cache_denoised, x_raw_debug=x_raw)
        q_loc, q_scale = self.get_s_t_params(z_t, t, s, x_latent=x_latent, x_raw_debug=x_raw)


        p_s_t = self.p_s_t(p_loc, p_scale, t, s)
        q_s_t = self.q_s_t(q_loc, q_scale)
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

    def forward(self, x_raw, z_1=None, recon_method=None, compress_mode=None, seed=None):
        """
        x_raw: [B, 3, H, W] in range [0, 255]
        Everything else happens in latent space
        """

        

        # Encode to latent space ONCE at the beginning
        x_pixel = 2 * ((x_raw.float() + .5) / self.config.model.vocab_size) - 1  # [-1, 1]
        with torch.no_grad():
            x_latent = self.vae.encode(x_pixel).latent_dist.sample() * self.vae_scale_factor
            x_latent = x_latent.detach()
        rescale_latent_to_bpd = 1. / (np.prod(x_latent.shape[1:]) * np.log(2.))
        rescale_pixel_to_bpd = 1. / (np.prod(x_raw.shape[1:]) * np.log(2.))


        # 1. PRIOR/LATENT LOSS - now in latent space
        if z_1 is None and not torch.is_inference_mode_enabled():
            q_1 = self.q_t(x_latent)
            p_1 = self.p_1()
            with local_seed(seed, i=0):
                z_1 = q_1.sample()
            loss_prior = kl_divergence(q_1, p_1).sum(dim=[1, 2, 3])
        else:
            if z_1 is None:
                p_1 = self.p_1()
                with local_seed(seed, i=0):
                    z_1 = p_1.sample(x_latent.shape)  # Sample in latent space shape
            loss_prior = torch.zeros(x_latent.shape[0], device=device)

        # 2. DIFFUSION LOSS - all in latent space
        z_s = z_1
        rate_s = loss_prior
        loss_diff = 0.
        times = torch.linspace(1, 0, self.config.model.n_timesteps + 1, device=device)
        metrics = []

        for i in range(len(times) - 1):
            z_t = z_s
            rate_t = rate_s
            t, s = times[i], times[i + 1]
            with local_seed(seed, i=i + 1):
                z_s, rate_s = self.transmit_q_s_t(x_latent, z_t, t, s, compress_mode=compress_mode,
                                                  cache_denoised=recon_method == 'denoise', x_raw=x_raw)
            loss_diff += rate_s

            if recon_method is not None:
                x_hat_t = self.denoise_z_t(z_t, recon_method, times=times[i:])
                metrics += [{
                    'prog_bpds': rate_t.cpu() * rescale_pixel_to_bpd,
                    'prog_x_hats': x_hat_t.detach().cpu(),
                    'prog_mses': torch.mean((x_hat_t - x_raw).float() ** 2, dim=[1, 2, 3]).cpu(),
                }]

        z_0_latent = z_s
        
        if recon_method is not None:
            if recon_method == 'ancestral':
                x_hat_t = self.decode_p_x_z_0(z_0_latent=z_0_latent, method='sample')
            else:
                x_hat_t = self.decode_p_x_z_0(z_0_latent=z_0_latent, method='argmax')
            metrics += [{
                'prog_bpds': rate_s.cpu() * rescale_pixel_to_bpd,
                'prog_x_hats': x_hat_t.detach().cpu(),
                'prog_mses': torch.mean((x_hat_t - x_raw).float() ** 2, dim=[1, 2, 3]).cpu(),
            }]

        # 3. RECONSTRUCTION LOSS
        log_probs = self.log_probs_x_z0(z_0_latent=z_0_latent, x_raw=x_raw)
        
        loss_recon = -log_probs.sum(dim=[1, 2, 3])
        x_raw = self.transmit_image(z_0_latent, x_raw, compress_mode=compress_mode)

        if recon_method is not None:
            metrics += [{
                'prog_bpds': loss_recon.cpu() * rescale_pixel_to_bpd,
                'prog_x_hats': x_raw.cpu(),
                'prog_mses': torch.zeros(x_pixel.shape[:1]),
            }]
            metrics = default_collate(metrics)
        else:
            metrics = {}

        bpd_latent = torch.mean(loss_prior) * rescale_latent_to_bpd
        bpd_diff   = torch.mean(loss_diff)  * rescale_latent_to_bpd
        bpd_recon  = torch.mean(loss_recon) * rescale_pixel_to_bpd
        loss = bpd_recon + bpd_latent + bpd_diff
        metrics.update({
            "bpd": loss,
            "bpd_latent": bpd_latent,
            "bpd_recon": bpd_recon,
            "bpd_diff": bpd_diff,
        })

        return loss, metrics



    @torch.no_grad()
    def sample(self, init_z=None, shape=None, times=None, deterministic=False,
            clip_samples=False, decode_method='argmax', return_hist=False):
        
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        scheduler.set_timesteps(50)

        if init_z is None:
            assert shape is not None
            z = torch.randn(shape, device=device)
        else:
            z = init_z

        if return_hist:
            samples = [z]

        null_cond = torch.zeros(z.shape[0], 77, 768, device=device)

        for t in scheduler.timesteps:
            noise_pred = self.score_net.unet(z, t, encoder_hidden_states=null_cond).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
            if return_hist:
                samples.append(z)

        x_raw = self.decode_p_x_z_0(z_0_latent=z, method=decode_method)

        if return_hist:
            return x_raw, samples + [x_raw]
        return x_raw

    

    def entropy_encode(self, k, p):
        """
        Encode integer array k to bits using a prior / coding distribution p.
        We might want to quantize scale for determinism and added stability across multiple machines.
        """
        # When using a scalar prior it would be better to quantize u as in tfc.UniversalBatchedEntropyModel
        assert self.config.model.learned_prior_scale
        em = EntropyModel(p)
        bitstring = em.compress(k)
        return bitstring

    def entropy_decode(self, bits, p):
        """
        Decode integer array from bits using the prior p.
        """
        assert self.config.model.learned_prior_scale
        em = EntropyModel(p)
        k = em.decompress(bits)
        return k

    @torch.inference_mode()
    def compress(self, image):
        # return the bits for each step
        self.compress_bits = []
        # accumulate bits
        self.forward(image.to(device), compress_mode='encode', seed=0)
        return self.compress_bits

    @torch.inference_mode()
    def decompress(self, bits, image_shape, recon_method='denoise'):
        # consume the bits for each step, return the intermediate reconstructions for each step
        self.compress_bits = bits.copy()
        # consume the bits for each step
        _, metrics = self.forward(torch.zeros(image_shape, device=device), compress_mode='decode',
                                  recon_method=recon_method, seed=0)
        return metrics['prog_x_hats']

    def log_probs_x_z0(self, z_0_latent, x_raw=None):
        """
        Decode z_0_latent and compute pixel probabilities
        z_0_latent: [B, 4, H/8, W/8]
        """
        # Decode latent to pixel space
        with torch.no_grad():
            z_0_pixel = self.vae.decode(z_0_latent / self.vae_scale_factor).sample
        
        gamma_0 = self.gamma(torch.tensor([0.0], device=device))

        z_0_rescaled = z_0_pixel / torch.sqrt(torch.sigmoid(-gamma_0))

        
        x_vals = torch.arange(self.config.model.vocab_size, device=z_0_rescaled.device)
        
        x_vals = 2 * ((x_vals + .5) / self.config.model.vocab_size) - 1
        x_vals = torch.reshape(x_vals, [1] * z_0_rescaled.ndim + [-1])
        
        z = z_0_rescaled.unsqueeze(-1)

        logits = -0.5 * torch.exp(-gamma_0) *  (z - x_vals) ** 2

        logprobs = torch.log_softmax(logits, dim=-1)

        if x_raw is None:
            return logprobs
        else:
            x_one_hot = nn.functional.one_hot(x_raw.long(), num_classes=self.config.model.vocab_size)
            log_probs = (x_one_hot * logprobs).sum(-1)
            return log_probs

    def decode_p_x_z_0(self, z_0_latent, method='argmax'):
        """Decode latent to pixels"""
        logprobs = self.log_probs_x_z0(z_0_latent=z_0_latent)
        if method == 'argmax':
            x_raw = torch.argmax(logprobs, dim=-1)
        elif method == 'sample':
            x_raw = torch.distributions.Categorical(logits=logprobs).sample()
        else:
            raise ValueError(f"Unknown decoding method {method}")
        return x_raw

    def denoise_z_t(self, z_t, recon_method, times=None):
        """z_t is in latent space"""
        if recon_method == 'ancestral':
            x_hat_t = self.sample(
                times=times, init_z=z_t,
                clip_samples=True, decode_method='argmax', return_hist=False
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
        # elif:    # add different noise schedules here
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
        torch.save({
            'model': self.score_net.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
        }, self.self.config.checkpoint_path)

    def load(self, path, use_sd=False):
        from diffusers import DDPMScheduler, UNet2DConditionModel
        self.score_net = SD15ScoreNet(self.config)
            # DON'T create optimizer for inference!
            # self.optimizer = torch.optim.Adam(...)  # ← REMOVE THIS
            
            # If you have SD checkpoint to load:
            # cp = torch.load(path, map_location=device, weights_only=False)
            # self.score_net.load_state_dict(cp['model'], strict=False)


    def trainer(self, train_iter, eval_iter=None):
        """
        Train UQDM for a specified number of steps on a train set.
        Hyperparameters are set via self.config.training, self.config.eval, and self.config.optim.
        """

        if self.step >= self.config.training.n_steps:
            print('Skipping training, increase training.n_steps if more steps are desired.')

        while self.step < self.config.training.n_steps:
            # Parameter update step
            batch = next(train_iter).to(device)
            self.optimizer.zero_grad()
            model.train()
            loss, metrics = self(batch)
            loss.backward()
            if self.config.optim.warmup > 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config.optim.lr * np.minimum(self.step / self.config.optim.warmup, 1.0)
            if self.config.optim.grad_clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.optim.grad_clip_norm)
            self.optimizer.step()
            self.step += 1
            self.ema.update(model.parameters())

            last = self.step == self.config.training.n_steps
            # Save model checkpoint
            if self.step % self.config.training.log_metrics_every_steps == 0 or last:
                self.save()
            # Print train metrics
            if self.step % self.config.training.log_metrics_every_steps == 0 or last:
                print(metrics)
            # Compute and print validation metrics
            if eval_iter is not None and (self.step % self.config.training.eval_every_steps == 0 or last):
                n_batches = self.config.training.eval_steps_to_run
                res = []
                for batch in tqdm(islice(eval_iter, n_batches), total=n_batches or len(eval_iter),
                                  desc='Evaluating on test set'):
                    batch = batch.to(device)
                    with torch.inference_mode():
                        self.ema.store(model.parameters())
                        self.ema.copy_to(model.parameters())
                        model.eval()
                        _, ths_metrics = self(batch)
                        self.ema.restore(model.parameters())
                    res += [ths_metrics]
                res = default_collate(res)
                print({k: v.mean().item() for k, v in res.items()})

    @staticmethod
    def mse_to_psnr(mse, max_val):
        with np.errstate(divide='ignore'):
            return -10 * (np.log10(mse) - 2 * np.log10(max_val))

    @torch.inference_mode()
    def evaluate(self, eval_iter, n_batches=None, seed=None):
        """
        Evaluate rate-distortion on the test set.

        Inputs:
        -------
        n_batches - (optionally) give a number of batches to evaluate
        """

        res = []
        for X in tqdm(islice(eval_iter, n_batches), total=n_batches or len(eval_iter), desc='Evaluating UQDM'):
            X = X.to(device)
            ths_res = {}
            
            for recon_method in ('denoise', 'ancestral', 'flow_based'):
                
                # If evaluating bpds as file sizes:
                # self.compress_bits = []
                # loss, metrics = self(X, recon_method=recon_method, seed=seed, compress_mode='encode')
                # bpds = np.cumsum([len(b) * 8 for b in self.compress_bits]) / np.prod(X.shape)
                loss, metrics = self(X, recon_method=recon_method, seed=seed)
                bpds = np.cumsum(metrics['prog_bpds'].mean(dim=1))
                psnrs = self.mse_to_psnr(metrics['prog_mses'].mean(dim=1), max_val=255.)
                ths_res[recon_method] = dict(bpds=bpds, psnrs=psnrs)
            res += [ths_res]
        res = default_collate(res)

        for recon_method in res.keys():
            bpps = np.round(3 * res[recon_method]['bpds'].mean(axis=0).numpy(), 4)
            psnrs = np.round(res[recon_method]['psnrs'].mean(axis=0).numpy(), 4)
            print('Reconstructions via: %s\nbpps:  %s\npsnrs: %s\n' % (recon_method, bpps, psnrs))


class UQDM_SD(Diffusion_SD):
    """
    Making Progressive Compression tractable with Universal Quantization.
    """

    def __init__(self, config):
        """
        See Diffusion.__init__ for hyperparameters.
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

            # very slow, ~ 25 symbols/s
            # cp = tfc.NoisyLogistic(loc=0.0, scale=(p.base_dist.scale / p.delta).cpu().numpy())
            # em2 = tfc.UniversalBatchedEntropyModel(cp, coding_rank=4, compression=True, num_noise_levels=30)
            # k = (q.mean - p.mean) / p.delta
            # bitstring = em2.compress(k.cpu())
            # k_hat = em2.decompress(bitstring, [])

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


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # model = load_checkpoint('checkpoints/uqdm-tiny')
    # model = load_checkpoint('checkpoints/uqdm-small')
    model = load_checkpoint_SD('checkpoints/uqdm-medium')
    # model = load_checkpoint('checkpoints/uqdm-big')
    train_iter, eval_iter = load_data('ImageNet64', model.config.data)
    
    # model.trainer(train_iter, eval_iter)
    model.evaluate(eval_iter, n_batches=10, seed=seed)

    # Compress one image
    image = next(iter(eval_iter))
    print(image, dir(image))
    compressed = model.compress(image)
    bits = [len(b) * 8 for b in compressed]
    reconstructions = model.decompress(compressed, image.shape, recon_method='ancestral')
    assert (reconstructions[-1] == image).all()
    print('Reconstructions via: denoise, compression to bits\nbpps:  %s'
          % np.round(np.cumsum(bits) / np.prod(image.shape) * 3, 4))
