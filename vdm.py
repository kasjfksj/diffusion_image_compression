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
import matplotlib.pyplot as plt
from itertools import islice
from ml_collections import ConfigDict
import numpy as np
import json
import os
from pathlib import Path
from contextlib import contextmanager
import zipfile
from tqdm import tqdm

from uqdm_sd import load_checkpoint_SD

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
            "runwayml/stable-diffusion-v1-5",
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
            self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).byte()),  # convert to [0, 255] uint8
            ])
            self.image_paths = list(self.folder_path.glob('*.jpg')) + \
                              list(self.folder_path.glob('*.png')) + \
                              list(self.folder_path.glob('*.jpeg')) + \
                              list(self.folder_path.glob('*.JPEG'))
            
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
    full_dataset = torch.utils.data.Subset(full_dataset, range(100)) 
    total_size = len(full_dataset)
    print(total_size)
    train_size = int(0.9 * total_size)
    eval_size = total_size - train_size
    print(train_size,eval_size)
    train_data, eval_data = random_split(full_dataset, [train_size, eval_size])
    
    train_iter = DataLoader(train_data, batch_size=32, shuffle=True,
                           pin_memory=True, num_workers=0)
    eval_iter = DataLoader(eval_data, batch_size=1, shuffle=False,
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
    # elif:   # Add more datasets here
    
    # Limit to only 5 data points
    train_data = Subset(train_data, range(min(5, len(train_data))))
    eval_data = Subset(eval_data, range(min(5, len(eval_data))))
    # print("asdfs",len(train_data))
    train_iter, eval_iter = [DataLoader(d, batch_size=cfg.batch_size, shuffle=cfg.get('shuffle', False),
                                        pin_memory=cfg.get('pin_memory', True), num_workers=cfg.get('num_workers', 0))
                             for d in [train_data, eval_data]]
    train_iter = cycle(train_iter)

    return train_iter, eval_iter


def load_checkpoint_SD_VDM(path, use_sd=False):
    """
    Load model from checkpoint.

    Input:
    ------
    path: path to a folder containing hyperparameters as config.json and parameters as checkpoint.pt
    """
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = ConfigDict(json.load(f))

    model = Diffusion_SD(config).to(device)
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
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, delta=1.0, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.delta = delta  # quantization step size
        base_dist = Uniform(torch.tensor(0, dtype=loc.dtype, device=loc.device),
                            torch.tensor(1, dtype=loc.dtype, device=loc.device))
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticDistribution, _instance)
        return super().expand(batch_shape, _instance=new)

    def cdf(self, x):
        return torch.sigmoid((x - self.loc) / self.scale)

    @staticmethod
    def log_sigmoid(x):
        return -torch.nn.functional.softplus(-x)

    def log_cdf(self, x):
        standardized = (x - self.loc) / self.scale
        return self.log_sigmoid(standardized)

    def log_survival_function(self, x):
        standardized = (x - self.loc) / self.scale
        return self.log_sigmoid(-standardized)

    def log_prob(self, x):
        # log p(x) = log CDF(x + delta/2) - log CDF(-(x - delta/2))
        # i.e. log of the probability mass in the bin [x-delta/2, x+delta/2]
        # This is more numerically stable than the default TransformedDistribution log_prob
        upper = (x + self.delta / 2 - self.loc) / self.scale
        lower = (x - self.delta / 2 - self.loc) / self.scale
        # log(sigmoid(upper) - sigmoid(lower)) using log-sum-exp trick for stability
        log_prob = self.log_sigmoid(upper) + torch.log1p(-torch.exp(self.log_sigmoid(lower) - self.log_sigmoid(upper)))
        return log_prob

    def discretize(self, u):
        """
        Build a discrete distribution for entropy coding under dither u ~ U(-0.5, 0.5).
        Returns a Categorical over integer symbols k, where:
            P(k) = CDF((k + 0.5 - loc_shifted) * delta) - CDF((k - 0.5 - loc_shifted) * delta)
        and loc_shifted = (loc + delta * u) / delta accounts for the dither.
        """
        # Number of symbols — covers ~6 std devs on each side
        K = max(512, int(6 * self.scale.max().item() / self.delta) * 2)
        K = min(K, 1024)  # cap to avoid memory explosion

        # Shift the loc to account for dither
        loc_shifted = (self.loc + self.delta * u) / self.delta  # in units of delta

        # Integer symbol range centered at 0
        k_vals = torch.arange(-K // 2, K // 2, device=self.loc.device).float()
        # Reshape for broadcasting: [1, 1, 1, 1, K]
        k_vals = k_vals.reshape([1] * self.loc.ndim + [-1])

        # CDF differences to get probability of each symbol
        upper = (k_vals + 0.5 - loc_shifted.unsqueeze(-1)) * (self.delta / self.scale.unsqueeze(-1))
        lower = (k_vals - 0.5 - loc_shifted.unsqueeze(-1)) * (self.delta / self.scale.unsqueeze(-1))

        log_probs = self.log_sigmoid(upper) + torch.log1p(
            -torch.exp(self.log_sigmoid(lower) - self.log_sigmoid(upper)).clamp(max=1 - 1e-6)
        )

        return torch.distributions.Categorical(logits=log_probs)


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
import math
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
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.register_buffer('alphas_cumprod', self.sd_scheduler.alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.sd_scheduler.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.sd_scheduler.alphas_cumprod))

        # ← NO self.gamma = get_noise_schedule() here, we use the method below instead
    def gamma(self, t):
        """Log-SNR at integer SD timestep t in [0, 999]."""
        alpha_bar = self.alphas_cumprod[t].clamp(1e-6, 1 - 1e-6)

        return torch.log((1.0 - alpha_bar) / alpha_bar)

    def sigma2(self, t):
        return (1.0 - self.alphas_cumprod[t]).clamp(1e-6, 1 - 1e-6)

    def sigma(self, t):
        return torch.sqrt(self.sigma2(t))

    def alpha(self, t):
        return torch.sqrt(self.alphas_cumprod[t].clamp(1e-6, 1 - 1e-6))
    
    def q_t(self, x_latent, t=1):
        # q(z_t | x_latent) = N(alpha_t * x_latent, sigma^2_t)
        # Now x_latent is in latent space [B, 4, H/8, W/8]
        return Normal(loc=self.alpha(t) * x_latent, scale=self.sigma(t))

    def p_1(self):
        # p(z_1) = N(0, 1) - still works in latent space
        return Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    def p_s_t(self, p_loc, p_scale, t, s):
        return NormalDistribution(loc=p_loc, scale=p_scale)

    def q_s_t(self, q_loc, q_scale):
        return NormalDistribution(loc=q_loc, scale=q_scale)

    def relative_entropy_coding(self, q, p, compress_mode=None):
        if not torch.is_inference_mode_enabled():
            z_s = q.sample()
        else:
            # Sample from posterior mean (deterministic) or sample
            z_s = q.sample()
        # print((p.scale-q.scale).sum())
        from torch.distributions import kl_divergence
        rate = kl_divergence(q, p).sum(dim=[1, 2, 3])
        return z_s, rate

    def get_s_t_params(self, z_t, t, s, x_latent=None, clip_denoised=True, cache_denoised=False, deterministic=False, x_raw_debug=None):
        """
        Now works in LATENT SPACE.
        z_t: [B, 4, H/8, W/8] noisy latent
        x_latent: [B, 4, H/8, W/8] clean latent (if provided)
        """
        # print(self.gamma(1))
        # import sys
        # sys.exit()
        gamma_t, gamma_s = self.gamma(t), self.gamma(s)
        alpha_t, alpha_s = self.alpha(t), self.alpha(s)
        sigma_t, sigma_s = self.sigma(t), self.sigma(s)
        expm1_term = (-torch.special.expm1(gamma_s - gamma_t))

        null_cond = torch.zeros(z_t.shape[0], 77, 768, device=z_t.device, dtype=z_t.dtype)
        
        # === COMPUTE SCALE ONCE (shared for both p and q) ===
        scale = sigma_s * torch.sqrt(expm1_term)
        
        if self.config.model.get('base_prior_scale', 'forward_kernel') == 'forward_kernel':
            scale = sigma_t * torch.sqrt(expm1_term)
        
        pred_scale_factors = None
        if x_latent is None:
            # Predict noise using score network
            if self.config.model.get('learned_prior_scale'):
                eps_hat, pred_scale_factors = self.score_net(z_t, gamma_t)
            else:
                eps_hat, _ = self.score_net(z_t, gamma_t)
            
            # Apply learned scale factors if they exist
            if self.config.model.get('learned_prior_scale'):
                scale = scale * pred_scale_factors
            
            # Compute denoised prediction in LATENT space
            if clip_denoised or cache_denoised:
                x_latent = (z_t - sigma_t * eps_hat) / alpha_t
            
            if clip_denoised:
                x_latent.clamp_(-4.0, 4.0)
            
            if cache_denoised:
                self.denoised = x_latent

        # Mean computation
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
        # 2. DIFFUSION LOSS - all in latent space
        z_s = z_1
        rate_s = loss_prior
        loss_diff = 0.
        sd_timesteps = self.sd_scheduler.timesteps  # tensor([999, 998, ..., 0])
        total_steps = len(sd_timesteps) - 1
        
        # Generate eval_indices based on total_steps, including both endpoints
        n_eval_samples = 8  # Customize this number
        eval_indices = set(np.linspace(0, total_steps - 1, n_eval_samples, dtype=int))
        
        metrics = []

        for i in range(total_steps):
            z_t = z_s
            rate_t = rate_s
            ts_t = sd_timesteps[i].item()
            ts_s = sd_timesteps[i + 1].item()

            with local_seed(seed, i=i + 1):
                z_s, rate_s = self.transmit_q_s_t(
                    x_latent, z_t, ts_t, ts_s,
                    compress_mode=compress_mode,
                    cache_denoised=recon_method == 'denoise',
                    x_raw=x_raw
                )
            loss_diff += rate_s

            if recon_method is not None :
                x_hat_t = self.denoise_z_t(z_t, recon_method, times=ts_t)
                metrics += [{
                    'prog_bpds': rate_t.cpu() * rescale_latent_to_bpd,  # per-step, same as original
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
                'prog_bpds': rate_s.cpu() * rescale_latent_to_bpd,
                'prog_x_hats': x_hat_t.detach().cpu(),
                'prog_mses': torch.mean((x_hat_t - x_raw).float() ** 2, dim=[1, 2, 3]).cpu(),
            }]

        # 3. RECONSTRUCTION LOSS
        log_probs = self.log_probs_x_z0(z_0_latent=z_0_latent, x_raw=x_raw)
        
        loss_recon = -log_probs.sum(dim=[1, 2, 3])
        x_raw = self.transmit_image(z_0_latent, x_raw, compress_mode=compress_mode)

        if recon_method is not None:
            metrics += [{
                'prog_bpds': loss_recon.cpu() * rescale_latent_to_bpd,
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

        if init_z is None:
            assert shape is not None
            z = torch.randn(shape, device=device)
        else:
            z = init_z

        if return_hist:
            samples = [z]

        null_cond = torch.zeros(z.shape[0], 77, 768, device=device)

        # times is now an integer SD timestep (e.g. 999, 500, 100)
        if times is not None:
            start_ts = int(times)   # already an integer SD timestep
            n_steps = max(start_ts // 100, 1)  # proportional number of denoising steps
        else:
            start_ts = 999
            n_steps = 10

        scheduler.set_timesteps(n_steps)

        # Filter to only timesteps <= start_ts
        valid_timesteps = [t for t in scheduler.timesteps if t.item() <= start_ts]

        for t in valid_timesteps:
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
        
        gamma_0 = self.gamma(torch.tensor([0], device=device))

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
        for X in tqdm(islice(eval_iter, n_batches), total=n_batches or len(eval_iter), desc='Evaluating VDM'):
            print('Evaluating batch %s...' % len(res))
            X = X.to(device)
            ths_res = {}
            recon_method = 'denoise'

            # If evaluating bpds as file sizes:
            # self.compress_bits = []
            # loss, metrics = self(X, recon_method=recon_method, seed=seed, compress_mode='encode')
            # bpds = np.cumsum([len(b) * 8 for b in self.compress_bits]) / np.prod(X.shape)
            # print(1)
            loss, metrics = self(X, recon_method=recon_method, seed=seed)
            # print(2)
            bpds = np.cumsum(metrics['prog_bpds'].mean(dim=1))
            # print(3)
            psnrs = self.mse_to_psnr(metrics['prog_mses'].mean(dim=1), max_val=255.)
            # print(4)
            ths_res[recon_method] = dict(bpds=bpds, psnrs=psnrs)
            # print(5)
            
            res += [ths_res]
        res = default_collate(res)

        for recon_method in res.keys():
            bpps = np.round(3 * res[recon_method]['bpds'].mean(axis=0).numpy(), 4)
            psnrs = np.round(res[recon_method]['psnrs'].mean(axis=0).numpy(), 4)
            print('Reconstructions via: %s\nbpps:  %s\npsnrs: %s\n' % (recon_method, bpps, psnrs))

        return bpps, psnrs
    @torch.no_grad()
    def compute_cost_matrix(self, data_iter, seed=None, timestep_stride=100,
                            num_images=1, use_kl=True, cache_path=None):
        """
        Build a BPP matrix averaged over num_images images from data_iter.
    
        Entry [i, j] = mean compression cost (bpp) of transmitting
        from timestep ts[i] → ts[j], averaged over real image diversity.
    
        This matrix stores RAW KL-based bpp only.
        DiffC cost transformation (I + log2(I) + 5) is applied separately
        in find_optimal_path_dp for DP optimization.
    
        Args:
            data_iter:       iterator yielding batches of [B, 3, H, W] images (0-255)
            seed:            random seed for reproducibility
            timestep_stride: step between timesteps
            num_images:      number of images to average over
            use_kl:          if True, use KL divergence
    
        Returns:
            cost_matrix: [T, T] upper-triangular matrix (numpy), entry [i,j] = mean bpp
            timesteps:   list of SD integer timesteps corresponding to matrix indices
        """
        # Select timesteps once (shared across all images)
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached cost matrix from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            cost_matrix = data['cost_matrix']
            timesteps = data['timesteps'].tolist()
            print(f"Loaded cost matrix of shape {cost_matrix.shape} with {len(timesteps)} timesteps")
            return cost_matrix, timesteps
        all_ts = self.sd_scheduler.timesteps
        timesteps = all_ts[::timestep_stride].tolist()
        if all_ts[-1].item() not in timesteps:
            timesteps.append(all_ts[-1].item())
        T = len(timesteps)
    
        total_transitions = T * (T - 1) // 2
        print(f"Computing BPP matrix for {T} timesteps (stride={timestep_stride})...")
        print(f"Total transitions to evaluate: {total_transitions}")
        print(f"Averaging over {num_images} images")
        print(f"Using {'KL divergence' if use_kl else 'relative entropy coding'}")
    
        # Accumulate raw BPP matrices across images, then divide
        cost_matrix_accum = np.zeros((T, T))
    
        import time
        start_time = time.time()
        print(num_images)
        for img_idx in range(num_images):
            x_raw = next(data_iter).to(device)  # [B, 3, H, W]
            print(img_idx)
            print(f"\n[Image {img_idx + 1}/{num_images}] Encoding to latent space...")
    
            x_pixel = 2 * ((x_raw.float() + .5) / self.config.model.vocab_size) - 1
            x_latent = self.vae.encode(x_pixel).latent_dist.sample() * self.vae_scale_factor
            x_latent = x_latent.detach()
            rescale_latent_to_bpd = 1. / (np.prod(x_raw.shape[1:]) * np.log(2.))
    
            # Per-image raw bpp matrix
            cost_matrix_img = np.zeros((T, T))
    
            # --- Cache z_t and score predictions for this image ---
            print(f"  Caching z_t and score predictions for {T} timesteps...")
            z_cache = {}
            eps_cache = {}
            scale_cache = {}
    
            with local_seed(seed, i=img_idx * 10_000):
                base_eps = torch.randn_like(x_latent)  # single noise realization

            for idx, ts_val in enumerate(timesteps):
                ts_tensor = torch.tensor(ts_val, device=x_latent.device)
                alpha_t = self.alpha(ts_tensor)
                sigma_t = self.sigma(ts_tensor)
                
                # z_t = alpha_t * x + sigma_t * eps  (same eps for all t)
                z_t = alpha_t * x_latent + sigma_t * base_eps
                z_cache[ts_val] = z_t.detach()

                gamma_t = self.gamma(ts_tensor)
                if self.config.model.get('learned_prior_scale'):
                    eps_hat, pred_scale_factors = self.score_net(z_t, gamma_t)
                    scale_cache[ts_val] = pred_scale_factors.detach()
                else:
                    eps_hat = self.score_net(z_t, gamma_t)[0]
                    scale_cache[ts_val] = None
                eps_cache[ts_val] = eps_hat.detach()
    
            # --- Fill raw BPP matrix for this image ---
            computed = 0
            for i, ts_t in enumerate(timesteps):
                z_t     = z_cache[ts_t]
                eps_t   = eps_cache[ts_t]
                scale_t = scale_cache[ts_t]
                ts_t_tensor = torch.tensor(ts_t, device=x_latent.device)
    
                for j in range(i + 1, T):
                    ts_s = timesteps[j]
                    ts_s_tensor = torch.tensor(ts_s, device=x_latent.device)
    
                    # Store raw KL-based bpp only — no DiffC transform here
                    rate_bpp = self._compute_kl_rate(
                        z_t, ts_t_tensor, ts_s_tensor, x_latent, rescale_latent_to_bpd,
                        eps_t, scale_t, use_kl,
                    )
                    cost_matrix_img[i, j] = rate_bpp
                    computed += 1
    
            cost_matrix_accum += cost_matrix_img
    
            elapsed = time.time() - start_time
            print(f"  Done. Elapsed: {elapsed:.1f}s")
    
            # Running mean diagnostic (raw bpp)
            running_mean = cost_matrix_accum / (img_idx + 1)
            row_path = sum(running_mean[i, i+1] for i in range(T-1))
            print(f"  Running mean consecutive-step BPP: {row_path:.4f}")
    
        # Final average over all images (raw bpp)
        cost_matrix = cost_matrix_accum / num_images
    
        row_path = sum(cost_matrix[i, i+1] for i in range(T-1))
        print(f"\nFinal mean consecutive-step BPP (over {num_images} images): {row_path:.4f}")
    
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, cost_matrix=cost_matrix, timesteps=np.array(timesteps))
            print(f"Saved cost matrix to {cache_path}")

        return cost_matrix, timesteps


    def _compute_kl_rate(self, z_t, t, s, x_latent, rescale_latent_to_bpd,
                        eps_t=None, scale_t=None, use_kl=True):
        """
        KL rate for a single image, using the cached z_t and eps_t.
        Returns raw KL-based bpp. No DiffC cost transform applied here.
        """
        alpha_t  = self.alpha(t)
        alpha_s  = self.alpha(s)
        sigma_t  = self.sigma(t)
        sigma_s  = self.sigma(s)
    
        alpha_ts  = alpha_t / alpha_s
        sigma2_ts = (sigma_t**2 - alpha_ts**2 * sigma_s**2).clamp(1e-8)
        sigma2_Q  = (sigma2_ts * sigma_s**2 / sigma_t**2).clamp(1e-8)
        coeff_x   = alpha_s * sigma2_ts / sigma_t**2
    
        x_hat = (z_t - sigma_t * eps_t) / alpha_t
        diff  = coeff_x * (x_latent - x_hat)
        kl    = (diff**2 / (2 * sigma2_Q)).sum(dim=[1, 2, 3])  # [B]
    
        return kl.mean().item() * rescale_latent_to_bpd
    



    @staticmethod
    def _gaussian_kl(q_loc, q_scale, p_loc, p_scale):
        """
        KL( N(q_loc, q_scale^2) || N(p_loc, p_scale^2) ) elementwise.
        Returns tensor same shape as inputs.
        
        Args:
            q_loc: Posterior mean
            q_scale: Posterior scale
            p_loc: Prior mean
            p_scale: Prior scale
        
        Returns:
            kl: KL divergence (same shape as inputs)
        """
        return (
            torch.log(p_scale / q_scale.clamp(1e-8))
            + (q_scale**2 + (q_loc - p_loc)**2) / (2 * p_scale**2 + 1e-8)
            - 0.5
        )


import time
import numpy as np

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
    print("Checking KL additivity...")
    T = len(bpp_matrix)
    max_violation = 0
    for i in range(T):
        for k in range(i+1, T):
            for j in range(k+1, T):
                direct    = bpp_matrix[i, j]
                via_k     = bpp_matrix[i, k] + bpp_matrix[k, j]
                violation = abs(direct - via_k)
                max_violation = max(max_violation, violation)
    print(max_violation)
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
if __name__ == '__main__':
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = load_checkpoint_SD_VDM('checkpoints/uqdm-small')
    train_iter, eval_iter = load_data_from_folder()

    # bpps, psnrs = model.evaluate(eval_iter, n_batches=10, seed=seed)
    # bpps = np.array(bpps)
    # psnrs = np.array(psnrs)

    # --- KL matrix on a single batch ---
    sample_batch = next(iter(eval_iter)).to(device)   # grab one batch

    cost_matrix, timesteps = model.compute_cost_matrix(
        iter(eval_iter),       # pass an iterator, not a batch
        seed=seed,
        timestep_stride=1,
        num_images=1,         # average over 50 real images
        use_kl=True,
        cache_path="cache/cost_matrix_stride100.npz"
    )
    # check_triangle_inequality(cost_matrix)

    torch.set_printoptions(precision=2, sci_mode=False)

    optimal_path,  path_timesteps, breakdown = find_optimal_path_dp(cost_matrix, timesteps)
    raw_bpp_optimal = sum(cost_matrix[optimal_path[k], optimal_path[k+1]] for k in range(len(optimal_path)-1))

    print(f"{breakdown['optimal_total_bpp']:.6f} bpp", breakdown['consecutive_bpp'], )
    print(f"{breakdown['num_transitions']:.2f} transitions  ", breakdown['saving_bpp'])

    import numpy as np
    np.set_printoptions(precision=3, suppress=True)


    # Check if big skip is really cheaper than sum of small steps
    sum_consecutive = sum(cost_matrix[i,i+1] for i in range(len(timesteps)-1))
    big_skip = cost_matrix[0, -1]
    

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(bpps, psnrs, '-', label='SD', linewidth=2, color='orange')
    # ax.set_xlabel('Rate (BPP)', fontsize=13)
    # ax.set_ylabel('PSNR (dB)', fontsize=13)
    # ax.set_title('SD 512*512: Rate-Distortion Curve', fontsize=14)
    # ax.legend(fontsize=12, loc='lower right')
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('sd_rd_curve_512.png', dpi=150, bbox_inches='tight')
    # print("Saved plot to sd_rd_curve_512.png")

