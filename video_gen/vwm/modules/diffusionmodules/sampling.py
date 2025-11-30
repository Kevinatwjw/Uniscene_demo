"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, List, Union

import numpy as np
import torch
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from torch.distributions import LogisticNormal
from tqdm import tqdm
from vwm.modules.diffusionmodules.sampling_utils import to_d
from vwm.util import append_dims, default, instantiate_from_config

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]):
    """Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def mean_flat(tensor: torch.Tensor, mask=None):
    """Take the mean over all non-batch dimensions."""
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """Compute training losses for a single timestep.

        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """Compatible with diffusers add_noise()"""
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise


####################################------------------------------------------------------------##################
class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(guider_config)
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, cond_mask, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, cond_mask, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling Setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EulerEDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
        reverse=False,
    ):
        if reverse == False:
            x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
            replace_cond_frames = cond_mask is not None and cond_mask.any()
            for i in self.get_sigma_gen(num_sigmas):
                if replace_cond_frames:
                    x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                    # if x.shape == cond_mask.shape:
                    #     x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                    # else:
                    #     cond_mask = repeat(cond_mask, "b -> (b t)", t=6)
                    #     x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                    #     embed()
                gamma = (
                    min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                    if self.s_tmin <= sigmas[i] <= self.s_tmax
                    else 0.0
                )
                x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], denoiser, x, cond, cond_mask, uc, gamma)
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
            return x

    def a__call__2(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
        reverse=False,
    ):
        if reverse == False:
            x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
            replace_cond_frames = cond_mask is not None and cond_mask.any()

            for i in self.get_sigma_gen(num_sigmas):
                if replace_cond_frames:
                    x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                    # if x.shape == cond_mask.shape:
                    #     x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                    # else:
                    #     cond_mask = repeat(cond_mask, "b -> (b t)", t=6)
                    #     x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                    #     embed()
                gamma = (
                    min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                    if self.s_tmin <= sigmas[i] <= self.s_tmax
                    else 0.0
                )
                x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], denoiser, x, cond, cond_mask, uc, gamma)
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
            return x

        else:
            x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
            replace_cond_frames = cond_mask is not None and cond_mask.any()
            for i in self.get_sigma_gen(num_sigmas):
                if replace_cond_frames:
                    x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
                gamma = (
                    min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                    if self.s_tmin <= sigmas[i] <= self.s_tmax
                    else 0.0
                )
                x = self.sampler_step(s_in * sigmas[i], s_in * sigmas[i + 1], denoiser, x, cond, cond_mask, uc, gamma)
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)
            return x

    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        ddim_scheduler,
    ):
        timestep, next_timestep = (
            min(timestep - ddim_scheduler.config.num_train_timesteps // self.diff_timestep, 999),
            timestep,
        )  ## ddim_scheduler.num_inference_steps
        alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
        alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    @torch.no_grad()
    def ddim_inversion(self, batch, num_inv_steps=50):
        device = batch["images"].device
        prompt_embds = []
        for prompt in batch["prompt"]:
            prompt_embds.append(self.encode_text(prompt, device)[0])
        latent = self.encode_image(batch["images"], self.vae)
        prompt_embds = torch.stack(prompt_embds, dim=1)
        all_latent = [latent]
        latent = latent.clone().detach().to(device)
        for i in tqdm(range(num_inv_steps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1].long().to(device)
            denoise, occ_out = self.mv_base_model(latent, t, prompt_embds, occ_data=batch["voxel_data"])
            latent = self.next_step(denoise, t, latent, self.scheduler)
            all_latent.append(latent)
        return all_latent
