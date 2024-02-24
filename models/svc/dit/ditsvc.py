# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from modules.diffusion import DiT
from modules.encoder.position_encoder import PositionEncoder


class DiTWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.diff_cfg = cfg.model.diffusion

        self.diff_encoder = PositionEncoder(
            d_raw_emb=self.diff_cfg.step_encoder.dim_raw_embedding,
            d_out=self.diff_cfg.bidilconv.base_channel,
            d_mlp=self.diff_cfg.step_encoder.dim_hidden_layer,
            activation_function=self.diff_cfg.step_encoder.activation,
            n_layer=self.diff_cfg.step_encoder.num_layer,
            max_period=self.diff_cfg.step_encoder.max_period,
        )

        # FIXME: Only support BiDilConv now for debug
        if self.diff_cfg.model_type.lower() == "DiT":
            self.neural_network = DiT(
                **self.diff_cfg.dit,
            )
        else:
            raise ValueError(
                f"Unsupported diffusion model type: {self.diff_cfg.model_type}"
            )

    def forward(self, x, t, c, seq_c, mask=None):
        """
        Args:
            x: [N, T, D] of latent
            t: Diffusion time step with shape of [N]
            c: [N, conditioner_size] of conditioner
            seq_c: [N, T, conditioner_size] of sequence conditioner
            mask: [N, T] of mask

        Returns:
            [N, T, D] of latent
        """

        assert (
            x.size()[:-1] == seq_c.size()[:-1]
        ), "x mismatch with seq_c, got \n x: {} \n c: {}".format(x.size(), c.size())
        assert x.size(0) == t.size(
            0
        ), "x mismatch with t, got \n x: {} \n t: {}".format(x.size(), t.size())
        assert t.dim() == 1, "t must be 1D tensor, got {}".format(t.dim())

        N, T, D = x.size()

        t = self.diff_encoder(t).contiguous()

        h = self.neural_network(x, t, c, seq_c, mask)


        assert h.size() == (
            N,
            T,
            D,
        ), "h mismatch with input x, got \n h: {} \n x: {}".format(
            h.size(), (N, T, D)
        )
        return h


class DiTSVC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.diff_cfg = cfg.model.diffusion

        self.denoise_fn = DiTWrapper(cfg.dit)

        self.P_mean = self.diff_cfg.P_mean
        self.P_std = self.diff_cfg.P_std
        self.sigma_data = self.diff_cfg.sigma_data
        self.sigma_min = self.diff_cfg.sigma_min
        self.sigma_max = self.diff_cfg.sigma_max
        self.rho = self.diff_cfg.rho
        self.N = self.diff_cfg.n_timesteps

         # Time step discretization
        step_indices = torch.arange(self.N)
        # karras boundaries formula
        t_steps = (
            self.sigma_min ** (1 / self.rho)
            + step_indices
            / (self.N - 1)
            * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho
        self.t_steps = torch.cat(
            [torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)]
        )
    
    def EDMPrecond(self, x, sigma, cond, seq_cond, mask, denoise_fn):
        """
        karras diffusion reverse process

        Args:
            x: noisy latent [B x L x D]
            sigma: noise level [B x 1 x 1]
            cond: conditioner [B x D]
            seq_cond: sequence conditioner [B x L x D]
            mask: mask of padded frames [B x L]
            denoise_fn: denoiser neural network e.g. DiT

        Returns:
            denoised latent [B x L x D]
        """
        sigma = sigma.reshape(-1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2).sqrt()
        )
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        x_in = x_in.transpose(1, 2)
        x = x.transpose(1, 2)
        cond = cond.transpose(1, 2)
        c_noise = c_noise.squeeze()
        if c_noise.dim() == 0:
            c_noise = c_noise.unsqueeze(0)
        F_x = denoise_fn(x_in, c_noise, cond, seq_cond, mask)
        D_x = c_skip * x + c_out * (F_x)
        D_x = D_x.transpose(1, 2)
        return D_x

    def EDMLoss(self, x_start, cond, seq_cond, mask):
        """
        compute loss for EDM model

        Args:
            x_start: ground truth mel-spectrogram [B x n_mel x L]
            cond: output of conformer encoder [B x n_mel x L]
            mask: mask of padded frames [B x n_mel x L]
        """
        rnd_normal = torch.randn([x_start.shape[0], 1, 1], device=x_start.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # follow Grad-TTS, start from Gaussian noise with mean cond and std I
        noise = (torch.randn_like(x_start) + cond) * sigma
        D_yn = self.EDMPrecond(x_start + noise, sigma, cond, seq_cond, mask, self.denoise_fn)
        loss = weight * ((D_yn - x_start) ** 2)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def edm_sampler(
        self,
        latents,
        cond,
        seq_cond,
        mask,
        num_steps=50,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """
        karras diffusion sampler

        Args:
            latents: noisy mel-spectrogram [B x n_mel x L]
            cond: output of conformer encoder [B x n_mel x L]
            nonpadding: mask of padded frames [B x n_mel x L]
            num_steps: number of steps for diffusion inference

        Returns:
            denoised mel-spectrogram [B x n_mel x L]
        """
        # Time step discretization.

        num_steps = num_steps + 1
        step_indices = torch.arange(num_steps, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_next = latents * t_steps[0]
        # wrap in tqdm for progress bar
        bar = tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])))
        for i, (t_cur, t_next) in bar:
            x_cur = x_next
            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            t = torch.zeros((x_cur.shape[0], 1, 1), device=x_cur.device)
            t[:, 0, 0] = t_hat
            t_hat = t
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)
            # Euler step.
            denoised = self.EDMPrecond(x_hat, t_hat, cond, seq_cond, mask, self.denoise_fn)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # add Heun’s 2nd order method
            # if i < num_steps - 1:
            #     t = torch.zeros((x_cur.shape[0], 1, 1), device=x_cur.device)
            #     t[:, 0, 0] = t_next
            #     #t_next = t
            #     denoised = self.EDMPrecond(x_next, t, cond, self.denoise_fn, nonpadding)
            #     d_prime = (x_next - denoised) / t_next
            #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def get_t_steps(self, N):
        N = N + 1
        step_indices = torch.arange(N)
        t_steps = (
            self.sigma_min ** (1 / self.rho)
            + step_indices
            / (N - 1)
            * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho

        return t_steps.flip(0)


    def forward(self, x, mask, cond, seq_cond, t_steps=1, infer=False):
        """
        calculate loss or sample latent

        Args:
            x:
                training: ground truth latent [B x L x D]
        """
        if not infer:
            loss = self.EDMLoss(x, cond, seq_cond, mask)
            return loss
        else:
            x = torch.randn(seq_cond.shape, device=x.device)
            x = self.edm_sampler(x, cond, mask, t_steps)

            return x