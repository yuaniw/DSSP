from typing import Dict, Optional

import copy
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from termcolor import cprint

from dssp.common.model_util import print_params
from dssp.common.pytorch_util import dict_apply
from dssp.model.common.normalizer import LinearNormalizer
from dssp.model.diffusion.dssp_denoiser import (
    DSSPDenoiser,
)
from dssp.model.diffusion.dssp_history_encoder import (
    DSSPHistoryEncoder,
)
from dssp.model.diffusion.mask_generator import LowdimMaskGenerator
from dssp.model.vision.pointnet_extractor import PointCloudObservationEncoder
from dssp.policy.base_policy import BasePolicy


class DSSPBase(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        n_groups=8,
        condition_type="film",
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        mamba_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        d_intermediate=0,
        use_rmsnorm=True,
        residual_in_fp32=False,
        fused_add_norm=False,
        d_model=512,
        mamba_version="Mamba1",
        use_mamba_obs_encoder=True,
        obs_encoder_mamba_layers=2,
        obs_encoder_d_model=512,
        obs_encoder_d_state=64,
        obs_encoder_d_conv=4,
        obs_encoder_expand=2,
        obs_encoder_d_intermediate=0,
        trajectory_position_batch_size=256,
        trajectory_anchor_sampling="uniform",
        train_max_history_steps=None,
        eval_max_history_steps=None,
        eval_use_history_latents=False,
        **kwargs,
    ):
        super().__init__()

        if not obs_as_global_cond:
            raise ValueError("dssp_base only supports obs_as_global_cond=True in RoboTwin.")
        if not use_mamba_obs_encoder:
            raise ValueError("dssp_base requires use_mamba_obs_encoder=True.")

        self.condition_type = condition_type

        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        obs_encoder = PointCloudObservationEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        obs_prefix_encoder = DSSPHistoryEncoder(
            input_dim=obs_feature_dim,
            d_model=obs_encoder_d_model,
            mamba_layers=obs_encoder_mamba_layers,
            d_state=obs_encoder_d_state,
            d_conv=obs_encoder_d_conv,
            expand=obs_encoder_expand,
            d_intermediate=obs_encoder_d_intermediate,
            mamba_version=mamba_version,
            use_rmsnorm=use_rmsnorm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
        )

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.trajectory_position_batch_size = int(trajectory_position_batch_size)
        if self.trajectory_position_batch_size <= 0:
            raise ValueError(
                f"trajectory_position_batch_size must be positive, got {self.trajectory_position_batch_size}."
            )
        self.trajectory_anchor_sampling = trajectory_anchor_sampling
        self.train_max_history_steps = self._normalize_history_limit(train_max_history_steps)
        self.eval_max_history_steps = self._normalize_history_limit(eval_max_history_steps)
        cprint(
            f"[DSSP] use_pc_color: {self.use_pc_color}",
            "yellow",
        )
        cprint(
            f"[DSSP] pointnet_type: {self.pointnet_type}",
            "yellow",
        )
        cprint(
            f"[DSSP] mamba_version: {mamba_version}",
            "yellow",
        )
        cprint(
            f"[DSSP] encoder prefix: last {n_obs_steps} Mamba obs tokens",
            "yellow",
        )

        model = DSSPDenoiser(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            mamba_layers=mamba_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            d_intermediate=d_intermediate,
            mamba_version=mamba_version,
            condition_type=condition_type,
            use_rmsnorm=use_rmsnorm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            **kwargs,
        )

        self.obs_encoder = obs_encoder
        self.obs_prefix_encoder = obs_prefix_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.streaming_eval_enabled = eval_use_history_latents
        self.streaming_eval_per_obs = eval_use_history_latents
        self.eval_obs_latents = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        condition_data_pc=None,
        condition_mask_pc=None,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
                n_obs_steps=self.n_obs_steps,
            )
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        return nobs

    def _encode_obs_latents(self, nobs, max_obs_steps=None):
        if max_obs_steps is None:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            value = next(iter(nobs.values()))
            batch_size, obs_steps = value.shape[:2]
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :max_obs_steps, ...].reshape(-1, *x.shape[2:]))
            value = next(iter(nobs.values()))
            batch_size = value.shape[0]
            obs_steps = max_obs_steps

        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, obs_steps, -1)

    def _normalize_history_limit(self, max_history_steps):
        if max_history_steps is None:
            return None
        max_history_steps = int(max_history_steps)
        if max_history_steps < 0:
            return None
        if max_history_steps == 0:
            raise ValueError("max_history_steps must be positive, null, or negative for full history.")
        return max_history_steps

    def _truncate_history(self, obs_latents: torch.Tensor, max_history_steps: Optional[int]):
        if max_history_steps is None or obs_latents.shape[1] <= max_history_steps:
            return obs_latents
        return obs_latents[:, -max_history_steps:, :]

    def _gather_history_windows(
        self,
        obs_latents: torch.Tensor,
        current_indices: torch.Tensor,
        history_steps: int,
    ):
        history_offsets = torch.arange(
            history_steps,
            device=obs_latents.device,
            dtype=torch.long,
        )
        history_indices = (
            current_indices.unsqueeze(1) - (history_steps - 1) + history_offsets.unsqueeze(0)
        ).clamp(min=0)
        return obs_latents[history_indices]

    def _get_last_obs_prefix_tokens(self, encoded_obs_latents: torch.Tensor):
        num_latents = encoded_obs_latents.shape[1]
        if num_latents >= self.n_obs_steps:
            return encoded_obs_latents[:, -self.n_obs_steps :, :]
        pad = encoded_obs_latents[:, :1, :].expand(-1, self.n_obs_steps - num_latents, -1)
        return torch.cat([pad, encoded_obs_latents], dim=1)

    def _build_global_condition_from_latents(
        self,
        obs_latents: torch.Tensor,
        max_history_steps: Optional[int] = None,
    ):
        obs_latents = self._truncate_history(obs_latents, max_history_steps=max_history_steps)
        encoded_obs_latents = self.obs_prefix_encoder(obs_latents)
        return self._get_last_obs_prefix_tokens(encoded_obs_latents)

    def _encode_global_condition(self, nobs, batch_size):
        nobs_features = self._encode_obs_latents(nobs, max_obs_steps=self.n_obs_steps)
        return self._build_global_condition_from_latents(nobs_features)

    def reset_eval_streaming_state(self):
        self.eval_obs_latents = None

    def advance_eval_streaming_state(self, obs_dict: Dict[str, torch.Tensor]):
        if not self.streaming_eval_enabled:
            return None
        nobs = self._normalize_obs(obs_dict)
        new_latents = self._encode_obs_latents(nobs)
        if self.eval_obs_latents is None:
            self.eval_obs_latents = new_latents
        else:
            self.eval_obs_latents = torch.cat([self.eval_obs_latents, new_latents], dim=1)
        self.eval_obs_latents = self._truncate_history(
            self.eval_obs_latents,
            max_history_steps=self.eval_max_history_steps,
        )
        return self.eval_obs_latents

    def _get_eval_global_condition(self, obs_dict: Dict[str, torch.Tensor]):
        if not self.streaming_eval_enabled:
            nobs = self._normalize_obs(obs_dict)
            value = next(iter(nobs.values()))
            batch_size = value.shape[0]
            return self._encode_global_condition(nobs, batch_size), batch_size
        if self.eval_obs_latents is None:
            nobs = self._normalize_obs(obs_dict)
            self.eval_obs_latents = self._encode_obs_latents(nobs)
        obs_latents = self.eval_obs_latents
        batch_size = obs_latents.shape[0]
        global_cond = self._build_global_condition_from_latents(
            obs_latents,
            max_history_steps=self.eval_max_history_steps,
        )
        return global_cond, batch_size

    def _sample_trajectory_window_starts(self, window_starts: torch.Tensor):
        if window_starts.numel() == 0:
            raise ValueError("Trajectory contains no valid window starts.")
        if self.trajectory_anchor_sampling != "uniform":
            raise ValueError(
                f"Unsupported trajectory_anchor_sampling: {self.trajectory_anchor_sampling}"
            )

        num_windows = window_starts.numel()
        sample_size = min(num_windows, self.trajectory_position_batch_size)
        if sample_size == num_windows:
            perm = torch.randperm(num_windows, device=window_starts.device)
            return window_starts[perm]

        perm = torch.randperm(num_windows, device=window_starts.device)[:sample_size]
        return window_starts[perm]

    def _build_history_global_condition(
        self,
        obs_latents: torch.Tensor,
        current_indices: torch.Tensor,
        max_history_steps: Optional[int],
    ):
        if max_history_steps is None:
            encoded_obs_latents = self.obs_prefix_encoder(obs_latents.unsqueeze(0)).squeeze(0)
            prefix_offsets = torch.arange(self.n_obs_steps, device=obs_latents.device, dtype=torch.long)
            prefix_indices = (
                current_indices.unsqueeze(1) - (self.n_obs_steps - 1) + prefix_offsets.unsqueeze(0)
            ).clamp(min=0)
            return encoded_obs_latents[prefix_indices]

        history_windows = self._gather_history_windows(
            obs_latents=obs_latents,
            current_indices=current_indices,
            history_steps=max_history_steps,
        )
        encoded_history_windows = self.obs_prefix_encoder(history_windows)
        return self._get_last_obs_prefix_tokens(encoded_history_windows)

    def _get_diffusion_target(self, trajectory, noise, timesteps):
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise
        if pred_type == "sample":
            return trajectory
        if pred_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            return alpha_t * noise - sigma_t * trajectory
        raise ValueError(f"Unsupported prediction type {pred_type}")

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)

        value = next(iter(nobs.values()))
        batch_size, _ = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_steps = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        global_cond = self._encode_global_condition(nobs, batch_size)
        cond_data = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def predict_action_streaming(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.streaming_eval_enabled:
            return self.predict_action(obs_dict)
        global_cond, batch_size = self._get_eval_global_condition(obs_dict)

        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss_trajectory(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        window_starts = batch["window_starts"]

        if nactions.shape[0] != 1:
            raise ValueError(
                f"compute_loss_trajectory expects dataloader batch_size=1, got {nactions.shape[0]}."
            )

        obs_latents = self._encode_obs_latents(nobs).squeeze(0)
        actions = nactions.squeeze(0)
        window_starts = window_starts.squeeze(0).to(device=actions.device, dtype=torch.long)
        sampled_window_starts = self._sample_trajectory_window_starts(window_starts)

        episode_length = actions.shape[0]
        window_offsets = torch.arange(self.horizon, device=actions.device, dtype=torch.long)
        window_indices = sampled_window_starts.unsqueeze(1) + window_offsets.unsqueeze(0)
        window_indices = window_indices.clamp(0, episode_length - 1)
        trajectory = actions[window_indices]

        current_indices = sampled_window_starts + (self.n_obs_steps - 1)
        current_indices = current_indices.clamp(0, episode_length - 1)
        global_cond = self._build_history_global_condition(
            obs_latents=obs_latents,
            current_indices=current_indices,
            max_history_steps=self.train_max_history_steps,
        )

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
            n_obs_steps=self.n_obs_steps,
        )

        target = self._get_diffusion_target(trajectory, noise, timesteps)
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        history_lengths = current_indices + 1
        if self.train_max_history_steps is not None:
            history_lengths = history_lengths.clamp(max=self.train_max_history_steps)
        return loss, {
            "bc_loss": loss.item(),
            "trajectory_position_batch_size": float(sampled_window_starts.shape[0]),
            "trajectory_mean_history_steps": float(history_lengths.float().mean().item()),
        }

    def compute_loss(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory
        global_cond = self._encode_global_condition(nobs, batch_size)

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
            n_obs_steps=self.n_obs_steps,
        )
        target = self._get_diffusion_target(trajectory, noise, timesteps)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        return loss, {"bc_loss": loss.item()}
