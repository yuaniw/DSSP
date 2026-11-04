from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from dssp.model.diffusion.latent_forward_dynamics import (
    LatentForwardDynamicsMLP,
)
from dssp.model.diffusion.latent_progress_mlp import (
    LatentProgressMLP,
)
from dssp.policy.dssp_base import DSSPBase


class DSSP(DSSPBase):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
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
        recent_obs_steps=3,
        use_dynamics_aux_loss=True,
        dynamics_loss_weight=0.05,
        dynamics_loss_type="cosine",
        dynamics_hidden_dim=256,
        use_progress_aux_loss=True,
        progress_loss_weight=0.05,
        progress_hidden_dim=256,
        progress_predict_mode="context_token",
        eval_recent_obs_latent_noise_sigma=0.0,
        eval_recent_obs_latent_noise_steps=3,
        **kwargs,
    ):
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
            condition_type=condition_type,
            encoder_output_dim=encoder_output_dim,
            crop_shape=crop_shape,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            mamba_layers=mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            d_intermediate=d_intermediate,
            use_rmsnorm=use_rmsnorm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            d_model=d_model,
            mamba_version=mamba_version,
            use_mamba_obs_encoder=use_mamba_obs_encoder,
            obs_encoder_mamba_layers=obs_encoder_mamba_layers,
            obs_encoder_d_model=obs_encoder_d_model,
            obs_encoder_d_state=obs_encoder_d_state,
            obs_encoder_d_conv=obs_encoder_d_conv,
            obs_encoder_expand=obs_encoder_expand,
            obs_encoder_d_intermediate=obs_encoder_d_intermediate,
            trajectory_position_batch_size=trajectory_position_batch_size,
            trajectory_anchor_sampling=trajectory_anchor_sampling,
            train_max_history_steps=train_max_history_steps,
            eval_max_history_steps=eval_max_history_steps,
            eval_use_history_latents=eval_use_history_latents,
            **kwargs,
        )

        if recent_obs_steps <= 0:
            raise ValueError(f"recent_obs_steps must be positive, got {recent_obs_steps}.")
        if recent_obs_steps > n_obs_steps:
            raise ValueError(
                f"recent_obs_steps must be <= n_obs_steps, got {recent_obs_steps} > {n_obs_steps}."
            )

        self.recent_obs_steps = recent_obs_steps
        self.use_dynamics_aux_loss = bool(use_dynamics_aux_loss)
        self.dynamics_loss_weight = float(dynamics_loss_weight)
        self.dynamics_loss_type = dynamics_loss_type
        self.use_progress_aux_loss = bool(use_progress_aux_loss)
        self.progress_loss_weight = float(progress_loss_weight)
        self.progress_predict_mode = progress_predict_mode
        self.eval_recent_obs_latent_noise_sigma = float(eval_recent_obs_latent_noise_sigma)
        self.eval_recent_obs_latent_noise_steps = int(eval_recent_obs_latent_noise_steps)
        self.dynamics_predict_mode = kwargs.get("dynamics_predict_mode", "context_to_raw")
        if self.dynamics_predict_mode not in ["raw_to_raw", "context_to_context", "context_to_raw"]:
            raise ValueError(f"Unsupported dynamics_predict_mode: {self.dynamics_predict_mode}")
        if self.progress_predict_mode not in ["context_token", "raw_pair"]:
            raise ValueError(f"Unsupported progress_predict_mode: {self.progress_predict_mode}")

        if self.dynamics_loss_type not in {"cosine", "mse"}:
            raise ValueError(
                f"Unsupported dynamics_loss_type: {self.dynamics_loss_type}. Expected one of ['cosine', 'mse']."
            )

        self.forward_dynamics = LatentForwardDynamicsMLP(
            obs_feature_dim=self.obs_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=int(dynamics_hidden_dim),
        )
        progress_input_dim = self.obs_feature_dim
        if self.progress_predict_mode == "raw_pair":
            progress_input_dim = 2 * self.obs_feature_dim
        self.progress_regressor = LatentProgressMLP(
            input_dim=progress_input_dim,
            hidden_dim=int(progress_hidden_dim),
        )

        cprint(
            (
                "[DSSP] encoder prefix: "
                f"1 last Mamba token + {self.recent_obs_steps} recent raw latents"
            ),
            "yellow",
        )
        cprint(
            (
                "[DSSP] aux losses: "
                f"dynamics={self.use_dynamics_aux_loss}, progress={self.use_progress_aux_loss}, "
                f"progress_mode={self.progress_predict_mode}"
            ),
            "yellow",
        )



    def _encode_mamba_obs_latents(self, obs_latents: torch.Tensor):
        return self.obs_prefix_encoder(obs_latents)

    def _get_recent_raw_tokens(self, obs_latents: torch.Tensor):
        num_latents = obs_latents.shape[1]
        if num_latents >= self.recent_obs_steps:
            return obs_latents[:, -self.recent_obs_steps :, :]
        pad = obs_latents[:, :1, :].expand(-1, self.recent_obs_steps - num_latents, -1)
        return torch.cat([pad, obs_latents], dim=1)

    def _gather_recent_raw_tokens(
        self,
        obs_latents: torch.Tensor,
        current_indices: torch.Tensor,
    ):
        recent_offsets = torch.arange(
            self.recent_obs_steps,
            device=obs_latents.device,
            dtype=torch.long,
        )
        recent_indices = (
            current_indices.unsqueeze(1) - (self.recent_obs_steps - 1) + recent_offsets.unsqueeze(0)
        ).clamp(min=0)
        return obs_latents[recent_indices]

    def _compose_global_condition(
        self,
        current_mamba_tokens: torch.Tensor,
        recent_raw_tokens: torch.Tensor,
    ):
        return torch.cat([current_mamba_tokens.unsqueeze(1), recent_raw_tokens], dim=1)

    def _add_recent_obs_latent_noise(self, obs_latents: torch.Tensor) -> torch.Tensor:
        if self.training or self.eval_recent_obs_latent_noise_sigma == 0.0:
            return obs_latents
        recent_steps = min(self.eval_recent_obs_latent_noise_steps, obs_latents.shape[1])
        noisy_obs_latents = obs_latents.clone()
        noisy_obs_latents[:, -recent_steps:, :] = noisy_obs_latents[:, -recent_steps:, :] + (
            torch.randn_like(noisy_obs_latents[:, -recent_steps:, :]) * self.eval_recent_obs_latent_noise_sigma
        )
        return noisy_obs_latents

    def _build_global_condition_from_latents(
        self,
        obs_latents: torch.Tensor,
        max_history_steps: Optional[int] = None,
    ):
        obs_latents = self._truncate_history(obs_latents, max_history_steps=max_history_steps)
        noisy_obs_latents = self._add_recent_obs_latent_noise(obs_latents)
        encoded_obs_latents = self._encode_mamba_obs_latents(noisy_obs_latents)
        current_mamba_tokens = encoded_obs_latents[:, -1, :]
        recent_raw_tokens = self._get_recent_raw_tokens(noisy_obs_latents)
        return self._compose_global_condition(current_mamba_tokens, recent_raw_tokens)

    def _gather_current_mamba_tokens(
        self,
        obs_latents: torch.Tensor,
        current_indices: torch.Tensor,
        max_history_steps: Optional[int],
        encoded_history_latents: Optional[torch.Tensor] = None,
    ):
        if max_history_steps is None:
            if encoded_history_latents is None:
                encoded_history_latents = self._encode_mamba_obs_latents(obs_latents.unsqueeze(0)).squeeze(0)
            return encoded_history_latents[current_indices]

        history_windows = self._gather_history_windows(
            obs_latents=obs_latents,
            current_indices=current_indices,
            history_steps=max_history_steps,
        )
        encoded_history = self._encode_mamba_obs_latents(history_windows)
        return encoded_history[:, -1, :]

    def _build_history_global_condition(
        self,
        obs_latents: torch.Tensor,
        current_indices: torch.Tensor,
        max_history_steps: Optional[int],
        encoded_history_latents: Optional[torch.Tensor] = None,
    ):
        if max_history_steps is None:
            current_mamba_tokens = self._gather_current_mamba_tokens(
                obs_latents=obs_latents,
                current_indices=current_indices,
                max_history_steps=None,
                encoded_history_latents=encoded_history_latents,
            )
            recent_raw_tokens = self._gather_recent_raw_tokens(
                obs_latents=obs_latents,
                current_indices=current_indices,
            )
            return self._compose_global_condition(current_mamba_tokens, recent_raw_tokens)

        history_windows = self._gather_history_windows(
            obs_latents=obs_latents,
            current_indices=current_indices,
            history_steps=max_history_steps,
        )
        encoded_history_windows = self._encode_mamba_obs_latents(history_windows)
        current_mamba_tokens = encoded_history_windows[:, -1, :]
        recent_raw_tokens = self._gather_recent_raw_tokens(
            obs_latents=obs_latents,
            current_indices=current_indices,
        )
        return self._compose_global_condition(current_mamba_tokens, recent_raw_tokens)

    def _build_progress_inputs(
        self,
        obs_latents: torch.Tensor,
        current_indices: torch.Tensor,
        max_history_steps: Optional[int],
        encoded_history_latents: Optional[torch.Tensor] = None,
        current_mamba_tokens: Optional[torch.Tensor] = None,
    ):
        if self.progress_predict_mode == "context_token":
            if current_mamba_tokens is None:
                current_mamba_tokens = self._gather_current_mamba_tokens(
                    obs_latents=obs_latents,
                    current_indices=current_indices,
                    max_history_steps=max_history_steps,
                    encoded_history_latents=encoded_history_latents,
                )
            return current_mamba_tokens
        first_raw_latent = obs_latents[:1].expand(current_indices.shape[0], -1)
        current_raw_latent = obs_latents[current_indices]
        return torch.cat([first_raw_latent, current_raw_latent], dim=-1)

    def _build_eval_progress_inputs(
        self,
        obs_latents: torch.Tensor,
        current_mamba_tokens: torch.Tensor,
    ):
        if self.progress_predict_mode == "context_token":
            return current_mamba_tokens

        first_raw_latent = obs_latents[:, 0, :]
        current_raw_latent = obs_latents[:, -1, :]
        return torch.cat([first_raw_latent, current_raw_latent], dim=-1)

    def _compute_dynamics_loss(
        self,
        obs_latents: torch.Tensor,
        actions: torch.Tensor,
        encoded_history_latents: Optional[torch.Tensor] = None,
    ):
        if obs_latents.shape[0] < 2:
            raise ValueError(
                f"Dynamics auxiliary loss expects at least 2 observations per episode, got {obs_latents.shape[0]}."
            )
        
        L = obs_latents.shape[0]
        current_actions = actions[:-1]
        if self.dynamics_predict_mode == "raw_to_raw":
            #  Raw Latent + Action -> Next Raw Latent
            current_latents = obs_latents[:-1]
            next_target_latents = obs_latents[1:].detach()
        elif self.dynamics_predict_mode == "context_to_context":
            # Context Latent + Action -> Next Context Latent
            all_indices = torch.arange(L, device=obs_latents.device, dtype=torch.long)
            all_context_latents = self._gather_current_mamba_tokens(
                obs_latents=obs_latents,
                current_indices=all_indices,
                max_history_steps=self.train_max_history_steps,
                encoded_history_latents=encoded_history_latents,
            )
            current_latents = all_context_latents[:-1]
            next_target_latents = all_context_latents[1:].detach()
        elif self.dynamics_predict_mode == "context_to_raw":
            # Context Latent + Action -> Next Raw Latent
            current_indices = torch.arange(L - 1, device=obs_latents.device, dtype=torch.long)
            current_latents = self._gather_current_mamba_tokens(
                obs_latents=obs_latents,
                current_indices=current_indices,
                max_history_steps=self.train_max_history_steps,
                encoded_history_latents=encoded_history_latents,
            )
            next_target_latents = obs_latents[1:].detach()
        else:
            raise ValueError(f"Unsupported dynamics_predict_mode: {self.dynamics_predict_mode}")
        pred_next_obs_latents = self.forward_dynamics(current_latents, current_actions)

        if self.dynamics_loss_type == "cosine":
            return (1.0 - F.cosine_similarity(pred_next_obs_latents, next_target_latents, dim=-1)).mean()
        if self.dynamics_loss_type == "mse":
            return F.mse_loss(pred_next_obs_latents, next_target_latents, reduction="mean")
        raise ValueError(f"Unsupported dynamics_loss_type: {self.dynamics_loss_type}")

    def _compute_progress_loss(
        self,
        progress_inputs: torch.Tensor,
        current_indices: torch.Tensor,
        episode_length: int,
    ):
        pred_progress = self.progress_regressor(progress_inputs).squeeze(-1)
        progress_target = current_indices.to(dtype=pred_progress.dtype) / float(episode_length)
        return F.l1_loss(pred_progress, progress_target, reduction="mean")








    def compute_loss_trajectory(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        window_starts = batch["window_starts"]

        if nactions.shape[0] != 1:
            raise ValueError(
                f"compute_loss_trajectory expects dataloader batch_size=1, got {nactions.shape[0]}."
            )

        obs_latents = self._encode_obs_latents(nobs).squeeze(0)
        encoded_history_latents = None
        if self.train_max_history_steps is None:
            encoded_history_latents = self._encode_mamba_obs_latents(obs_latents.unsqueeze(0)).squeeze(0)

        actions = nactions.squeeze(0)
        window_starts = window_starts.squeeze(0).to(device=actions.device, dtype=torch.long)
        sampled_window_starts = self._sample_trajectory_window_starts(window_starts)

        episode_length = int(batch["episode_length"].squeeze(0).item())
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
            encoded_history_latents=encoded_history_latents,
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
        bc_loss = F.mse_loss(pred, target, reduction="none")
        bc_loss = bc_loss * loss_mask.type(bc_loss.dtype)
        bc_loss = reduce(bc_loss, "b ... -> b (...)", "mean")
        bc_loss = bc_loss.mean()

        dynamics_loss = bc_loss.new_zeros(())
        if self.use_dynamics_aux_loss:
            dynamics_loss = self._compute_dynamics_loss(
                obs_latents=obs_latents,
                actions=actions,
                encoded_history_latents=encoded_history_latents,
            )

        progress_loss = bc_loss.new_zeros(())
        if self.use_progress_aux_loss:
            current_mamba_tokens = self._gather_current_mamba_tokens(
                obs_latents=obs_latents,
                current_indices=current_indices,
                max_history_steps=self.train_max_history_steps,
                encoded_history_latents=encoded_history_latents,
            )
            progress_inputs = self._build_progress_inputs(
                obs_latents=obs_latents,
                current_indices=current_indices,
                max_history_steps=self.train_max_history_steps,
                encoded_history_latents=encoded_history_latents,
                current_mamba_tokens=current_mamba_tokens,
            )
            progress_loss = self._compute_progress_loss(
                progress_inputs=progress_inputs,
                current_indices=current_indices,
                episode_length=episode_length,
            )

        total_loss = bc_loss
        total_loss = total_loss + (self.dynamics_loss_weight * dynamics_loss)
        total_loss = total_loss + (self.progress_loss_weight * progress_loss)

        history_lengths = current_indices + 1
        if self.train_max_history_steps is not None:
            history_lengths = history_lengths.clamp(max=self.train_max_history_steps)
        return total_loss, {
            "bc_loss": bc_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "progress_loss": progress_loss.item(),
            "total_loss": total_loss.item(),
            "dynamics_loss_weight": self.dynamics_loss_weight,
            "progress_loss_weight": self.progress_loss_weight,
            "trajectory_position_batch_size": float(sampled_window_starts.shape[0]),
            "trajectory_mean_history_steps": float(history_lengths.float().mean().item()),
        }

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_action(obs_dict)

    def predict_action_streaming(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super().predict_action_streaming(obs_dict)

    def compute_loss(self, batch):
        return super().compute_loss(batch)
