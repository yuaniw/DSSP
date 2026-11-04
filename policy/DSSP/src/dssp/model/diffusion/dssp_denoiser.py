import logging
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mlp import GatedMLP

from dssp.model.diffusion.positional_embedding import SinusoidalPosEmb
from dssp.model.diffusion.rms_norm_compat import RMSNorm

logger = logging.getLogger(__name__)


class AdaLNActionMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        cond_dim,
        mixer_cls,
        mlp_cls,
        norm_cls,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.norm = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        self.mlp = mlp_cls(d_model)

        self.mixer_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 3),
        )
        self.mlp_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 3),
        )

        nn.init.zeros_(self.mixer_modulation[-1].weight)
        nn.init.zeros_(self.mixer_modulation[-1].bias)
        nn.init.zeros_(self.mlp_modulation[-1].weight)
        nn.init.zeros_(self.mlp_modulation[-1].bias)

    def _apply_action_adaln(self, x, cond_embed, prefix_len, modulation):
        if prefix_len >= x.shape[1]:
            return x, None

        shift, scale, gate = modulation(cond_embed).chunk(3, dim=-1)
        action_tokens = x[:, prefix_len:, :]
        action_tokens = action_tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        if prefix_len == 0:
            return action_tokens, gate.unsqueeze(1)
        return torch.cat((x[:, :prefix_len, :], action_tokens), dim=1), gate.unsqueeze(1)

    def _apply_action_gate(self, update, gate, prefix_len):
        if gate is None or prefix_len >= update.shape[1]:
            return update

        action_update = update[:, prefix_len:, :] * gate
        if prefix_len == 0:
            return action_update
        return torch.cat((update[:, :prefix_len, :], action_update), dim=1)

    def _residual_add(self, x, update):
        if self.residual_in_fp32:
            return (x.float() + update.float()).to(dtype=x.dtype)
        return x + update

    def forward(self, x, cond_embed, prefix_len):
        y = self.norm(x)
        y, gate = self._apply_action_adaln(y, cond_embed, prefix_len, self.mixer_modulation)
        y = self.mixer(y)
        y = self._apply_action_gate(y, gate, prefix_len)
        x = self._residual_add(x, y)

        y = self.norm2(x)
        y, gate = self._apply_action_adaln(y, cond_embed, prefix_len, self.mlp_modulation)
        y = self.mlp(y)
        y = self._apply_action_gate(y, gate, prefix_len)
        x = self._residual_add(x, y)
        return x


class DSSPDenoiser(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        mamba_layers=8,
        d_model=512,
        d_state=16,
        d_conv=4,
        expand=2,
        d_intermediate=0,
        mamba_version="Mamba1",
        condition_type="concat",
        condition_token_num=3,
        condition_strategy="prefix",
        condition_fusion="add",
        use_positional_embedding=True,
        use_rmsnorm=True,
        residual_in_fp32=False,
        fused_add_norm=False,
        **kwargs,
    ):
        super().__init__()

        if condition_strategy != "prefix":
            raise ValueError("DSSPDenoiser only supports condition_strategy='prefix'.")
        if fused_add_norm:
            raise ValueError("DSSPDenoiser does not support fused_add_norm.")

        self.condition_type = condition_type
        self.condition_token_num = condition_token_num
        self.condition_strategy = condition_strategy
        self.condition_fusion = condition_fusion
        self.use_positional_embedding = use_positional_embedding
        self.use_rmsnorm = use_rmsnorm

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        if global_cond_dim is not None:
            self.global_cond_proj = nn.Linear(global_cond_dim, d_model)

        self.input_proj = nn.Linear(input_dim, d_model)

        if use_positional_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model))

        self.mamba_layers = nn.ModuleList([])
        for i in range(mamba_layers):
            if mamba_version == "Mamba2":
                mixer_cls = partial(Mamba2, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i)
            else:
                mixer_cls = partial(Mamba, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i)

            if use_rmsnorm:
                norm_cls = partial(RMSNorm, eps=1e-5)
            else:
                norm_cls = partial(nn.LayerNorm, eps=1e-5)

            if d_intermediate == 0:
                mlp_cls = partial(GatedMLP, out_features=d_model)
            else:
                mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model)

            block = AdaLNActionMambaBlock(
                d_model=d_model,
                cond_dim=dsed,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                norm_cls=norm_cls,
                residual_in_fp32=residual_in_fp32,
            )
            self.mamba_layers.append(block)

        if use_rmsnorm:
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, input_dim)

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            self.local_cond_encoder = nn.Sequential(
                nn.Linear(local_cond_dim, d_model),
                Rearrange("b t d -> b d t"),
            )

        logger.info("AdaLN Mamba decoder parameters: %e", sum(p.numel() for p in self.parameters()))
        logger.info("Using %s with d_intermediate=%s", mamba_version, d_intermediate)

    def _add_action_positional_embedding(self, x, prefix_len):
        if not self.use_positional_embedding or prefix_len >= x.shape[1]:
            return x

        action_pos_embedding = self.pos_embedding[:, : x.shape[1] - prefix_len, :]
        action_tokens = x[:, prefix_len:, :] + action_pos_embedding

        if prefix_len == 0:
            return action_tokens
        return torch.cat((x[:, :prefix_len, :], action_tokens), dim=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        local_cond=None,
        global_cond=None,
        n_obs_steps=None,
        **kwargs,
    ):
        batch_size, _, _ = sample.shape

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(batch_size)

        timestep_embed = self.diffusion_step_encoder(timesteps)

        cond_tokens = None
        if global_cond is not None:
            if len(global_cond.shape) == 2:
                global_cond = global_cond.unsqueeze(1)
            elif len(global_cond.shape) != 3:
                raise ValueError(
                    f"global_cond must have shape [B, D] or [B, T, D], got {global_cond.shape}"
                )
            cond_tokens = self.global_cond_proj(global_cond)

        x = self.input_proj(sample)

        if self.local_cond_encoder is not None and local_cond is not None:
            local_cond = self.local_cond_encoder(local_cond).permute(0, 2, 1)
            if self.condition_fusion == "add":
                x = x + local_cond
            elif self.condition_fusion == "concat":
                x = torch.cat([x, local_cond], dim=-1)

        prefix_len = 0
        if cond_tokens is not None:
            x = torch.cat([cond_tokens, x], dim=1)
            prefix_len = cond_tokens.size(1)

        x = self._add_action_positional_embedding(x, prefix_len)

        for mamba_block in self.mamba_layers:
            x = mamba_block(x, timestep_embed, prefix_len)

        x = self.final_norm(x)

        if prefix_len > 0:
            x = x[:, prefix_len:, :]

        x = self.output_proj(x)
        return x
