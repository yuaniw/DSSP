from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mlp import GatedMLP

from dssp.model.diffusion.rms_norm_compat import RMSNorm


class DSSPHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=512,
        mamba_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        d_intermediate=0,
        mamba_version="Mamba1",
        use_rmsnorm=True,
        residual_in_fp32=False,
        fused_add_norm=False,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

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

            block = Block(
                d_model,
                mixer_cls,
                mlp_cls=mlp_cls,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )
            self.mamba_layers.append(block)

        if use_rmsnorm:
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        if obs_tokens.ndim != 3:
            raise ValueError(f"Expected obs_tokens to have shape [B, T, D], got {obs_tokens.shape}")

        x = self.input_proj(obs_tokens)

        residual = None
        for mamba_block in self.mamba_layers:
            x, residual = mamba_block(x, residual)

        if residual is None:
            residual = x
        else:
            residual = residual + x
        x = self.final_norm(residual.to(dtype=self.final_norm.weight.dtype))

        return self.output_proj(x)
