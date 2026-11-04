import torch
import torch.nn as nn


class LatentForwardDynamicsMLP(nn.Module):
    def __init__(
        self,
        obs_feature_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        input_dim = obs_feature_dim + action_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_feature_dim),
        )

    def forward(self, history_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([history_latent, action], dim=-1))


class LatentFutureStateMLP(nn.Module):
    def __init__(
        self,
        obs_feature_dim: int,
        action_dim: int,
        action_steps: int,
        hidden_dim: int,
        output_steps: int,
    ):
        super().__init__()

        self.obs_feature_dim = int(obs_feature_dim)
        self.action_steps = int(action_steps)
        self.output_steps = int(output_steps)
        input_dim = self.obs_feature_dim + (int(action_dim) * self.action_steps)
        output_dim = self.obs_feature_dim * self.output_steps
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context_latent: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        flat_actions = future_actions.reshape(future_actions.shape[0], -1)
        pred = self.model(torch.cat([context_latent, flat_actions], dim=-1))
        return pred.reshape(context_latent.shape[0], self.output_steps, self.obs_feature_dim)
