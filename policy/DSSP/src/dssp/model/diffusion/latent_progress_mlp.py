import torch
import torch.nn as nn


class LatentProgressMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, progress_input: torch.Tensor) -> torch.Tensor:
        return self.model(progress_input)
