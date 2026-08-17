"""features_extractor.py — recalé sur InputPreprocessor.cs (11x30x30, 6 stats)."""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

N_GRID_CHANNELS = 11   # ElementCodes: '.','#','t','o','*','+','@','F','M','X','G'
GRID_H, GRID_W = 30, 30
N_SCALARS = 6          # stamina, battery, timeRemaining/maxTime, posX/W, posY/H, isOnEngine


class GridScalarExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(N_GRID_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(N_SCALARS, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.combined = nn.Sequential(nn.Linear(64 + 32, features_dim), nn.ReLU())

    def forward(self, observations):
        grid_feat = self.cnn(observations["grid"]).flatten(1)
        scalar_feat = self.mlp(observations["scalars"])
        return self.combined(torch.cat([grid_feat, scalar_feat], dim=1))