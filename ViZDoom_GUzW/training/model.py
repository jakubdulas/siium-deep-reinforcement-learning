from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=4, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=4, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.Flatten(),
        )

        # observation_space.shape = (1, 100, 160)

        with th.no_grad():
            # n_flatten = self.cnn1(
            #     th.as_tensor(observation_space.sample()[None]).float()
            # ).shape[1]
            n_flatten = self.cnn1(th.randn((1, 1, 100, 160))).shape[-1]

        self.linear1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.GELU(),
            nn.Linear(features_dim, 64),  # Additional fully connected layer
            nn.GELU(),
            nn.Linear(64, features_dim // 2),  # Another fully connected layer
            nn.GELU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.GELU(),
            nn.Linear(features_dim, 64),  # Additional fully connected layer
            nn.GELU(),
            nn.Linear(64, features_dim // 2),  # Another fully connected layer
            nn.GELU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("observations", observations.shape)
        # plt.imsave("img.png", observations[0][0])
        # plt.imsave("img2.png", observations[0][1])
        x1 = self.linear1(self.cnn1(observations[:, 0:1]))
        x2 = self.linear2(self.cnn2(observations[:, 1:2]))
        return th.concat([x1, x2], dim=-1)
