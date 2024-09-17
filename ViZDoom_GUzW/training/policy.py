from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
        )

        # Define dimensions based on the output of CustomCNN
        features_dim = 64  # This should match the output dimension of CustomCNN

        # Additional fully connected layers
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(
                128, self.action_space.n
            ),  # Assuming action_space.n is the output size for policy
        )

        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Single output for value function
        )


def forward(self, obs, deterministic=False):
    features = self.extract_features(obs)
    latent_pi, latent_vf = self.mlp_extractor(features)

    # Get action distribution from latent policy
    distribution = self._get_action_dist_from_latent(latent_pi)
    values = self.value_net(latent_vf)

    actions = distribution.get_actions(deterministic=deterministic)
    log_prob = distribution.log_prob(actions)
    return actions, values, log_prob


def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    features = self.extract_features(observation)
    latent_pi, _ = self.mlp_extractor(features)
    distribution = self._get_action_dist_from_latent(latent_pi)
    return distribution.get_actions(deterministic=deterministic)
