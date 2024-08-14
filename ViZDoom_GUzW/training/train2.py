import os
import time
import cv2
import numpy as np
import vizdoom as vzd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium


class VizDoomGym(Env):
    def __init__(self, config, num_actions=3, render=False):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config)
        
        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        self.game.init()
        
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 3), dtype=np.uint8)
        self.action_space = Discrete(num_actions)
        self.num_actions = num_actions
        self.prev_game_vars = None

    def get_reward(self):
        if not self.game.get_state():
            return 0
        
        game_variables = np.array(self.game.get_state().game_variables)
        reward = 0.01

        if self.prev_game_vars is None:
            self.prev_game_vars = game_variables
            return reward
        
        if game_variables[..., 2] <= 0:
            reward -= 1
        
        did_kill = game_variables[..., 0] - self.prev_game_vars[..., 0] > 0
        
        if did_kill:
            reward += 1

        did_damage = game_variables[..., 1] - self.prev_game_vars[..., 1] > 0
        did_loose_health = game_variables[..., 2] - self.prev_game_vars[..., 2] < 0
        did_gain_health = game_variables[..., 2] - self.prev_game_vars[..., 2] > 0
        did_loose_armor = game_variables[..., 3] - self.prev_game_vars[..., 3] < 0
        did_get_armor = game_variables[..., 3] - self.prev_game_vars[..., 3] > 0
        did_shoot = game_variables[..., 6:11].sum() - self.prev_game_vars[..., 6:11].sum() < 0

        if did_damage:
            reward += 0.3
        if did_loose_health:
            reward -= 0.05
        if did_gain_health:
            reward += 0.2
        if did_loose_armor:
            reward -= 0.05
        if did_get_armor:
            reward += 0.2
        if did_shoot:
            reward -= .005
        
        return reward
        
        
    def step(self, action):
        actions = np.identity(self.num_actions)
        self.game.make_action(actions[action], 4)
        reward = self.get_reward()
        # print(reward)
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.scale_down(state)
            killcount = self.game.get_state().game_variables[0]
            info = {"info": killcount}
        else:
            state = np.zeros(self.observation_space.shape)
            info = {"info": 0}
        
        done = self.game.is_episode_finished()
        
        return state, reward, done, False, info
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.game.set_seed(seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        self.prev_game_vars = None

        return self.scale_down(state), {}
    
    def scale_down(self, observation):
        resize = cv2.resize(
            np.moveaxis(observation, 0, -1), (160, 100), interpolation=cv2.INTER_CUBIC
        )
        state = np.reshape(resize, (100, 160, 3))
        return state
    
    def close(self):
        self.game.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 256),  # Additional fully connected layer
            nn.ReLU(),
            nn.Linear(256, features_dim),  # Another fully connected layer
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


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
            nn.Linear(128, self.action_space.n)  # Assuming action_space.n is the output size for policy
        )

        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for value function
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


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True


if __name__ == "__main__":
    CHECKPOINT_DIR = './train/train_basic'
    LOG_DIR = './logs/log_basic'
    TRAIN = True

    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

    env = VizDoomGym("scenarios/deathmatch.cfg", render=not TRAIN)
    model = PPO(CustomActorCriticPolicy, env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048, policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(
                pi=[256, 128, 64],
                vf=[256, 128, 64],
            ),
        ))

    if TRAIN:
        model.learn(total_timesteps=25000, callback=callback)
    else:
        model = PPO.load("train/train_basic/best_model_26000.zip")

    for episode in range(20):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(f'Total Reward for episode {episode} is {total_reward}')
