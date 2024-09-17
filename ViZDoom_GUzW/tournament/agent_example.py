from typing import List, Tuple
import cv2
import numpy as np
from stable_baselines3 import PPO


class Agent:
    def __init__(self, actions: List[List[int]]) -> None:
        self.actions = actions
        self.agent = PPO.load(
            "/Users/jakubdulas/Documents/Studia/DRL/ViZDoom_GUzW/training/train_dapth_and_labels/train_deathmatch_0/best_model"
        )

    def choose_action(self, game_state) -> Tuple[List[int], int]:
        if game_state.depth_buffer is not None:
            state1 = self.game.get_state().labels_buffer
            state1 = self.scale_down(state1)
            state2 = self.game.get_state().depth_buffer
            state2 = self.scale_down(state2)
            state = np.concatenate([state1, state2], -1)
            pred, _ = self.agent.predict(state, deterministic=True)

        return self.actions[pred], pred

    def scale_down(self, observation):
        resize = cv2.resize(observation, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state
