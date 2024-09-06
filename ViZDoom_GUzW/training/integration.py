from typing import List, Tuple
import cv2
import numpy as np
from stablebaselines3 import PPO
from ActorCriticPolicy import CustomActorCriticPolicy
from CnnPolicy import CustomCNN


class Agent:
    def init(self, actions: List[List[int]]) -> None:
        self.actions = actions
        self.agent = PPO.load("defendthecenter.zip")

    def choose_action(self, game_state) -> Tuple[List[int], int]:
        if game_state.screen_buffer is not None:
            screen = game_state.screen_buffer
            resize = cv2.resize(
                screen, (160, 100), interpolation=cv2.INTER_CUBIC
            )
            state = np.repeat(resize, 3, axis=2)
            pred,  = self.agent.predict(state, deterministic=True)

        return self.actions[pred], pred
