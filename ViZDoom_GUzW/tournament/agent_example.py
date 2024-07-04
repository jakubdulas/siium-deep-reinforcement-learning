from typing import List, Tuple
import random

class Agent():
    def __init__(self, actions: List[List[int]]) -> None:
        self.actions = actions

    def choose_action(self, game_state) -> Tuple[List[int], int]:
        # do something with game_state
        action_index = random.randrange(len(self.actions))
        return self.actions[action_index], action_index

    def agent_train(self, game_state, action, next_game_state, done):
        # Calculate rewards with game_state.game_variables and next_game_state.game_variables
        # Get a frame image with game_state.screen_buffer and next_game_state.screen_buffer
        pass