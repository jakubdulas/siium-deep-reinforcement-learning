import numpy as np
MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
}

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class frozenLake:

    def __init__(self, type):
        """
            Frozen Lake mdp:
        """

        assert type in ["4x4", "8x8"]

        self.type = type
        self._states = [i for i in range(len(MAPS[self.type][0]) * len(MAPS[self.type]))]
        self._initial_state = 0
        self._current_state = self._initial_state
        self._n_states = len(self._states)
        self.holes = []
        for line in MAPS[self.type]:
            for cell in line:
                if cell == 'H':
                    self.holes.append(1)
                else:
                    self.holes.append(0)
        self.goal = []
        for line in MAPS[self.type]:
            for cell in line:
                if cell == 'G':
                    self.goal.append(1)
                else:
                    self.goal.append(0)

        self.player = []
        for line in MAPS[self.type]:
            for cell in line:
                if cell == 'S':
                    self.player.append(1)
                else:
                    self.player.append(0)

    def reset(self):
        """ resets state of the environment """
        self._current_state = self._initial_state
        self.player = []
        for line in MAPS[self.type]:
            for cell in line:
                if cell == 'S':
                    self.player.append(1)
                else:
                    self.player.append(0)
        return [self.holes, self.player, self.goal]

    def get_all_states(self):
        """ return a list of all possible states """
        return self._states

    def is_terminal(self, state):
        """ return true if state is terminal or false otherwise """
        x, y = self.__state_to_xy(state)
        if MAPS[self.type][y][x] in ['G', 'H']:
            return True
        return False

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        return [LEFT, DOWN, RIGHT, UP]

    def get_number_of_states(self):
        return self._n_states

    # def get_next_states(self, state, action):
    #     """ return a set of possible next states and probabilities of moving into them """
    #     assert action in self.get_possible_actions(
    #         state), "cannot do action %s from state %s" % (action, state)
    #     return self._transition_probs[state][action]

    def get_reward(self, state, action, next_state):
        """ return the reward after taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        x, y = self.__state_to_xy(next_state)
        return 1 if MAPS[self.type][y][x] == 'G' else 0

    def step(self, action):
        current_x, current_y = self.__state_to_xy(self._current_state)
        if action == LEFT and current_x > 0:
            current_x -= 1
        elif action == RIGHT and current_x < len(MAPS[self.type][0]) - 1:
            current_x += 1
        elif action == UP and current_y > 0:
            current_y -= 1
        elif action == DOWN and current_y < len(MAPS[self.type]) - 1:
            current_y += 1

        prev_state = self._current_state
        self._current_state = current_y * len(MAPS[self.type][0]) + current_x
        self.player = np.zeros(self._n_states)
        self.player[self._current_state] = 1
        return [self.holes, self.player, self.goal], self.get_reward(prev_state, action, self._current_state), \
               self.is_terminal(self._current_state), None

    def __state_to_xy(self, state):
        return state % len(MAPS[self.type][0]), state // len(MAPS[self.type][0])
