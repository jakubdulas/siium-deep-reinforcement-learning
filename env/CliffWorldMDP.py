MAP = [
    "FFFFFFFFFF",
    "FFFFFFFFFF",
    "FFFFFFFFFF",
    "SHHHHHHHHG"
]

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class CliffWorld:

    def __init__(self):
        """
            Frozen Lake mdp:
        """

        self._states = [i for i in range(len(MAP[0]) * len(MAP))]

        self._initial_state = 0
        for i in range(len(MAP[0]) * len(MAP)):
            if MAP[i // len(MAP[0])][i % len(MAP[0])] == 'S':
                self._initial_state = i

        self._current_state = self._initial_state
        self._n_states = len(self._states)

    def reset(self):
        """ resets state of the environment """
        self._current_state = self._initial_state
        return self._current_state

    def get_all_states(self):
        """ return a list of all possible states """
        return self._states

    def is_terminal(self, state):
        """ return true if state is terminal or false otherwise """
        x, y = self.__state_to_xy(state)
        if MAP[y][x] in ['G', 'H']:
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
        reward = -1
        if MAP[y][x] == 'G':
            reward = 1
        elif MAP[y][x] == 'H':
            reward = -100

        return reward

    def step(self, action):
        current_x, current_y = self.__state_to_xy(self._current_state)
        if action == LEFT and current_x > 0:
            current_x -= 1
        elif action == RIGHT and current_x < len(MAP[0]) - 1:
            current_x += 1
        elif action == UP and current_y > 0:
            current_y -= 1
        elif action == DOWN and current_y < len(MAP) - 1:
            current_y += 1

        prev_state = self._current_state
        self._current_state = current_y * len(MAP[0]) + current_x
        return self._current_state, self.get_reward(prev_state, action, self._current_state), \
               self.is_terminal(self._current_state), None

    def __state_to_xy(self, state):
        return state % len(MAP[0]), state // len(MAP[0])
