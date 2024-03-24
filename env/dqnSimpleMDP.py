import numpy as np
LEFT = 1
RIGHT = 0


class dqnSimpleMDP:

    def __init__(self, mu=-0.1, std=1, max_actions=10):
        """
            Simple mdp:
        """
        self.mu = mu
        self.std = std
        self.max_actions = max_actions

        # Define actions available for each state
        self.state_actions = {
            'A': [RIGHT, LEFT],
            'B': [i for i in range(max_actions)],
            'C': [RIGHT],
            'D': [LEFT]}

        self.state = 'A'

    def step(self, action):
        assert (action in self.state_actions[self.state])
        reward = 0
        if self.state == 'A':
            self.state = 'C' if action == RIGHT else 'B'
        elif self.state == 'B':
            self.state = 'D'
            reward = np.random.normal(self.mu, self.std)
        done = True if self.state == 'D' or self.state == 'C' else False
        return self.state, reward, done, None

    def get_possible_actions(self, state=None):
        if state is None:
            return self.state_actions[self.state]
        else:
            return self.state_actions[state]

    def sample_actions(self):
        return np.random.choice(self.get_possible_actions())

    def reset(self):
        self.state = 'A'
        return self.state