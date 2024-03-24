class simpleMDP:

    def __init__(self):
        """
            Simple mdp:
        """
        self._rewards = {
            's1': {'a0': {'s0': +5}},
            's2': {'a1': {'s0': -1}}
        }
        self._transition_probs = {
              's0': {
                'a0': {'s0': 0.5, 's2': 0.5},
                'a1': {'s2': 1}
              },
              's1': {
                'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
                'a1': {'s1': 0.95, 's2': 0.05}
              },
              's2': {
                'a0': {'s0': 0.4, 's2': 0.6},
                'a1': {'s0': 0.3, 's1': 0.3, 's2':0.4}
              }
        }
        self._states = list(self._transition_probs.keys())
        self._initial_state = "s0"
        self._current_state = self._initial_state
        self._n_states = len(self._transition_probs)

    def reset(self):
        """ resets state of the environment """
        self._current_state = self._initial_state

    def get_all_states(self):
        """ return a list of all possible states """
        return self._states

    def is_terminal(self, state):
        """ return true if state is terminal or false otherwise """
        return False

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        return tuple(self._transition_probs.get(state, {}).keys())

    def get_next_states(self, state, action):
        """ return a set of possible next states and probabilities of moving into them """
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        return self._transition_probs[state][action]

    def get_reward(self, state, action, next_state):
        """ return the reward after taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(
            state), "cannot do action %s from state %s" % (action, state)
        return self._rewards.get(state, {}).get(action, {}).get(next_state, 0.0)

    def step(self, action):
        #TODO
        pass
