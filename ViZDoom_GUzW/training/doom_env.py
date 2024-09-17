from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2
import numpy as np
import vizdoom as vzd


class VizDoomGym(Env):
    def __init__(
        self,
        config,
        num_actions=13,
        render=False,
        with_bots=False,
        living_reward=0,
        default_reward_weight=0.0,
    ):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config)

        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)

        if with_bots:
            self.game.set_doom_map("map03")
            self.game.add_game_args("+viz_bots_path scenarios/bots.cfg")
            self.game.add_game_args(
                "-host 1 -deathmatch +timelimit 10.0 "
                "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 cl_run 1"
                "+viz_respawn_delay 0"
            )
            self.game.add_game_args("-deathmatch")
            self.game.set_mode(vzd.Mode.PLAYER)
            self.bots = 7

        self.game.init()

        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 2), dtype=np.uint8
        )
        self.action_space = Discrete(num_actions)
        self.num_actions = num_actions
        self.prev_game_vars = None
        self.living_reward = living_reward
        self.default_reward_weight = default_reward_weight

    def get_reward(self):
        if not self.game.get_state():
            return 0

        game_variables = np.array(self.game.get_state().game_variables)
        reward = self.living_reward

        if self.prev_game_vars is None:
            self.prev_game_vars = game_variables
            return reward

        if game_variables[..., 2] <= 0:
            reward -= 1
            return reward

        did_kill = game_variables[..., 0] - self.prev_game_vars[..., 0] > 0

        if did_kill:
            reward += 1

        did_damage = game_variables[..., 1] - self.prev_game_vars[..., 1] > 0
        did_loose_health = game_variables[..., 2] - self.prev_game_vars[..., 2] < 0
        did_gain_health = game_variables[..., 2] - self.prev_game_vars[..., 2] > 0
        did_loose_armor = game_variables[..., 3] - self.prev_game_vars[..., 3] < 0
        did_get_armor = game_variables[..., 3] - self.prev_game_vars[..., 3] > 0
        did_shoot = (
            game_variables[..., 6:11].sum() - self.prev_game_vars[..., 6:11].sum() < 0
        )
        ammo_collected = (
            game_variables[..., 6:11].sum() - self.prev_game_vars[..., 6:11].sum() > 0
        )

        if did_damage:
            reward += 0.3
        if did_loose_health:
            reward -= 0.1
        if did_gain_health:
            reward += 0.1 / (game_variables[..., 2] / 100)
        if did_loose_armor:
            reward -= 0.1
        if did_get_armor:
            reward += 0.1
        if ammo_collected:
            reward += 0.1
        if not did_damage and did_shoot:
            reward -= 0.05

        self.prev_game_vars = game_variables

        return np.clip(reward, -1, 1)

    def step(self, action):
        actions = np.identity(self.num_actions)
        r_ = self.game.make_action(actions[action], 1)
        reward = self.get_reward() + self.default_reward_weight * r_
        # print(r_)
        # print(r_ * self.default_reward_weight)
        if r_ == -100:
            reward += -1.5

        if self.game.get_state():
            # print("STEP")
            state1 = self.game.get_state().labels_buffer
            state1 = self.scale_down(state1)
            state2 = self.game.get_state().depth_buffer
            state2 = self.scale_down(state2)
            state = np.concatenate([state1, state2], -1)

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
        state = np.zeros((100, 160, 1))
        self.prev_game_vars = None

        return self.scale_down(state), {}

    def scale_down(self, observation):
        resize = cv2.resize(observation, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    def close(self):
        self.game.close()
