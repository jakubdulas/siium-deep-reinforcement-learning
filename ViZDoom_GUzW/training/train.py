from stable_baselines3 import PPO
from doom_env import VizDoomGym
from callbacks import TrainAndLoggingCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from model import CustomCNN
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    model = None
    prev_map = "deathmatch"
    map_configs = [
        # {
        #     "map": "basic",
        #     "total_timesteps": 100_000,
        #     "with_bots": False,
        #     "living_reward": -0.05,
        # },
        # {
        #     "map": "deadly_corridor",
        #     "total_timesteps": 100_000,
        #     "with_bots": False,
        #     "living_reward": 0.0,
        #     "default_reward_weight": 0.005,
        # },
        # {
        #     "map": "defend_the_center",
        #     "total_timesteps": 100_000,
        #     "with_bots": False,
        #     "living_reward": 0.0,
        #     "default_reward_weight": 0.005,
        # },
        # {
        #     "map": "health_gathering",
        #     "total_timesteps": 10_000,
        #     "with_bots": False,
        #     "living_reward": 0.0,
        #     "default_reward_weight": 0.005,
        # },
        # {
        #     "map": "health_gathering_supreme",
        #     "total_timesteps": 10_000,
        #     "with_bots": False,
        #     "living_reward": 0.0,
        #     "default_reward_weight": 0.005,
        # },
        {
            "map": "deathmatch",
            "total_timesteps": 10_000,
            "with_bots": False,
            "living_reward": 0.0,
            "default_reward_weight": 0.005,
        },
        {
            "map": "deathmatch",
            "total_timesteps": 1_500_000,
            "with_bots": True,
            "living_reward": 0.005,
            "default_reward_weight": 0.005,
        },
    ]
    model_path = f"./train_dapth_and_labels/train_{prev_map}_0/best_model"
    n_envs = 1

    for map_conf in map_configs:
        map_ = map_conf["map"]
        total_timesteps = map_conf["total_timesteps"]
        with_bots = map_conf["with_bots"]
        living_reward = map_conf["living_reward"]
        default_reward_weight = map_conf["default_reward_weight"]

        for doom_skill in range(0, 1):
            CHECKPOINT_DIR = f"./train_dapth_and_labels/train_{map_}_{doom_skill}"
            LOG_DIR = f"./logs/log_{map_}_{doom_skill}"

            callback = TrainAndLoggingCallback(
                check_freq=1000, save_path=CHECKPOINT_DIR
            )

            def make_env():
                def _init():
                    env = VizDoomGym(
                        f"scenarios/{map_}.cfg",
                        render=False,
                        with_bots=with_bots,
                        living_reward=living_reward,
                        default_reward_weight=default_reward_weight,
                    )
                    env.game.set_doom_skill(doom_skill)
                    return env

                return _init

            # env = SubprocVecEnv([make_env() for i in range(n_envs)])
            env = make_env()()

            if model_path:
                model = PPO.load(model_path, env=env)
                model_path = None
            elif doom_skill == 0:
                model = PPO.load(
                    f"./train_dapth_and_labels/train_{prev_map}_4/best_model", env=env
                )

                # model = PPO(
                #     ActorCriticCnnPolicy,
                #     env,
                #     tensorboard_log=LOG_DIR,
                #     verbose=1,
                #     learning_rate=0.0001,
                #     n_steps=2048,
                #     policy_kwargs=dict(
                #        net_arch={"pi": [32, 32, 32], "vf": [32, 32, 32]}
                #     ),
                # )

                # model = PPO(
                #     ActorCriticPolicy,
                #     env,
                #     tensorboard_log=LOG_DIR,
                #     verbose=1,
                #     learning_rate=0.0001,
                #     n_steps=2048,
                #     policy_kwargs=dict(
                #         features_extractor_class=CustomCNN,
                #         features_extractor_kwargs=dict(features_dim=128),
                #     ),
                # )

            model.learn(total_timesteps=total_timesteps, callback=callback)

            # for episode in range(20):
            #     obs, _ = env.reset()
            #     done = False
            #     total_reward = 0
            #     while not done:
            #         action, _ = model.predict(obs)
            #         obs, reward, done, truncated, info = env.step(action)
            #         total_reward += reward
            #     print(f"Total Reward for episode {episode} is {total_reward}")

        prev_map = map_
