from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from train import VizDoomGym


map_ = "deathmatch"
lvl = 0
env = VizDoomGym(config=f"./scenarios/{map_}.cfg", render=True)
model = PPO.load(f"./train_dapth_and_labels/train_{map_}_{lvl}/best_model")

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward}")
