from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from train import VizDoomGym, ActorCriticPolicy, CustomCNN

env = VizDoomGym(config="./scenarios/deathmatch.cfg", render=True)

model = PPO(ActorCriticPolicy, env, verbose=1, learning_rate=0.0001, n_steps=2048, policy_kwargs = dict(
# model = PPO('CnnPolicy', env, verbose=1, learning_rate=0.0001, n_steps=2048, policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64),
    ))

model = PPO.load("./train_labels_buffer/train_basic_0/best_model")

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward}")
