from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from train2 import VizDoomGym, CustomActorCriticPolicy, CustomCNN

env = VizDoomGym(config="scenarios/deathmatch.cfg", render=True)

model = PPO(CustomActorCriticPolicy, env, verbose=1, learning_rate=0.0001, n_steps=2048, policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[256, 128, 64],
            vf=[256, 128, 64],
        ),
    ))

model.load("best_model_10000.zip")

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward}")
