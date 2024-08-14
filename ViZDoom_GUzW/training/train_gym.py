import gymnasium
from vizdoom import gymnasium_wrapper

env = gymnasium.make("VizdoomCorridor-v0", depth=True, labels=True)

observation, info = env.reset()
for _ in range(1000):
#    action = policy(observation)  # this is where you would insert your policy
    print(observation.keys())
    break
    # observation, reward, terminated, truncated, info = env.step(action)

    # if terminated or truncated:
        # observation, info = env.reset()

env.close()
