import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
# import argparse



gymenv = gym.make("Humanoid-v4", render_mode='human')


## ---- GYM SAC ------ ##
model = SAC.load("models/350000.zip", env=gymenv)


obs = gymenv.reset()[0]
done = False
extra_steps = 1500

# print(obs)
while True:
    action, _ = model.predict(obs)
    obs, _, done, _, _ = gymenv.step(action)

    if done:
        extra_steps -= 1

        if extra_steps < 0:
            break


