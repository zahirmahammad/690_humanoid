import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, PPO, DQN, DDPG, HER
from stable_baselines3.common.vec_env import DummyVecEnv
import os
# import argparse


gymenv1 = gym.make("Humanoid-v4", render_mode="human")
# train(gymenv, "SAC")

print("Action Space K: ", gymenv1.action_space)

print("Obs Space K: ", gymenv1.observation_space)

## ---------- Random Actions Check ---------- ##
# gymenv1.reset()
# c=0
# for i in range(0, 10):
#     c+=1
#     action = gymenv1.action_space.sample()
#     observation, reward, terminated, done, info = gymenv1.step(action)
#     print(len(observation))
# print(c)
## ---------- Random Actions Check ---------- ##




# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


model = SAC('MlpPolicy', gymenv1, verbose=1, device='cuda', tensorboard_log=log_dir)

TIMESTEPS = 20000
iters = 0
while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/{TIMESTEPS*iters}")

    if TIMESTEPS*iters == 600000:
        break
    
# model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
