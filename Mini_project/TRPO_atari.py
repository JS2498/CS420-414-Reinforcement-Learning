# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:45:47 2022

@author: Jayanth S
"""

# import gym
# import numpy as np

# from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# from stable_baselines3.common.cmd_util import make_atari_env
# # from stable_baselines3.common.policies import CnnPolicy
# from stable_baselines3.common.vec_env import VecFrameStack
# # from stable_baselines3 import ACER

# # %%
# env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

# # %%
# # The noise objects for TD3
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = TD3("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("td3_pendulum")
# env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = TD3.load("td3_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
    
   
    
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

#%%
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
# env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

# model = A2C("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=int(5e6))
# obs = env.reset()
# #model = A2C.load("A2C_breakout") #uncomment to load saved model
# model.save("A2C_breakout")
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
    
    
# %%


# import gym

# from stable_baselines.common.policies import MlpPolicy
# from sb3_contrib import TRPO

# env = make_atari_env('PongNoFrameskip-v4', seed=0)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

# model = TRPO("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=int(5e6))
# obs = env.reset()
# #model = A2C.load("A2C_breakout") #uncomment to load saved model
# model.save("TRPO_breakout")
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
    
    

# %%

from sb3_contrib import TRPO

env = make_atari_env('PongNoFrameskip-v4', seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = TRPO("CnnPolicy", env, gamma=0.9, verbose=1, tensorboard_log="D:/Jayanth/MSR/Reinforcement_Learning/RL_codes/RL_assignment/Mini_project/trpo_breakout_tensorboard/")
model.learn(total_timesteps=int(5e6), log_interval=4)

model.save("D:/Jayanth/MSR/Reinforcement_Learning/RL_codes/RL_assignment/Mini_project/trpo_pong")

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
    
    
# %%
# import numpy as np
# import matplotlib.pyplot as plt

# from stable_baselines3 import SAC
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.logger import Figure

# model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


# class FigureRecorderCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(FigureRecorderCallback, self).__init__(verbose)

#     def _on_step(self):
#         # Plot values (here a random variable)
#         figure = plt.figure()
#         figure.add_subplot().plot(np.random.random(3))
#         # Close the figure after logging it
#         self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
#         plt.close()
#         return True


# model.learn(5000, callback=FigureRecorderCallback())

# %%


# from stable_baselines3 import A2C

# model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="D:/Jayanth/MSR/Reinforcement_Learning/RL_codes/RL_assignment/Mini_project/a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=10000)
