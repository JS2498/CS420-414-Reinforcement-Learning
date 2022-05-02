# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:02:18 2022

@author: IIT
"""



import ray

ray.shutdown()    # shutdown the Ray it was started previously
ray.init(ignore_reinit_error=True)  # initialize the ray by ignoring the reinitilization error

# %%
from ray import tune

config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "BreakoutNoFrameskip-v4",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 16,
    "num_gpus": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    #"model": {
    #    "fcnet_hiddens": [64, 64],
    #    "fcnet_activation": "relu",
    #},
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

# %%
tune.run("A2C", 
         config = config,
         local_dir = './breakout/',
         checkpoint_freq = 100,
         verbose=1, 
         checkpoint_at_end= True
         )


# %%

from ray import tune

config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "BreakoutNoFrameskip-v4",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers":1,
    "num_gpus" : 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    #"model": {
    #    "fcnet_hiddens": [64, 64],
    #    "fcnet_activation": "relu",
    #},
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
        "record_env": True
    },
}

# %%

from ray.rllib.agents.a3c.a2c import A2CTrainer

agent = A2CTrainer(config = config)

agent.restore("./breakout/A2C/A2C_BreakoutNoFrameskip-v4_75d2f_00000_0_2022-04-26_16-26-05/checkpoint_003000/checkpoint-3000")


# %%

#Below code is for recording

import gym
import numpy as np
from gym.wrappers import RecordVideo
from gym.wrappers import Monitor

from ray.rllib.env.wrappers.atari_wrappers import WarpFrame, FrameStack

env = RecordVideo(env = gym.make("BreakoutNoFrameskip-v4"), video_folder="./breakout_results/A2C/")
env = WarpFrame(env, 84)
env = FrameStack(env, 4)
obs = env.reset()

agent.evaluate()

# %%
while True:
    action = agent.compute_action(observation = obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
    
env.close()
    


    

# %%

# #Loading the saved model for retraining

# agent = A2CTrainer(config = config)

# agent.restore("C:/Users/IIT/Desktop/Jayanth_codes/cartpole_v0/A2C/A2C_CartPole-v0_c0840_00000_0_2022-04-23_14-22-49/checkpoint_000170/checkpoint-170")

# for i in range(300):
#     agent.train()

# agent.evaluate()

# #%%


# tune.run("A2C", 
#           config = config,
#           local_dir = 'cartpole_v0',
#           checkpoint_freq = 10,
#           verbose=1,
#           restore = "./cartpole_v0/A2C/A2C_CartPole-v0_c0840_00000_0_2022-04-23_14-22-49/checkpoint_000170/checkpoint-170", checkpoint_at_end= True
#           )
