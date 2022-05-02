# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:21:18 2022

@author: IIT
"""

import ray

ray.shutdown()    # shutdown the Ray it was started previously
ray.init(ignore_reinit_error=True)  # initialize the ray by ignoring the reinitilization error



# %%

from ray import tune

config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "PongNoFrameskip-v4",
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

#%%
tune.run("PPO", 
         config = config,
         local_dir = './pong/',
         checkpoint_freq = 50,
         verbose=0,
         )

# %%

from ray.rllib.agents.ppo import PPOTrainer

agent = PPOTrainer(config = config)

agent.restore("./pong/PPO/PPO_PongNoFrameskip-v4_aca11_00000_0_2022-04-24_02-25-14/checkpoint_002500/checkpoint-2500")


# %%

import gym
import numpy as np
from gym.wrappers import RecordVideo
from gym.wrappers import Monitor

from ray.rllib.env.wrappers.atari_wrappers import WarpFrame, FrameStack

env = RecordVideo(env = gym.make("PongNoFrameskip-v4"), video_folder="./pong_video/ppo")
env = WarpFrame(env, 84)
env = FrameStack(env, 4)
obs = env.reset()

# %%
while True:
    action = agent.compute_action(observation = obs)
    obs, reward, done, info = env.step(action)
    # print(obs)
    if done:
        break
    
env.close()


# %%
agent.evaluate()