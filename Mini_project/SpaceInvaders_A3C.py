# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 02:44:34 2022

@author: IIT
"""


import ray

ray.shutdown()    # shutdown the Ray it was started previously
ray.init(ignore_reinit_error=True)  # initialize the ray by ignoring the reinitilization error


# %%


from ray import tune

config = {
    
    "use_critic" : True,
    "use_gae" : True,
    "lambda" : 1.0,
    "grad_clip" : 40.0,
    "lr_schedule" : None,
    "vf_loss_coeff" : 0.5,
    "entropy_coeff" : 0.01,
    "entropy_coeff_schedule" : None,
    "sample_async" : True,

    # Override some of TrainerConfig's default values with PPO-specific values.
    "rollout_fragment_length" : 10,
    "lr" : 0.0001,
    # Min time (in seconds) per reporting.
    # This causes not every call to `training_iteration` to be reported,
    # but to wait until n seconds have passed and then to summarize the
    # thus far collected results.
    "min_time_s_per_reporting" : 5,
    "_disable_execution_plan_api" : True,
    # Environment (RLlib understands openAI gym registered strings).
    "env": "SpaceInvadersNoFrameskip-v4",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 16,
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
    },
}

#%%

tune.run("A3C", 
         config = config,
         local_dir = './spaceinvaders/',
         checkpoint_freq = 100,
         verbose=1,
         checkpoint_at_end=True
         )


#%%

from ray import tune

config = {   
   "use_critic" : True,
   "use_gae" : True,
   "lambda" : 1.0,
   "grad_clip" : 40.0,
   "lr_schedule" : None,
   "vf_loss_coeff" : 0.5,
   "entropy_coeff" : 0.01,
   "entropy_coeff_schedule" : None,
   "sample_async" : True,

   # Override some of TrainerConfig's default values with PPO-specific values.
   "rollout_fragment_length" : 10,
   "lr" : 0.0001,
   # Min time (in seconds) per reporting.
   # This causes not every call to `training_iteration` to be reported,
   # but to wait until n seconds have passed and then to summarize the
   # thus far collected results.
   "min_time_s_per_reporting" : 5,
   "_disable_execution_plan_api" : True,
    # Environment (RLlib understands openAI gym registered strings).
    "env": "SpaceInvadersNoFrameskip-v4",
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


from ray.rllib.agents.a3c.a3c import A3CTrainer

agent = A3CTrainer(config = config)

agent.restore("./spaceinvaders/A3C/A3C_SpaceInvadersNoFrameskip-v4_b0253_00000_0_2022-05-01_10-52-26/checkpoint_010800/checkpoint-10800")


# %%

#Below code is for recording

import gym
import numpy as np
from gym.wrappers import RecordVideo
from gym.wrappers import Monitor

from ray.rllib.env.wrappers.atari_wrappers import WarpFrame, FrameStack

env = RecordVideo(env = gym.make("SpaceInvadersNoFrameskip-v4"), video_folder="./spaceinvaders_results/A3C/")
env = WarpFrame(env, 84)
env = FrameStack(env, 4)
obs = env.reset()

agent.evaluate()

# %% Restoring the model


tune.run("A3C", 
         config = config,
         local_dir = './spaceinvaders/',
         checkpoint_freq = 100,
         verbose=1,
         checkpoint_at_end=True,
         restore= "./spaceinvaders/A3C/A3C_SpaceInvadersNoFrameskip-v4_bfb79_00000_0_2022-04-24_21-52-34/checkpoint_010000/checkpoint-10000",
         )
