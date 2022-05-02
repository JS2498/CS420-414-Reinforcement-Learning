# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:04:28 2022

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
    "env": "BreakoutNoFrameskip-v4",
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
         local_dir = './breakout/',
         checkpoint_freq = 100,
         verbose=1,
         checkpoint_at_end=True
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

from ray.rllib.agents.a3c.a3c import A3CTrainer

agent = A3CTrainer(config = config)

agent.restore("./breakout/A3C/A3C_BreakoutNoFrameskip-v4_48237_00000_0_2022-04-26_16-24-52/checkpoint_012400/checkpoint-12400")




# %%

#Below code is for recording

import gym
import numpy as np
from gym.wrappers import RecordVideo
from gym.wrappers import Monitor

from ray.rllib.env.wrappers.atari_wrappers import WarpFrame, FrameStack

env = RecordVideo(env = gym.make("BreakoutNoFrameskip-v4"), video_folder="./breakout_results/A3C/")
env = WarpFrame(env, 84)
env = FrameStack(env, 4)
obs = env.reset()

agent.evaluate()

# %%
while True:
    action = agent.compute_action(observation = obs)
    obs, reward, done, info = env.step(action)
    print(obs)
    if done:
        break
    
env.close()
    




#%%


# Example config causing
config = {
    # Also try common gym envs like: "CartPole-v0" or "Pendulum-v1".
    "env": 'BreakoutNoFrameskip-v4',
    # Evaluate once per training iteration.
    "evaluation_interval": 1,
    # Run evaluation on (at least) two episodes
    "evaluation_duration": 20,
    # ... using one evaluation worker (setting this to 0 will cause
    # evaluation to run on the local evaluation worker, blocking
    # training until evaluation is done).
    "evaluation_num_workers": 1,
    # Special evaluation config. Keys specified here will override
    # the same keys in the main config, but only for evaluation.

    'record_video_config': {
    'frequency': 1,
    'directory': 'C:/Users/IIT/Desktop/Jayanth_codes/Mini_project_rllib_codes/videos/',
    'include_global': True,
    'include_agents': False,
    },
    "evaluation_config": {
        # Store videos in this relative directory here inside
        # the default output dir (~/ray_results/...).
        # Alternatively, you can specify an absolute path.
        # Set to True for using the default output dir (~/ray_results/...).
        # Set to False for not recording anything.
        "record_env": "C:/Users/IIT/Desktop/Jayanth_codes/Mini_project_rllib_codes/videos/",
        # "record_env": "/Users/xyz/my_videos/",
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
        "render_env": True,
    },
    "num_workers": 1,
    "framework": "torch",
}

stop = {
    "training_iteration": 20,
}

results = tune.run("A3C", 
                   config=config, 
                   restore = "./breakout/A3C/A3C_BreakoutNoFrameskip-v4_48237_00000_0_2022-04-26_16-24-52/checkpoint_012400/checkpoint-12400",
                   stop=stop)