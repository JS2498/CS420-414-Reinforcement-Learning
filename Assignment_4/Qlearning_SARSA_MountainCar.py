#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 06:16:42 2022

@author: jayanths
"""
# %%


# %%
import numpy as np
import gym
import matplotlib.pyplot as plt

import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

# %%


class MonteCarlo:
    def __init__(self):
        pass

    def choose_action(self, state):
        pass

# %%


class SARSA:
    def __init__(self, num_states, epsilon=0.2, n_actions=3):
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states[0], num_states[1], n_actions))
        self.Q[-1, :, :] = 0
        self.eps_reduction = 0.9999

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            action = np.argmax(self.Q[state[0], state[1], :])
        return action
    
    def learn(self, discrete_obs, action, reward, discrete_next):
        
        next_action = self.choose_action(discrete_next)
        Q_next = self.Q[discrete_next[0], discrete_next[1], next_action]
        self.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - self.Q[discrete_obs[0], discrete_obs[1], action])
        

# %%


class Qlearning:
    def __init__(self,  num_states, epsilon=0.2, n_actions=3):
        # print(n_actions)
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states[0], num_states[1], n_actions))
        self.Q[-1, :, :] = 0
        self.eps_reduction = 0.9999

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            action = np.argmax(self.Q[state[0], state[1], :])
        return action
    
    def learn(self, discrete_obs, action, reward, discrete_next):
        
         self.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(self.Q[discrete_next[0], discrete_next[1],:]) - self.Q[discrete_obs[0], discrete_obs[1], action])



def discretize_obs(continuous_state):
    # pos_precision = 0.00
    # vel_precision = 0.0007

    min_pos = -1.2
    min_vel = -0.07

    # max_pos = 0.6
    # max_vel = 0.07

    dis_pos = int((continuous_state[0] - min_pos) / pos_precision)
    dis_vel = int((continuous_state[1] - min_vel) / vel_precision)

    # print(dis_pos, dis_vel)
    return np.array([dis_pos, dis_vel])

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im


def save_agent_gif(env):
    frames = []
    done = False
    score = 0
    i = 0
    observation = env.reset()  # initalizes the starting state of the car
    discrete_obs = discretize_obs(observation)  
    while not done:
        action = agent.choose_action(discrete_obs)
        next_observation, reward, done, info = env.step(action)
        discrete_next = discretize_obs(next_observation)
        # print(discrete_next, reward, action)
        score += reward

        observation = next_observation

        # Update step for Q-learning
        agent.learn(discrete_obs, action, reward, discrete_next)
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(agent.Q[discrete_next[0], discrete_next[1],:]) - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # Update step for SARSA
        # next_action = agent.choose_action(discrete_next)
        # Q_next = agent.Q[discrete_next[0], discrete_next[1], next_action]
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - agent.Q[discrete_obs[0], discrete_obs[1], action])
        # print(observation, discrete_next, discrete_obs, action)
        discrete_obs = discrete_next

        frame = env.render(mode='rgb_array')
        frames.append(_label_with_episode_number(frame, episode_num=i))
        i = i+1

        # state, _, done, _ = env.step(action)
        if done:
            break

    env.close()

    imageio.mimwrite(os.path.join('./videos/', 'SARSA_agent.gif'), frames, fps=60)


# %%


env = gym.make("MountainCar-v0")  # specify the environment you wish to have

"""
Environment Description :
    A car has to travel in a one-dimensional (horizontal) track with the road
    being similar to a valley between mountains on either side of it. The goal
    is to reach the top of right side mountain starting in the valley.

State space : 2-D states representing (position, velocity)
Action space :
    0. Push left (accelerate left),
    1. Push right (accelerate right),
    2. Do nothing
Reward :
    * -1 for any action taken
    * 0 on reaching the goal


Initial state : (x,0) with x taking value between [-.0,-0.4] uniformally

Termination :
    * On reaching the goal i.e. x>0.5
    * Else if length of episode is 200

Remember :
    * x belongs to [-1.2, 0.6]
    * max speed = 0.07
    * Mountain car is a continuous state space

"""


lr = 0.01  # learning rate
gamma = 0.99
n_games = 150000
epsilon = 0.98
scores = []
avg_scores = []

# specify the precision you require while discretizing the state space
pos_precision = 0.03
vel_precision = 0.005

num_states = (env.observation_space.high - env.observation_space.low)/([pos_precision, vel_precision])
num_states = num_states.astype(int)

agent = SARSA(epsilon=epsilon, n_actions=env.action_space.n, num_states=num_states)


for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    discrete_obs = discretize_obs(observation)

    while not done:
        if i > n_games-20:
            env.render()
        action = agent.choose_action(discrete_obs)
        next_observation, reward, done, info = env.step(action)
        discrete_next = discretize_obs(next_observation)
        # print(discrete_next, reward, action)
        score += reward

        observation = next_observation

        # Update step for Q-learning
        agent.learn(discrete_obs, action, reward, discrete_next)
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(agent.Q[discrete_next[0], discrete_next[1],:]) - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # Update step for SARSA
        # next_action = agent.choose_action(discrete_next)
        # Q_next = agent.Q[discrete_next[0], discrete_next[1], next_action]
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - agent.Q[discrete_obs[0], discrete_obs[1], action])
        # print(observation, discrete_next, discrete_obs, action)
        discrete_obs = discrete_next
        
        # if observation[0] > 0.5:
        #     print(f"The car has reached the goal in the episode {i}")

    agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))
    scores.append(score)
    avg_scores.append(np.mean(scores[-100:]))

    if (i % 10000 == 0):
        print(f"The average reward is {np.mean(scores[-100:])} and episode no. is {i}")
        
    # if (np.mean(scores[-100:])>-110):
    #     i = n_games
    
save_agent_gif(env)
            
# %%
x = [i+1 for i in range(n_games)]
plt.figure(figsize = (10,6))
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Score")
plt.title("SARSA for Mountain Car")
plt.savefig("SARSA_MountainCar.png")

# %%

env.close()

"""
Observations :
    * If we consider large state space then we need to play many games 
    (close to 1 lakh) for the Qlearning agent to learn. This might be true
    since for large state space we need to enter many states for closely
    approximating the state-action values. However this may not be a good way
    
    * The SARSA algorithm was able to find a optimal policy to reach the goal
    in fewer episodes when compared with the Qlearning algorithm
    
    * Also the minimum steps to reach the goal was less for SARSA algorithm
    when compared with Qlearning algorithm
    
"""



# env = gym.make('CartPole-v1')
