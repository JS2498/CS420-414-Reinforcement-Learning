#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:40:24 2022

@author: jayanths
"""

# %%

"""
References :
    1. http://incompleteideas.net/tiles/tiles3.html for tile coding
"""

"""
Tile Coding :
     A way to represent a vector of continuous values using a vector of sparse
    binary vector.
    
"""


# %%

import numpy as np
import matplotlib.pyplot as plt
import gym
from tiles3 import tiles, IHT
import time

# %%


class MountainCarTileCoder:
    def __init__(self, max_size, num_tiles, num_tilings):
        """
        Arguments :
            max_size : Maximum size of the integer hash table 
                       (Usually in powers of 2)
            num_tiles : Number of tiles
            num_tilings : Number of tilings
            
            For the below one, num_tilings = 2
            
               --------------------
               \                   \
            ----------------\      \
            \               \      \
            \               \      \
            \               \      \
            \               \      \
            \               \------\
            \               \        
            \---------------\
            
        """
        
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.iht = IHT(max_size)
        
        

    def get_tiles(self, state):
        """"
        Arguments:
            pos : 1-D position of the car
            vel: velocity of the car
        """
        
        pos, vel = state
        min_pos = -1.2
        min_vel = -0.07

        max_pos = 0.5
        max_vel = 0.07
        
        pos_scaled = self.num_tiles * ((pos - min_pos)/(max_pos - min_pos))
        vel_scaled = self.num_tiles * ((vel - min_vel)/(max_vel - min_vel))
        
        # print(pos_scaled, vel_scaled)
        tiles_list = tiles(self.iht, self.num_tilings,  [pos_scaled, vel_scaled])
        
        return np.array(tiles_list)
        
class QlearningAgent:
    """
    Q learning with linear function approximation using tile coding
    """

    def __init__(self, agent_info):

        self.num_tiles = agent_info["num_tiles"]
        self.num_tilings = agent_info["num_tilings"]
        self.iht_size = agent_info["iht_size"]
        self.num_actions = 3

        self.w = np.ones((self.iht_size, self.num_actions))
        self.state_tc = np.zeros((1, self.w.shape[0]))
        self.next_state_tc = np.zeros((1, self.w.shape[0]))


        self.tile_coder = MountainCarTileCoder(max_size= self.iht_size, num_tiles = self.num_tiles, num_tilings = self.num_tilings)


        self.epsilon = agent_info["init_epsilon"]
        self.init_epsilon = agent_info["init_epsilon"]

        self.alpha = agent_info["learning_rate"]
        self.gamma = agent_info["discount"]

        self.eps_reduction = 0.9999
        self.epsilon_min = 0.01





    def choose_action(self, state):

        Q = []

        for i in range(self.num_actions):
            Q_val = self.Qvalue(state, i)
            Q.append(Q_val)

        if (np.random.random()  < self.epsilon) :
            action = np.random.randint(0, self.num_actions)

        else :
            action = np.argmax(Q)

        return action, Q[action]
    
    def Qvalue(self, state, action):
        
        # pos, vel = state
        
        # identify the non - zero tiles 
        nz_tiles = self.tile_coder.get_tiles(state)
        state_tc = np.zeros((self.w.shape[0], ))

        state_tc[nz_tiles] = 1

        Q_val = np.dot(state_tc, self.w[:,action])
        
        return Q_val


    def update_weights(self, Q_value, state, action, reward, next_state, done):

        pos, vel = state
        # .get the index of non-zero tiles
        nz_tiles = self.tile_coder.get_tiles(state)
        state_tc = np.zeros((self.w.shape[0],))

        # print(nz_tiles.shape)
        # print(state_tc.shape)
        state_tc[nz_tiles] = 1
        
        Q_next_values = []
 
        for i in range(self.num_actions):
            Q_next_value = self.Qvalue(next_observation, i)
            Q_next_values.append(Q_next_value)
            
        # print(Q_value, Q_next_value)
        self.w[:, action] += self.alpha * (reward + (1-done) * gamma * np.max(Q_next_values) - Q_value) * state_tc


class SarsaAgent:
    """
    SARSA with linear function approximation using tile coding
    """

    def __init__(self, agent_info):

        self.num_tiles = agent_info["num_tiles"]
        self.num_tilings = agent_info["num_tilings"]
        self.iht_size = agent_info["iht_size"]
        self.num_actions = 3

        self.w = np.ones((self.iht_size, self.num_actions))
        self.state_tc = np.zeros((1, self.w.shape[0]))
        self.next_state_tc = np.zeros((1, self.w.shape[0]))


        self.tile_coder = MountainCarTileCoder(max_size= self.iht_size, num_tiles = self.num_tiles, num_tilings = self.num_tilings)


        self.epsilon = agent_info["init_epsilon"]
        self.init_epsilon = agent_info["init_epsilon"]

        self.alpha = agent_info["learning_rate"]
        self.gamma = agent_info["discount"]

        self.eps_reduction = 0.9999
        self.epsilon_min = 0.01





    def choose_action(self, state):

        Q = []

        for i in range(self.num_actions):
            Q_val = self.Qvalue(state, i)
            Q.append(Q_val)

        if (np.random.random()  < self.epsilon) :
            action = np.random.randint(0, self.num_actions)

        else :
            action = np.argmax(Q)

        return action, Q[action]
    
    def Qvalue(self, state, action):
        
        # pos, vel = state
        
        # identify the non - zero tiles 
        nz_tiles = self.tile_coder.get_tiles(state)
        state_tc = np.zeros((self.w.shape[0], ))

        state_tc[nz_tiles] = 1

        Q_val = np.dot(state_tc, self.w[:,action])
        
        return Q_val


    def update_weights(self, Q_value, state, action, reward, next_state, done):

        pos, vel = state
        # .get the index of non-zero tiles
        nz_tiles = self.tile_coder.get_tiles(state)
        state_tc = np.zeros((self.w.shape[0],))

        # print(nz_tiles.shape)
        # print(state_tc.shape)
        state_tc[nz_tiles] = 1
        
        # Q_next_values = []
 
        # for i in range(self.num_actions):
        #     Q_next_value = self.Qvalue(next_observation, i)
        #     Q_next_values.append(Q_next_value)
        
        next_action, Q_next_value = self.choose_action(next_state)
        # print(next_action)
            
        # print(Q_value, Q_next_value)
        self.w[:, action] += self.alpha * (reward + (1-done) * gamma * Q_next_value - Q_value) * state_tc



# %%

env = gym.make("MountainCar-v0")  # specify the environment you wish to have

lr = 0.01  # learning rate
gamma = 0.99
n_games = 100000
epsilon = 0.98
scores = []
avg_scores = []

# specify the precision you require while discretizing the state space
# pos_precision = 0.03
# vel_precision = 0.005

# num_states = (env.observation_space.high - env.observation_space.low)/([pos_precision, vel_precision])
# num_states = num_states.astype(int)


agent_info = {"num_tiles": 16, 
              "num_tilings": 8,
              "iht_size": 4096,
              "init_epsilon": 0.99,
              "learning_rate": 0.1,
              "discount": 0.99
              }

agent = SarsaAgent(agent_info)

for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    # discrete_obs = discretize_obs(observation)

    while not done:
        if i > n_games-200:
            env.render()
            time.sleep(.005)
        action, Q_value = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        # discrete_next = discretize_obs(next_observation)
        # print(discrete_next, reward, action)
        score += reward

        # observation = next_observation

        # Update step for Q-learning
        # _, Q_next_value = agent.choose_action(next_observation, 0)
        # agent.w[:, action] += self.alpha * (reward + (1-done) * gamma * Q_next_value - Q_value)*  
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(agent.Q[discrete_next[0], discrete_next[1],:]) - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # Update step for SARSA
        # next_action = agent.choose_action(discrete_next)
        # Q_next = agent.Q[discrete_next[0], discrete_next[1], next_action]
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - agent.Q[discrete_obs[0], discrete_obs[1], action])
        # print(observation, discrete_next, discrete_obs, action)
        # discrete_obs = discrete_next
        agent.update_weights(Q_value, observation, action, reward, next_observation, done)
        observation = next_observation

    agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))
    scores.append(score)
    avg_scores.append(np.mean(scores))

    if (i % 10000 == 0):
        print(f"The average reward is {np.mean(scores[-100:])}")
        
    # if (np.mean(scores[-100:])>-110):
    #     i = n_games
               

x = [i+1 for i in range(n_games)]
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Score")


env.close()








# %%

# env = gym.make("MountainCar-v0")  # specify the environment you wish to have

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


# %%    
        
"""
To do :
For mountain car problem implement
    1. Q-learning  and SARSA with multiple tile coding (Done)
    2. Radial basis function with suitable choice of centers and variance
    3. Reinforce without baseline 
    4. Reinforce with baseline

Use ML with phil codes for implementing
5. DQN for cartpole or Breakout with Rllib code
6. A2C for cartpole or Breakout with Rllib code

"""