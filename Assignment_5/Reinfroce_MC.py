#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:05:10 2022

@author: jayanths
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
# from tiles3 import tiles, IHT
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import tensorflow_probability as tfp


# %%


class ReinforceNetwork(keras.Model):
    def __init__(self, lr=0.01, n_actions=3, input_dims=2, fc1_dmis=128, fc2_dims=128):
        
        super(ReinforceNetwork, self).__init__()
        
        self.learning_rate = lr
        self.input_dims = lr
        self.fc1_dmis = fc1_dmis
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions


        self.fc1 = Dense(fc1_dmis, activation = "relu", name = 'fc1')
        self.fc2 = Dense(fc2_dims, activation = 'relu', name = 'fc2')
        
        self.pi = Dense(n_actions, activation="softmax", name = "randomized_action")
        
        # model = keras.Sequential()
        # model.add()
        # model.add(Dense(fc2_dims, activation = "relu"))
        # model.add(Dense(n_actions, activation="softmax"))
        
        
    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(state)

        pi = self.pi(x)

        return pi


class ReinforceAgent:
    def __init__(self, gamma = 0.99, learning_rate= 0.001):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.input_dims = 2
        self.fc1_dmis = 64
        self.fc2_dims = 64
        self.n_actions = 3

        self.model = ReinforceNetwork(self.learning_rate, self.n_actions, self.input_dims, self.fc1_dmis, self.fc2_dims)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(), loss = self.custom_loss)
        

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype = tf.float32)
        action_probs = self.model(state)

        # return the numpy version of the tensor
        return action_probs.numpy()[0]


    def learn(self, episode):
        G_t = 0
        T = len(episode)

        for t in range(T):

            state = episode[T-1-t][0]
            state = tf.convert_to_tensor([state], dtype = tf.float32)
            action = episode[T-1-t][1]
            reward = episode[T-1-t][2]


            G_t = self.gamma * G_t + reward
            
            with tf.GradientTape() as tape:
                action_probs = self.model(state)
                action_probs = tfp.distributions.Categorical(probs = action_probs)

                log_probs = action_probs.log_prob(action)

                loss = self.custom_loss(G_t, log_probs)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients([grad, var] for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)


    def custom_loss(self, total_reward, log_probs):
        loss = - total_reward * log_probs

        return loss


    def update_weights(self):
        pass


# %%

env = gym.make("MountainCar-v0")  # specify the environment you wish to have
env._max_episode_steps = 200
env.max_episodes_steps = 200



lr = 0.01  # learning rate
gamma = 0.99
n_games = 100000
epsilon = 0.98
scores = []
avg_scores = []

agent = ReinforceAgent()

for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    # discrete_obs = discretize_obs(observation)
    episode = []
    while not done:
        if (i % 100 == 0): # n_games-200:
            env.render()
            time.sleep(.002)
        action_probs = agent.choose_action(observation)
        # print(action_probs)
        action = np.random.choice(env.action_space.n, 1, p = action_probs)
        # print(action)
        next_observation, reward, done, info = env.step(action[0])

        score += reward
        
        episode.append((observation, action, reward))
        
        observation = next_observation

        # agent.update_weights(Q_value, observation, action, reward, next_observation, done)

    # agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))

    agent.learn(episode)
    scores.append(score)
    avg_scores.append(np.mean(scores))

    if (i % 100 == 0):
        print(f"The average reward is {np.mean(scores[-100:])}")
        
    
        
    # if (np.mean(scores[-100:])>-110):
    #     i = n_games
               

x = [i+1 for i in range(n_games)]
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Score")


env.close()

# %%

"""
To do :
    
    1) Reinforce without baseline for mountain car 
    2) Reinforce with baseline
    
"""


