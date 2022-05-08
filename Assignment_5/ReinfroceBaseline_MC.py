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

        self.fc1_dmis = fc1_dmis
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc3_dims = fc2_dims
        self.fc4_dims = fc2_dims

        self.fc1 = Dense(self.fc1_dmis, activation=None, name='fc1')
        self.fc2 = Dense(self.fc2_dims, activation=None, name='fc2')
        self.fc3 = Dense(self.fc3_dims, activation=None, name='fc3')
        self.fc4 = Dense(self.fc4_dims, activation=None, name='fc4')
        

        self.pi = Dense(self.n_actions, activation="softmax", name="randomized_action")

        # model = keras.Sequential()
        # model.add()
        # model.add(Dense(fc2_dims, activation = "relu"))
        # model.add(Dense(n_actions, activation="softmax"))

    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        pi = self.pi(x)

        return pi


class ValueNetwork(keras.Model):
    def __init__(self, n_actions=3, input_dims=2, fc1_dims=64, fc2_dims=32):
        super(ValueNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc3_dims = fc2_dims
        self.fc4_dims = fc2_dims
        
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation=None, name="fc1")
        self.fc2 = Dense(self.fc2_dims, activation=None, name="fc2")
        self.fc3 = Dense(self.fc3_dims, activation=None, name='fc3')
        self.fc4 = Dense(self.fc4_dims, activation=None, name='fc4')

        self.v = Dense(1, activation=None, name='state_value')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        state_value = self.v(x)

        return state_value


class ReinforceAgent:
    def __init__(self, gamma=0.99, learning_rate=0.001):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.input_dims = 2
        self.fc1_dmis = 32
        self.fc2_dims = 32
        self.n_actions = 3

        self.model = ReinforceNetwork(self.learning_rate, self.n_actions, self.input_dims, self.fc1_dmis, self.fc2_dims)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(), loss = self.custom_loss)

        self.value_model = ValueNetwork(self.n_actions, self.input_dims, self.fc1_dmis, self.fc2_dims)
        self.value_model.compile(optimizer = tf.keras.optimizers.Adam() , loss = self.mse_loss)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.model(state)

        # return the numpy version of the tensor
        return action_probs.numpy()[0]

    def learn(self, episode):
        G_t = 0
        T = len(episode)

        for t in range(T):
            reward = episode[t][2]
            G_t += (self.gamma**t) * reward

        for t in range(T):

            if t > 0:
                G_t = (G_t - episode[t-1][2])/self.gamma

            state = episode[t][0]
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            action = episode[t][1]
            reward = episode[t][2]


            # G_t += self.gamma**t * reward
            with tf.GradientTape() as tape_state:
                state_value = self.value_model(state_tensor)

                loss = self.mse_loss(G_t, state_value)

            gradients = tape_state.gradient(loss, self.value_model.trainable_variables)
            self.value_model.optimizer.apply_gradients([(grad, var) for (grad, var) in zip(gradients, self.value_model.trainable_variables) if grad is not None])
            
            with tf.GradientTape() as tape:
                action_probs = self.model(state_tensor) + 10e-6
                # action_probs_np = action_probs.numpy()[0]
                action_probs = tfp.distributions.Categorical(probs = action_probs)
                
                # log_probs = np.log(action_probs_np[int(action)])#action_probs.log_prob(action)
                # log_probs = tf.convert_to_tensor([log_probs], dtype=tf.float32)

                log_probs = action_probs.log_prob(action)
                
                # print(log_probs)
            
                loss = self.custom_loss(G_t, log_probs, t)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients([(grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None])
            



    def custom_loss(self, total_reward, log_probs, t):
        loss = - (self.gamma**t) * total_reward * log_probs

        return loss

    def mse_loss(self, total_reward, state_value):
        loss = (total_reward - state_value)**2

        return loss

# %%


env = gym.make("MountainCar-v0")  # specify the environment you wish to have
env._max_episode_steps = 200
env.max_episodes_steps = 200

lr = 0.01  # learning rate
gamma = 0.99
n_games = 10000
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
        if (i % 1 == 0):  # n_games-200:
            env.render()
            # time.sleep(.001)
        action_probs = agent.choose_action(observation)
        # print(action_probs)
        action = np.random.choice(env.action_space.n, 1, p=action_probs)
        # print(action)
        next_observation, reward, done, info = env.step(action[0])

        # if next_observation[0] > -0.5:
        #     reward += np.exp((next_observation[0] + 0.5))

        if next_observation[0] >= 0.5:
            reward += 100
            print(f"Reached destination and game no. is {i}")

        score += reward

        episode.append((observation, action, reward))

        observation = next_observation

        # agent.update_weights(Q_value, observation, action, reward, next_observation, done)

    # agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))

    agent.learn(episode)
    scores.append(score)
    avg_scores.append(np.mean(scores[-100:]))

    if (i % 500 == 0):
        print(f"The average reward is {np.mean(scores[-100:])} and game no. is {i}")

    # if (np.mean(scores[-100:])>-110):
    #     i = n_games
env.close()
# %%
x = [i+1 for i in range(n_games)]

plt.figure(figsize = (10,6))
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Score")
plt.title("Reinforce with baseline algorithm for Mountain Car-v0")
plt.savefig("Reinforce_Baseline_MountainCar_avg_return.png")


# %%


for i in range(100):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    episode_length = 0
    # episode = []

    while not done:
        env.render()
        
        action_probs = agent.choose_action(observation)

        action = np.random.choice(env.action_space.n, 1, p=action_probs)
        next_observation, reward, done, info = env.step(action[0])
        
        # if next_observation[0] > -0.5:
        #     reward += np.exp((next_observation[0] + 0.5))

        if next_observation[0] >= 0.5:
            reward += 100
            print(f"Reached destination and game no. is {i}")
        score += reward
        

        episode_length += 1
        observation = next_observation

    print(f"The epsiode is {i} with episode length {episode_length} and total reward {score: .2f}")
# %%

"""
To do :
    1) Reinforce without baseline for mountain car
    2) Reinforce with baseline
"""
