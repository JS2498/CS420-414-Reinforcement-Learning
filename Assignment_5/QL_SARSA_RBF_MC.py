# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:52:31 2022

@author: Keerthana_Jayanth
"""

# %%
import numpy as np
import gym
import matplotlib.pyplot as plt

import sklearn
import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

# %%

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

# %%

env = gym.make("MountainCar-v0")

observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = FeatureUnion([("rbf1", RBFSampler(gamma = 5, n_components = 100)),
                                            ("rbf2", RBFSampler(gamma = 2, n_components = 100)),
                                            ("rbf3", RBFSampler(gamma = 1, n_components = 100)),
                                            ("rbf4", RBFSampler(gamma = 0.8, n_components = 100)),
                                            ("rbf5", RBFSampler(gamma = 0.2, n_components = 100))])

featurizer.fit(scaler.transform(observation_examples))


# %%

class Estimator():
    
    def __init__(self):
        
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate = "constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
            
        
    def featurize_state(self, state):
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        
        return featurized[0]
    
    def predict(self, s, a):
        
        features = self.featurize_state(s)
        prediction = self.models[a].predict([features])
        
        return prediction[0] #self.models[a].predict([features])
    
    def update(self, s, a, y):
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
        

# %%

class SARSA:
    def __init__(self, num_states, epsilon=0.2, n_actions=3):
            # print(n_actions)
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states[0], num_states[1], n_actions))
        self.Q[-1, :, :] = 0
        self.eps_reduction = 0.9999
        self.gamma = 0.99
        self.n_actions = n_actions
        self.estimator = Estimator()

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            Q = []
            for a in range(self.n_actions):
                Q_val = self.estimator.predict(state, a)
                Q.append(Q_val)
                
            action = np.argmax(Q)
        return action
    
    def learn(self, observation, action, reward, next_observation):
        
        next_action = self.choose_action(next_observation)
        Q_next = self.estimator.predict(next_observation,next_action)
        Q_val = reward + self.gamma * Q_next
        
        self.estimator.update(observation, action, Q_val)
        
    
    # def learn(self, discrete_obs, action, reward, discrete_next):
        
    #     next_action = self.choose_action(discrete_next)
    #     Q_next = self.Q[discrete_next[0], discrete_next[1], next_action]
    #     self.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - self.Q[discrete_obs[0], discrete_obs[1], action])
        

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
        self.gamma = 0.99
        self.n_actions = n_actions
        self.estimator = Estimator()

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            Q = []
            for a in range(self.n_actions):
                Q_val = self.estimator.predict(state, a)
                Q.append(Q_val)
                
            action = np.argmax(Q)
        return action
    
    def learn(self, observation, action, reward, next_observation):
        
        Q = []
        for a in range(self.n_actions):
            Q_val = self.estimator.predict(next_observation, a)
            Q.append(Q_val)
            
        Q_next = np.max(Q)
        Q_val = reward + self.gamma * Q_next
        
        self.estimator.update(observation, action, Q_val)
        
         # self.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(self.Q[discrete_next[0], discrete_next[1],:]) - self.Q[discrete_obs[0], discrete_obs[1], action])



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
    
    while not done:
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
       
        score += reward
        # Update step for Q-learning
        agent.learn(observation, action, reward, next_observation)
        observation = next_observation

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
n_games = 50000
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


    while not done:
        if i > n_games-20:
            env.render()
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        # print(discrete_next, reward, action)
        score += reward

        # Update step for Q-learning
        agent.learn(observation, action, reward, next_observation)
        observation = next_observation
        
        # if observation[0] > 0.5:
        #     print(f"The car has reached the goal in the episode {i}")

    agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))
    scores.append(score)
    avg_scores.append(np.mean(scores[-100:]))

    if (i % 1000 == 0):
        print(f"The average reward is {np.mean(scores[-100:])} and episode no. is {i}")
        
    # if (np.mean(scores[-100:])>-110):
    #     i = n_games


env.close()
# save_agent_gif(env)
            
# %%
x = [i+1 for i in range(n_games)]
plt.figure(figsize = (10,6))
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Score")
plt.title("SARSA using RBF for Mountain Car")
plt.savefig("SARSA_RBF_MountainCar.png")

# %%
for i in range(20):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car

    while not done:

        env.render()
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        # print(discrete_next, reward, action)
        score += reward
        observation = next_observation
        
    print(f"The reward in game {i} is {score}")
env.close()
