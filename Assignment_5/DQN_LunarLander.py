#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:38:25 2022

@author: jayanths
"""

import gym
import numpy as np
import tensorflow as tf
import keras
# from keras.optimizers import Adam
from keras.models import load_model
# from utlis import plotLearning


# %%

# store (s,a,r,s') pairs in a buffer such that they can be used for training the neural network
# Specify the buffer size and the dimenision of the input while defining the buffer
class ReplyBuffer():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0  #points to the starting point of the ReplyBuffer
        self.state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32) 
        #if input_dims is a list then *input_dims will unpack the list (advantage is it can be expandable)
        
        self.new_state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32) 
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        
        self.reward_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.int32) #to keep track of episodes that end with a win
        
    # define a function that stores (s,a,r,s') in the buffer
    def store_transition(self,state,action,reward,new_state,done):
        index = self.mem_cntr%self.mem_size # tells where the (s,a,r,s') pair has to be stored
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = int(done) #why (1-done) ?  
        self.mem_cntr += 1
     
    # define a function that samples a set of (s,a,r,s') pairs so that it can be fed to NN for training 
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  #to avoid sampling just zeros
        batch = np.random.choice(max_mem, batch_size, replace = False) 
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        termination = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, termination
        
        
def build_dqn(lr,n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation = 'relu'),
        keras.layers.Dense(fc2_dims, activation = 'relu'),
        keras.layers.Dense(n_actions, activation = None)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss = 'mse')
    
    return model
        
    
class Agent():
    def __init__(self,lr,n_actions,gamma,epsilon,batch_size,
                input_dims, epsilon_dec = 1e-4, epsilon_end = 0.01,
                mem_size = 100000, fname = 'dqn_model.h5', alpha = 0.2):
        self.action = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.decrease_epsilon = epsilon_dec
        self.eps_reduction = 0.9999
        self.memory = ReplyBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims,128, 128)
        self.action_space =  np.arange(n_actions, dtype = np.int32)
        
        self.alpha = alpha
        
    def store_transition(self, state, action, reward, new_state, done):
        # to store (s,a,r,s') in to buffer
        self.memory.store_transition(state,action,reward,new_state,done)
        
    def choose_action(self, observation): #observation of environment
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            
        else:
            state = np.array([observation]) #to match with the input dimension of deep Q network
            actions = self.q_eval.predict(state) #similar to model.predict
            
            action = np.argmax(actions)  
        
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:  # to learn at least we need to have samples of batch size
            return
        
        #sample 'batch_size' number of (s,a,r,s') pairs
        states, actions, rewards, new_states, done = self.memory.sample_buffer(self.batch_size)
        
        # use the neural network to predict the action for all the states and also for all the new_states
        q_eval1 = self.q_eval.predict(states)
        q_next = self.q_eval.predict(new_states)
        
        
        q_target = np.copy(q_eval1)
        batch_index = np.arange(self.batch_size, dtype = np.int32)
        
        q_target[batch_index, actions] +=  self.alpha * (rewards + self.gamma *np.max(q_next, axis =1)*(1-done) - q_target[batch_index, actions]) #why dones
        
        # One case when it might be nice to use train_on_batch is for updating a pre-trained model 
        # on a single new batch of samples. Suppose you've already trained and deployed a model, 
        # and sometime later you've received a new set of training samples previously never used. 
        # You could use train_on_batch to directly update the existing model only on those samples. 
        # Other methods can do this too, but it is rather explicit to use train_on_batch for thiscase.
        
        #model.fit will train 1 or more epochs. That means it will train multiple batches. 
        #model.train_on_batch, as the name implies, trains only one batch.
        self.q_eval.train_on_batch(states, q_target)
        
        # temp = self.decrease_epsilon if self.epsilon > self.min_epsilon else 0
        # self.epsilon = self.epsilon - temp
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_reduction)
        
        
    def save_model(self):
        self.q_eval.save(self.model_file)
        
    def load_model(self):
        self.q_eval = load_model(self.model_file)
        
        
# %%

import gym

env = gym.make('CartPole-v0')
lr = 0.001
n_games = 1000
agent = Agent(gamma = 0.9, epsilon=1, lr = lr, alpha = 0.2, input_dims = env.observation_space.shape, n_actions = env.action_space.n, mem_size = 100000, batch_size = 64, epsilon_end = 0.01)
scores = []
eps_history = []

for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        if (i%100 == 0):
            env.render()
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, next_observation, done)
        observation = next_observation
        agent.learn()
    eps_history.append(agent.epsilon)
    scores.append(score)
    
    avg_score = np.mean(scores[-100:])
    print('episode: ', i, 'score %.2f' %score, 'average %.2f' %avg_score, 'epsilon %.3f' %agent.epsilon)


# %%

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

filename = 'CartPole.png'
x = [i+1 for i in range(n_games)]

# plot_learning_curve(x,scores,eps_history,filename)

# x = [i+1 for i in range(n_games)]

# plotLearning(x,scores,eps_history,filename)
plt.plot(x, scores)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
plt.savefig('saved_figure.png')

# %%

for i in range(10):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = agent.choose_action(observation)
        
        next_observation, reward, done, info = env.step(action)
        score += reward

        observation = next_observation
        
    print(score)