#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 03:39:27 2022

@author: jayanths
"""

import os
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras import models, Model


# %%

class ACNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                 name="AC_network", chkpt_dir="Actor_critic"):
        # create a directory with the same name as 'chkpt_dir' to store the
        # weights of the model

        # Python has super function which allows us to access temporary object
        # of the super class. Use of super class is, we need not use the base
        # class name explicitly and it also helps in working with
        # multiple inheritance (main advantage).
        super(ACNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.n_actions = n_actions
        self.model_name = name  #don't use name which is reserved for the base class

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        # no need to specify the input dimension because keras will infer that when a input is given
        self.fc1 = tf.keras.layers.Dense(self.fc1_dims, activation = 'relu', name = 'fc1')
        self.fc2 = tf.keras.layers.Dense(self.fc2_dims, activation = 'relu', name = 'fc2')

        # we use the same model architecture for both state values and the policy
        self.v = tf.keras.layers.Dense(1, activation=None, name='state_values')
        self.pi = tf.keras.layers.Dense(n_actions, activation='softmax', name='action_prob')


    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(state)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi


# %%

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        # self.alpha = alpha
        self.n_actions = n_actions

        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        # Build the actor_critic model
        self.actor_critic = ACNetwork(n_actions)
        self.actor_critic.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = alpha), loss = self.custom_loss)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation]) #the extra [] is because NN takes input as a batch
        val, probs = self.actor_critic(state)

        # np.choices can also be used
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample() #sample of the distribution defined above
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        # gym is only compatiable with numpy hence convert tensor to numpy value and [0] is because of additional
        # batch dimension we added in the 1st line of the function
        return action.numpy()[0] 

    def save_model(self):

        print('saving models.............')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_model(self):

        print('..........loding models')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)


    def learn(self, state, reward, next_state, done):

        # just to be safe
        state = tf.convert_to_tensor([state], dtype = tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype = tf.float32)
        reward = tf.convert_to_tensor(reward, dtype = tf.float32)
        
        # Gradient tape is used to record ("tape") a sequence of operations performed upon some input and 
        # producing some output, so that the output can be differentiated with respect to the input
        with tf.GradientTape() as tape:
            
            # the gradient tape will dispose the operations/stored values after the calculation of gradients. 
            #By default, the resources held by GradientTape are released as soon as GradientTape.gradient() method 
            #is called. To compute multiple gradients over the same computation, create a persistent gradient tape. 
            #This allows multiple calls to the gradient() method as resources are released when the tape object is 
            #garbage collected. Use 'del tape' to delete it (should be done manually)
            
            #persistent = True is not required here. Required if separate actor and critic networks are used
            state_value, probs = self.actor_critic(state)
            nxtstate_value, no_probs = self.actor_critic(next_state)
            
            state_value = tf.squeeze(state_value) #to get rid of batch dimension (loss works well with 1D tensors)
            nxtstate_value = tf.squeeze(nxtstate_value)
            
            action_probabilities = tfp.distributions.Categorical(probs = probs)
            #calculate the log probability of the action we have taken
            log_probs = action_probabilities.log_prob(self.action) 
            
            TD_error = reward + self.gamma*nxtstate_value*(1.0-int(done)) -state_value
#             updated_state_value = state_value + 0.2*TD_error #check if this works
#             actor_loss = - TD_error * log_probs
#             critic_loss =  TD_error * TD_error
            
#             #why are we trying to minimize the the total loss instead of individual loss. Check if
#             #individual loss with different architecture for actor and critic networks works well.
#             total_loss = actor_loss + critic_loss

            total_loss = self.custom_loss(TD_error, log_probs)
        gradients = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients([(grad,var) for (grad,var) in zip(gradients, self.actor_critic.trainable_variables) if grad is not None])
        
    def custom_loss(self, TD_error, log_probs ):
        actor_loss =  - TD_error * log_probs
        critic_loss = TD_error**2 #else use Huber loss

        #why are we trying to minimize the the total loss instead of individual loss. Check if
        #individual loss with different architecture for actor and critic networks works well.
        total_loss = actor_loss + critic_loss
        
        return total_loss

        
# %%

# import warnings; warnings.simplefilter('ignore')

env = gym.make('LunarLander-v2')
# print(env.action_space.__doc__)
agent = Agent(alpha = 1e-5, n_actions = env.action_space.n) #instantiate the agent

n_games = 10000

figure_file = './plots/LunarLander_AC.png'

best_score = env.reward_range[0] #lowest possible reward
print(env.reward_range)

score_history = [] #empty list to store the returns
load_checkpoint = False # True : if we have a pre-trained model

if load_checkpoint:
    agent.load_model()
    
for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        # env.render()-+
        action = agent.choose_action(observation)
        
        next_observation, reward, done, info = env.step(action)
        score += reward
        
        if not load_checkpoint:
            agent.learn(observation, reward, next_observation, done)
        observation = next_observation
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_model() #repalces the already saved model with best weights 
        
    print("Episode : ",i, ' score: %.1f' %score, ' avg_score: %.1f' %avg_score)

# %%

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

filename = 'LunarLander.png'
x = [i+1 for i in range(n_games)]

# plot_learning_curve(x,scores,eps_history,filename)

# x = [i+1 for i in range(n_games)]

# plotLearning(x,scores,eps_history,filename)
plt.plot(x, score_history)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
plt.savefig('saved_figure_AC.png')
    
# %%

avg_score_history = []
i =0
x = [i+1 for i in range(n_games-100)]
i=0
for i in range(n_games-100):
    avg_score_history.append(np.mean(score_history[i:i+100]))
    
plt.plot(x, avg_score_history)
plt.xlabel("Episodes")
plt.ylabel("Reward averaged over 100 epsiodes")
# plt.xlim([1,18000])
plt.show()

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
