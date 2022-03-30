#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 06:16:42 2022

@author: jayanths
"""


# %%
import numpy as np
import gym
import matplotlib.pyplot as plt

# %%


class MonteCarlo:
    def __init__(self, num_states, epsilon=1, n_actions=3):
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.eps_reduction = 0.99999  #(self.epsilon - self.epsilon_min)/400000

        self.action_space = np.arange(n_actions)

        self.Q = np.random.uniform(size=(num_states[0], num_states[1], n_actions))
        self.Q[-1, :, :] = 0

        self.count = np.zeros(shape=(num_states[0], num_states[1], n_actions))

        self.gamma = 1
        self.alpha = 0.001

    def update_Qvalues(self, episode):
        T = len(episode)
        G = 0
        states_visited = []
        for i in range(T):
            states_visited.append(list(episode[T-1-i][0]))

        # print(states_visited)
        for i in range(T):
            # print(i,T)
            state = episode[T-1-i][0]
            action = episode[T-1-i][1]
            reward = episode[T-1-i][2]
            # print(list(state))


            G =  self.gamma * G + reward

            if list(state) not in states_visited[:T-2-i] :
                # print("hi")
                self.count[state[0], state[1], action] += 1
                self.Q[state[0], state[1], action] =   (G - self.Q[state[0], state[1], action])/self.count[state[0], state[1], action]
        # print(G)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            action = np.argmax(self.Q[state[0], state[1], :])
        return action

# %%

class OffPolicy_MonteCarlo:
    def __init__(self, num_states, epsilon=1, n_actions=3):
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.eps_reduction = 0.9999  #(self.epsilon - self.epsilon_min)/400000

        self.action_space = np.arange(n_actions)

        self.Q = np.random.uniform(size=(num_states[0], num_states[1], n_actions))
        self.Q[-1, :, :] = 0

        self.pi = np.zeros(shape = (num_states[0], num_states[1], n_actions))

        self.count = np.zeros(shape=(num_states[0], num_states[1], n_actions))

        self.gamma = 1
        self.alpha = 0.001

    def update_Qvalues(self, episode):
        T = len(episode)
        G = 0
        states_visited = []
        for i in range(T):
            states_visited.append(list(episode[T-1-i][0]))

        # print(states_visited)
        # w = 1
        for i in range(T):
            # print(i,T)
            state = episode[T-1-i][0]
            action = episode[T-1-i][1]
            reward = episode[T-1-i][2]
            # print(list(state))

            G =  self.gamma * G + reward

            # if list(state) not in states_visited[:T-2-i] :
                # print("hi")
            self.count[state[0], state[1], action] += 1
            self.Q[state[0], state[1], action] =   ((G - self.Q[state[0], state[1], action]))/self.count[state[0], state[1], action]
            self.pi[state] = np.argmax(self.Q[state, :])

            # if action != self.pi[state]:
                # break

            # w = w*(len(self.action_space))
        # print(G)
    def choose_action(self, state):
        if np.random.random() < 1:
            action = np.random.choice(self.action_space)

        else:
            action = np.argmax(self.Q[state[0], state[1], :])
        return action

    def target_policy(self, s):
        return np.argmax(self.Q[s, :])


# %%


class SARSA:
    def __init__(self):
        pass

    def choose_action(self, state):
        pass

# %%


class Qlearning:
    def __init__(self,  num_states, epsilon=0.2, n_actions=3):
        # print(n_actions)
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states[0], num_states[1], n_actions))
        self.Q[-1, :, :] = 0
        self.eps_reduction = (self.epsilon - self.epsilon_min)/2000

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            action = np.argmax(self.Q[state[0], state[1], :])
        return action


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

# %%


env = gym.make("MountainCar-v0")  # specify the environment you wish to have
env._max_episodes_steps = 500
env._max_episode_steps = 500

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
n_games = 1000000
epsilon = 1
scores = []
avg_scores = []

# specify the precision you require while discretizing the state space
pos_precision = 0.09
vel_precision = 0.007

num_states = (env.observation_space.high - env.observation_space.low)/([pos_precision, vel_precision])
num_states = num_states.astype(int)

agent = MonteCarlo(epsilon=epsilon, n_actions=env.action_space.n, num_states=num_states)

for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    discrete_obs = discretize_obs(observation)
    episode = []

    while not done:
        if i%5001 == 0:
            # print(f"Episode is {i}")
            env.render()
        action = agent.choose_action(discrete_obs)
        next_observation, reward, done, info = env.step(action)
        discrete_next = discretize_obs(next_observation)
        # print(discrete_next, reward, action)

        if next_observation[0] > 0.05:
            reward += np.exp(next_observation[0]*5)
        if done:
            reward += 100
        score += reward


        # observation = np.copy(next_observation)

        episode.append((discrete_obs, action, reward))
        # Update step for Q-learning
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(agent.Q[discrete_next[0], discrete_next[1],:]) - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # Update step for SARSA
        # next_action = agent.choose_action(discrete_next)
        # Q_next = agent.Q[discrete_next[0], discrete_next[1], next_action]
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # if i % 2000 == 0:
        #     print(next_observation, discrete_next, discrete_obs, action)
        discrete_obs = np.copy(discrete_next)

    agent.update_Qvalues(episode)
    agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))
    scores.append(score)
    avg_scores.append(np.mean(scores))

    if (i % 5000 == 0):
        print(f"The average reward is {np.mean(scores[-100:])} and epsiode is {i}")

    # if (np.mean(scores[-100:])>-110):
    #     i = n_games


x = [i+1 for i in range(n_games)]
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Score")


env.close()


# %%

# Test the trained model

n_games = 20


for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    discrete_obs = discretize_obs(observation)
    episodes_length = []
    episode_length = 0
    # episode = []

    while not done:
        # if i%5001 == 0:
            # print(f"Episode is {i}")
        env.render()
        
        action = agent.choose_action(discrete_obs)
        next_observation, reward, done, info = env.step(action)
        discrete_next = discretize_obs(next_observation)
        # print(discrete_next, reward, action)

        if next_observation[0] > 0.05:
            reward += np.exp(next_observation[0]*5)
        if done:
            reward += 100
        score += reward

        # observation = np.copy(next_observation)

        # episode.append((discrete_obs, action, reward))

        # Update step for Q-learning
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * np.max(agent.Q[discrete_next[0], discrete_next[1],:]) - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # Update step for SARSA
        # next_action = agent.choose_action(discrete_next)
        # Q_next = agent.Q[discrete_next[0], discrete_next[1], next_action]
        # agent.Q[discrete_obs[0], discrete_obs[1], action] += lr * (reward + (1-done)*gamma * Q_next  - agent.Q[discrete_obs[0], discrete_obs[1], action])

        # if i % 2000 == 0:
        #     print(next_observation, discrete_next, discrete_obs, action)
        discrete_obs = np.copy(discrete_next)
        episode_length += 1

    episodes_length.append(episode_length)
    # agent.update_Qvalues(episode)
    # agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))
    scores.append(score)
    avg_scores.append(np.mean(scores[-10:]))

    # if (i % 5000 == 0):
    print(f"The epsiode is {i} with episode length {episode_length} and total reward {score: .2f}")

    # if (np.mean(scores[-100:])>-110):
    #     i = n_games

# %%

x = [i+1 for i in range(n_games)]
plt.plot(x, avg_scores[1000100:1000120])
plt.xlabel("Epsiodes")
plt.ylabel("Score")


env.close()

# %%

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

    * For Monte Carlo the On-policy every visit algortihm was not able to find
    a path/ trajectory to reach the destination. Hence, we changed the reward
    structure as follows:
        - For each step reward = -1
        - If the position in next state > 0.05, then a additional reward of
        exp{position of next state * 5} is given
        - If it reaches the destination then we gave a reward of 100

        - We also increased maximum number of episodes to 500
"""


"""
To do :
    - Modify the code for every visit monte carlo keeping the fist visit code
    as it is
    - Check if the code for Off-policy monte carlo works
    - Modify the reward structure for the Monte carlo methods such that the
    agent tries to reach the destination in minimium number of steps

    - Write down the report
"""
