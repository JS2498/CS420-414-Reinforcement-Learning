# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:25:00 2022

@author: Jayanth S
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import random


# %%

""" 
Reference : https://www.samyzaf.com/ML/rl/qmaze.html


""" 

# In the below matrix '0' represents wall and '1' represent the free cells
maze = np.array([[1,0,1,1,1],
                [1,0,1,0,1],
                [1,1,1,0,1],
                [0,0,1,0,1],
                [1,1,1,0,1]])

# maze = np.array([[1,0,1,1,1,0,1],
#                 [1,0,1,0,1,0,1],
#                 [1,1,1,0,1,0,1],
#                 [0,0,1,0,1,0,1],
#                 [1,1,1,0,1,1,1]])


maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

class MDP_Maze:
    def __init__(self, maze, init_pos = [0,0], random_action_prob = 0.01, terminal_state = (maze.shape[1], maze.shape[0]) ):
        """
        Setup the MAZE environemnt 
        
        Arguments :
            maze -- A 2D matix with each entry either being '0' (represents wall) 
                    or '1' (represents free cell)
            init_pos -- Initial position of the robot
        """
        
        self.maze = maze
        
        self.maze_size = self.maze.shape
        
        # Starting point of the maze problem
        self.start_state = init_pos
        self.state = init_pos
        self.reward = 0
        self.is_terminal = 0 # 1 : for terminal state
        
        # End point of the maze problem
        self.terminal_state =  terminal_state #(6,0) #(self.maze_size[0]-1, self.maze_size[1]-1)
        
        # Store all the free cells through which the robot can move in a list
        
        #Remember that the grid world represents the (x,y) co-ordinates not the
        #row and column indexes hence the flip in the below list
        self.free_cells = [[c,r] for r in range(self.maze_size[0]) for c in range(self.maze_size[1]) if self.maze[r,c]==1]
        # self.free_cells.remove(list(self.terminal_state))
        
        self.n_states = len(self.free_cells)    # no. of valid states
        self.n_actions = 4 #no. of valid actions
    
        # create a dictionary with key : index value of free cell
        #                          value : (x,y) coordinates of a cell
        self.index_state = dict(enumerate(self.free_cells))
        
        self.state_index = dict((str(v), k) for k, v in self.index_state.items())
        
        # Probability Transition Matrix for the given problem 
        # Dimension : n_actions x n_states x n_states
        self.PTM = np.zeros((self.n_actions, self.n_states, self.n_states))
        
        # Reward Matrix for the given problem 
        # Dimension : n_actions x n_states x n_states
        self.reward_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        
        self.random_action_prob = random_action_prob
        
    def reset(self):
        """
        Method that has to called initially
        
        Retruns:
            Starting state of the environment
        """
        
        start_state = random.choice(self.free_cells)
        return start_state 
        
        
    def step(self, action, state):
        """
        A step taken in the environment
        
        Arguments :
            action : Action taken by the agent
            
                action = 0 ==> left
                action = 1 ==> up 
                action = 2 ==> right
                action = 3 ==> down
            
            state : 2D input representing the (x,y) co-ordinates of the robot
            
        Retruns :
            (reward, next_state, is_terminal)
            
            
        Reward Structure :
            * Movement to an adjacent state will cost = 0.1
            * Attempt to move towards a wall will cost = 0.75 and resulting state 
              remains same as the starting state
            * Attempt to move towards boundary will cost = 0.8 and resulting state
              remains same as the starting state
            * Reaching the terminal state will result in a reward = 10
        """
        
        # Special case when the state is a terminal state
        # Any action from terminal state will result in terminal state itself with reward 0
        if (state[0] == self.terminal_state[0]) and (state[1] == self.terminal_state[1]):
            # or use self.state == self.maze_size - 1 if both are arrays
            self.state = np.copy(state)
            self.reward = 0
            self.is_terminal = 1 
            
            return  (state, self.reward, self.is_terminal)
        
        
        next_state = np.copy(state)
        next_state = list(next_state)
        reward = -0.1
        is_terminal = 0
        
        
        if action == 0:
            next_state[0] -= 1 #decrement the x-cordinate
            
        elif action == 1:
            next_state[1] += 1 #increment the y-cordinate
            
        elif action == 2:
            next_state[0] += 1 #increment the x-cordinate
            
        elif action == 3:
            next_state[1] -= 1  # decrement the y-cordinate
        
        
        # print(self.state, state)
        # check if the robot is going out of boundary
        
        if (next_state[0]) < 0 or (next_state[0] >= self.maze_size[0]) :
            next_state[0] = state[0]
            reward = -0.8
        
        if next_state[1] < 0 or next_state[1] >= self.maze_size[1]  :
            next_state[1] = state[1]
            reward = -0.8
            
        if next_state not in self.free_cells :
            next_state = state
            reward = -0.75

        if (next_state[0] == self.terminal_state[0]) and (next_state[1] == self.terminal_state[1]):
            # or use self.state == self.maze_size - 1 if both are arrays
            reward = 10
            is_terminal = 1
        
        self.state = next_state
        self.reward = reward
        self.is_terminal = is_terminal
        
        return (next_state, reward, is_terminal)
    
    # def ProbReward_Matrices(self):
    
    #     """
    #     Returns :
    #         PTM : Probability transition matrix (|A|*|S|*|S|)
    #         reward_matrix : Reward matrix (|A|*|S|*|S|)
    #     """
        
    #     self.PTM = np.zeros((self.n_actions, self.n_states, self.n_states))
    #     self.reward_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        
    #     for s in range(self.n_states):
    #         for a in range(self.n_actions):
                
    #             # obtain the next state and the reward when action 'a' is taken
    #             # in state 's'
                
    #             (s_, reward, is_terminal) = self.env_step(a, self.index_state[s])
                
    #             s_ = self.state_index[str(s_)] # index of the state
                
    #             # when action 'a' is taken we move to state 's_' with probability
    #             # some probability <= 1. And with remaining probability we may 
    #             # move in some other direction
                
    #             self.PTM[a,s, s_] = 1 - self.random_action_prob  
    #             self.reward_matrix[a, s, s_] = reward
    #             for rand_action in range(self.n_actions):
    #                 if rand_action != a:
    #                     (s_, reward, is_terminal) = self.env_step(rand_action, self.index_state[s])
    #                     s_ = self.state_index[str(s_)]
                        
    #                     self.PTM[a,s,s_] += self.random_action_prob/(self.n_actions-1)
    #                     self.reward_matrix[a, s, s_] += reward
                        
    #     return self.PTM, self.reward_matrix
        
# %%

class OnPolicy_MonteCarlo:
    def __init__(self, env, num_states, epsilon=0.2, n_actions=3, learning_rate=0.01, gamma=0.999):
        """
        Arguments :
            env : environment (maze)
            num_states : no. of states
            epsilon : initial value of epsilon for epsilon-greedy policy
            n_actions : no. of possible actions
            learning_rate : learning rate in the update rule
            gamma : discount factor
        """
        self.env = env
        
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.eps_reduction = 0.9
        
        self.gamma = gamma
        
        self.lr = learning_rate
        
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states, n_actions))
        terminal_state_index = env.state_index[str(list(env.terminal_state))]
        self.Q[terminal_state_index,:] = 0
        
        self.count = np.zeros(shape = (env.n_states, env.n_actions))
       

    def choose_action(self, state):
        """
        Arguments :
            state : state index 
        
        Return:
            epsilon-greedy action for the given state
        """
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.Q[state, :])
            
        return action
    
    def learn(self, episode):
        """
        Arguments :
           
        Update the Q table according to Monte Carlo update rule
            
        """
        T = len(episode)
        G = 0
        
        states_visited = []
        for i in range(T):
            state = episode[i][0]
            states_visited.append(state)
        for i in range(T):
            state = episode[T-1-i][0]
            action = episode[T-1-i][1]
            reward = episode[T-1-i][2]
            
            
            G = self.gamma * G + reward
            
            # if state is not in states_visited[0:T-2-i]:
            self.count[state, action] += 1
            self.Q[state, action] +=  (G - self.Q[state,action]) / self.count[state, action]
            
    def policy_value(self):
        V = np.zeros((self.env.n_states,1))
        pi = np.ones((self.env.n_states,1))
        
        for s in range(n_states):
            pi[s] = np.argmax(self.Q[s,:])
            V[s] = np.max(self.Q[s,:])
            
        return V, pi
    

class OffPolicy_MonteCarlo:
    def __init__(self, env, num_states, epsilon=0.2, n_actions=3, learning_rate=0.01, gamma=0.999):
        """
        Arguments :
            env : environment (maze)
            num_states : no. of states
            epsilon : initial value of epsilon for epsilon-greedy policy
            n_actions : no. of possible actions
            learning_rate : learning rate in the update rule
            gamma : discount factor
        """
        self.env = env
        
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.eps_reduction = 0.9
        
        self.gamma = gamma
        
        self.lr = learning_rate
        
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states, n_actions))
        terminal_state_index = env.state_index[str(list(env.terminal_state))]
        self.Q[terminal_state_index,:] = 0
        
        self.count = np.zeros(shape = (env.n_states, env.n_actions))
        
        self.pi = np.zeros((num_states,1))
        
        self.weight = 1
       

    def choose_action(self, state):
        """
        Arguments :
            state : state index 
        
        Return:
            epsilon-greedy action for the given state
        """
        
        if np.random.random() < 1:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.Q[state, :])
            
        return action
    
    def learn(self, episode):
        """
        Arguments :
           
        Update the Q table according to Monte Carlo update rule
            
        """
        T = len(episode)
        G = 0
        
        states_visited = []
        for i in range(T):
            state = episode[i][0]
            states_visited.append(state)
            
        self.weight = 1
        for i in range(T):
            state = episode[T-1-i][0]
            action = episode[T-1-i][1]
            reward = episode[T-1-i][2]
            
            
            G = self.gamma * G + reward
            
            # if state is not in states_visited[0:T-2-i]:
            self.count[state, action] += self.weight
            self.Q[state, action] +=  ((G - self.Q[state,action]) * self.weight) / self.count[state, action]
            
            self.pi[state] = np.argmax(self.Q[state,:])
            
            if action != self.pi[state]:
                break
            
            self.weight = self.weight*(len(self.action_space))
            
            
    def policy_value(self):
        V = np.zeros((self.env.n_states,1))
        pi = np.ones((self.env.n_states,1))
        
        for s in range(n_states):
            pi[s] = np.argmax(self.Q[s,:])
            V[s] = np.max(self.Q[s,:])
            
        return V, pi
            
# %%


class SARSA:
    def __init__(self, env, num_states, epsilon=0.2, n_actions=3, learning_rate=0.01, gamma=0.99):
        """
        Arguments :
            env : environment (maze)
            num_states : no. of states
            epsilon : initial value of epsilon for epsilon-greedy policy
            n_actions : no. of possible actions
            learning_rate : learning rate in the update rule
            gamma : discount factor
        """
        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.eps_reduction = 0.999

        
        self.gamma = gamma
        
        self.lr = learning_rate
        
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states, n_actions))
        terminal_state_index = env.state_index[str(list(env.terminal_state))]
        self.Q[terminal_state_index,:] = 0
       
    def choose_action(self, state):
        """
        Arguments :
            state : state index 
        
        Return:
            epsilon-greedy action for the given state
        """
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.Q[state, :])
            
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """
        Arguments :
            state : index of present state
            action: action taken by the agent according to epsilon-greedy policy
            reward: reward obtained from environment for the (state-action) pair
            next_state : index of next state
            done : 1 if next state is terminal else 0
            
        Update the Q table according to SARSA update rule
            
        """
        next_action = self.choose_action(next_observation)
        Q_next = self.Q[next_observation, next_action]
        self.Q[state, action] += self.lr * (reward + (1-done)*self.gamma * Q_next - agent.Q[observation, action])
    
    def policy_value(self):
        """
        Returns the value corresponding to the deterministic policy and also the policy
        """
        V = np.zeros((self.env.n_states,1))
        pi = np.ones((self.env.n_states,1))
        
        for s in range(n_states):
            pi[s] = np.argmax(self.Q[s,:])
            V[s] = np.max(self.Q[s,:])
            
        return V, pi

# %%


class Qlearning:
    def __init__(self, env, num_states, epsilon=0.2, n_actions=3, learning_rate=0.01, gamma=0.99):
        """
        Arguments :
            env : environment (maze)
            num_states : no. of states
            epsilon : initial value of epsilon for epsilon-greedy policy
            n_actions : no. of possible actions
            learning_rate : learning rate in the update rule
            gamma : discount factor
        """
        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        
        self.gamma = gamma
        
        self.lr = learning_rate
        
        self.action_space = np.arange(n_actions)
        self.Q = np.random.uniform(size=(num_states, n_actions))
        terminal_state_index = env.state_index[str(list(env.terminal_state))]
        self.Q[terminal_state_index,:] = 0
        self.eps_reduction = (self.epsilon - self.epsilon_min)/2000

    def choose_action(self, state):
        """
        Arguments :
            state : state index 
        
        Return:
            epsilon-greedy action for the given state
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        else:
            action = np.argmax(self.Q[state, :])
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """
        Arguments :
            state : index of present state
            action: action taken by the agent according to epsilon-greedy policy
            reward: reward obtained from environment for the (state-action) pair
            next_state : index of next state
            done : 1 if next state is terminal else 0
            
        Update the Q table according to Q-update rule
        """
        
        self.Q[state, action] += self.lr * (reward + (1-done)*self.gamma * np.max(agent.Q[next_state,:]) - agent.Q[observation, action])

    
    def policy_value(self):
        V = np.zeros((self.env.n_states,1))
        pi = np.ones((self.env.n_states,1))
        
        for s in range(n_states):
            pi[s] = np.argmax(self.Q[s,:])
            V[s] = np.max(self.Q[s,:])
            
        return V, pi
            
        
# %%

# Function to plot the grid world

def show(maze, mdp_maze, algorithm):
    
    """
    maze : 2D matrix representing with 0 : walls and 1 : free cells
    mdp_maze : Object corresponding to the grid world problem
    algorithm : Dictionary with 'V' : Optimal state values
                                'pi' : Optimal policy
                                'algorithm' : value/policy iteration
    """
    action_dict = {0 : '<', 
                   1 : '^',
                   2 : '>',
                   3 : 'v'}
    
    V = algorithm['V']
    pi = algorithm['pi']
    algorithm_name = algorithm['algorithm']
    # plt.grid('on')
    # ax = plt.gca()
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].set_xticks(np.arange(0.5, maze.shape[0], 1))
    ax[0].set_yticks(np.arange(0.5, maze.shape[1], 1))
    
    maze[mdp_maze.start_state[0], mdp_maze.start_state[1]] = 0.2
    maze[mdp_maze.terminal_state] = 0.2
    canvas = maze[-1:0:-1]
    canvas = np.concatenate((canvas, [maze[0,:]]), axis = 0)
    
    ax[0].imshow(canvas)
    ax[1].imshow(canvas)
    
    # img = sns.heatmap(canvas,  cmap='inferno')
    free_cells = np.array(mdp_maze.free_cells)
    for i  in range(len(mdp_maze.free_cells)):
        ax[0].text(free_cells[i,0], canvas.shape[0]-1- free_cells[i,1], round(V[i,0],2), ha = 'center', va = 'center')
        ax[1].text(free_cells[i,0], canvas.shape[0]-1- free_cells[i,1], action_dict[int(pi[i])], ha = 'center', va = 'center')
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    plt.suptitle(algorithm_name, fontsize = '16')
    plt.savefig(f'Value_poily_of_{algorithm_name}.png')
    plt.show()

# %%

def Trajectory(start_pos, maze, mdp_maze, algorithm):
    
    """
    start_pos : (x,y) co-ordinates corresponding to the starting state
                of the trajectory
    maze : 2D matrix representing with 0 : walls and 1 : free cells
    mdp_maze : Object corresponding to the grid world problem
    algorithm : Dictionary with 'V' : Optimal state values
                                'pi' : Optimal policy
                                'algorithm' : value/policy iteration
    """
    
    action_dict = {0 : '<', 
                   1 : '^',
                   2 : '>',
                   3 : 'v'}
    # V = algorithm['V']
    pi = algorithm['pi']

    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, maze.shape[0], 1))
    ax.set_yticks(np.arange(0.5, maze.shape[1], 1))
    
    
    canvas = maze[-1:0:-1]
    canvas = np.concatenate((canvas, [maze[0,:]]), axis = 0)*150
    next_state = start_pos
    # print(canvas)
    # ax.imshow(canvas)  
    canvas[canvas.shape[0]-1-next_state[1],next_state[0]] = 50
    is_terminal = 0
    # free_cells = np.array(mdp_maze.free_cells)
    while(is_terminal!=1):
        state_indx = mdp_maze.state_index[str(next_state)] 
        ax.text(next_state[0], 7-next_state[1], action_dict[int(pi[state_indx])], ha = 'center', va = 'center')
        (next_state, reward, is_terminal) = mdp_maze.env_step(pi[state_indx], next_state)
        canvas[canvas.shape[0]-1-next_state[1],next_state[0]] = 80
       
    canvas[canvas.shape[0]-1-next_state[1],next_state[0]] = 50
    ax.imshow(canvas)  
    # print(canvas)
    # img = sns.heatmap(canvas,  cmap='inferno')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title(f"A sample trajectory from (x ={start_pos[0]}, y = {start_pos[1]})", fontsize = '5')
    plt.savefig('Sample_Trajectory.png')
    plt.show()

#%% 


lr = 0.01  # learning rate
gamma = 0.99
n_games = 100000
epsilon = 0.98
scores = []
avg_scores = []

# # instantiate the grid world problem
env = MDP_Maze(maze, terminal_state= (7,7))


state_dict = dict(enumerate(env.free_cells))
n_states = len(state_dict)


agent = SARSA(env, epsilon=epsilon, n_actions=env.n_actions, num_states=n_states)

for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()  # initalizes the starting state of the car
    observation = env.state_index[str(observation)]
    episode = []

    while not done:

        action = agent.choose_action(observation)
        next_observation, reward, done = env.step(action, env.index_state[observation])
        
        next_observation = env.state_index[str(next_observation)]
        
        episode.append((observation, action, reward))

        score += reward

        agent.learn(observation,action, reward, next_observation, done)
        
        # print(observation, next_observation, action)
        observation = next_observation
        
        if score < -200:
            break
        
    # agent.learn(episode)
    agent.epsilon = np.max((agent.epsilon*agent.eps_reduction, agent.epsilon_min))
    scores.append(score)
    avg_scores.append(np.mean(scores))

    if (i % 5000 == 0):
        print(f"The average reward is {np.mean(scores[-100:]):.3f} and the episode is {i}")
        V, pi = agent.policy_value()
        Q_learning_final ={'V' : V, 'pi' : pi, 'algorithm' : 'SARSA'}
        # print(f"The time it took policy iteration to run is : {end-start:.3f}")
        show(maze, env, Q_learning_final)
        
    # if (np.mean(scores[-100:])>-110):
    #     i = n_games
               

V, pi = agent.policy_value()
Q_learning_final ={'V' : V, 'pi' : pi, 'algorithm' : 'SARSA'}
# print(f"The time it took policy iteration to run is : {end-start:.3f}")
show(maze, env, Q_learning_final)

# %%
x = [i+1 for i in range(n_games)]
plt.figure(figsize = (6,6))
plt.plot(x, avg_scores)
plt.xlabel("Epsiodes")
plt.ylabel("Running average score over last 100 episodes ")
plt.title("SARSA algorithm")


        
# %%

"""
Tasks yet to be completed :
    1. Modify the Q-learning and SARSA algorithms for grid-world
    2. Implement Monte-Carlo Every Visit On-policy control
    3. Implement Monte-Carlo First Visit On-policy control
    4. Implement Monte-Carlo Every Visit Off-policy control
    5. Implement Monte-Carlo First Visit Off-policy control
    
"""
        
        
        
        
            

        
        
        
        
        
        
        
        
        
        
        