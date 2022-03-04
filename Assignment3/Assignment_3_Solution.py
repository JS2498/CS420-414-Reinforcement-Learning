# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:25:00 2022

@author: Jayanth S
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time


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
        
    def env_start(self):
        """
        Method that has to called initially
        
        Retruns:
            Starting state of the environment
        """
        
        return self.state
        
        
    def env_step(self, action, state):
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
    
    def ProbReward_Matrices(self):
        
        """
        Returns :
            PTM : Probability transition matrix (|A|*|S|*|S|)
            reward_matrix : Reward matrix (|A|*|S|*|S|)
        """
        
        self.PTM = np.zeros((self.n_actions, self.n_states, self.n_states))
        self.reward_matrix = np.zeros((self.n_actions, self.n_states, self.n_states))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                # obtain the next state and the reward when action 'a' is taken
                # in state 's'
                
                (s_, reward, is_terminal) = self.env_step(a, self.index_state[s])
                
                s_ = self.state_index[str(s_)] # index of the state
                
                # when action 'a' is taken we move to state 's_' with probability
                # some probability <= 1. And with remaining probability we may 
                # move in some other direction
                
                self.PTM[a,s, s_] = 1 - self.random_action_prob  
                self.reward_matrix[a, s, s_] = reward
                for rand_action in range(self.n_actions):
                    if rand_action != a:
                        (s_, reward, is_terminal) = self.env_step(rand_action, self.index_state[s])
                        s_ = self.state_index[str(s_)]
                        
                        self.PTM[a,s,s_] += self.random_action_prob/(self.n_actions-1)
                        self.reward_matrix[a, s, s_] += reward
                        
        return self.PTM, self.reward_matrix
        
# %%

def Value_Iteration(Prob_Transition, Reward, discount, error_threshold):
    
    """
    Arguments :
        Prob_Transtion : Probability Transition Matrix (|A|*|S|*|S|)
        Reward : Reward matrix (|A|*|S|*|S|)
        Discount : Value between 0 to 1
        error_threshold : Defines when to stop the alogrithm
        
    Returns :
        V : Optimal state values obtained by value iteration
        pi : Optimal policy corresponding to opitmal state values
    """
    
    n_actions = Prob_Transition.shape[0]
    n_states = Prob_Transition.shape[-1]
    V = np.zeros((n_states,1))
    pi = np.zeros((n_states,1))
    
    itr = 0
    while(1):
        error = 0
        V_prev = np.copy(V)
        for s in range(n_states):
            # Vs = V[s]
            Q = np.zeros((n_actions,1))
            for a in range(n_actions):
                # next_state, reward, is_terminal = mdp_maze.env_step(a,state_dict[i]) 
                Prob = Prob_Transition[a,s,:] # The probability transition vector corresponding 
                # to the action 'a' in state 's'.
                Prob = np.reshape(Prob, newshape = (1,-1))
                
                # Rewards (vector) associated with state 's' and action 'a'
                reward = Reward[a,s,:]
                reward = np.reshape(reward, newshape = (-1,1))
                
                # Calculate Q_pi(s,a)
                Q[a] = Prob @ (reward + discount * V)
           
            # State value is the max of the state-action values
            V[s] = np.max(Q)
            
            # improve the policy which corresponds to max stat-action value
            pi[s] = np.argmax(Q)
            
        
        # Find the difference between the state values for consecutive iterations
        # If the max difference among these is less than the predefined value
        # we say that the value iteration has converged to optimal value and hence optimal policy
        error = np.max(np.abs(V_prev - V)) if  np.max(np.abs(V_prev - V)) >  error else error
        itr += 1
        if error < error_threshold:
            print(f"The error in iteration {itr} is : {error} which is less than threshold")
            print("Value iteration has converged")
            break
            
        
    return(V, pi)

# %%        

def Policy_Iteration(Prob_Transition, Reward, discount, error_threshold):
    
    """
    Arguments :
        Prob_Transtion : Probability Transition Matrix (|A|*|S|*|S|)
        Reward : Reward matrix (|A|*|S|*|S|)
        Discount : Value between 0 to 1
        error_threshold : Defines when to stop the alogrithm
        
    Returns :
        
        pi : Optimal policy obtained by the policy iteration
        V : Optimal state values obtained using optimal policy
    """
    
    n_actions = Prob_Transition.shape[0]
    n_states = Prob_Transition.shape[-1]
    V = np.zeros((n_states,1))
    pi = np.ones((n_states,1)) #np.random.randint(0,4,n_states)
    #this worked better that random initialization for this problem
    
    itr = 0
    while(1):
        
        # Policy evaluation step
        while(1):
            error = 0
            V_prev = np.copy(V)
            for s in range(n_states):
                action = int(pi[s])
                # print(action)
                Prob = Prob_Transition[action,s,:]
                Prob = np.reshape(Prob, newshape = (1,-1))
                reward = Reward[action,s,:]
                reward = np.reshape(reward, newshape = (-1,1))
                # print(Prob.shape, reward.shape)
                V[s] = Prob @ (reward + discount * V)
                    
            error = np.max(np.abs(V_prev - V)) if  np.max(np.abs(V_prev - V)) >  error else error
            itr += 1
            if error < error_threshold:
                # print(f"The error in iteration {itr} is : {error} which is less than threshold")
                break
        
        # Policy improvement step
        policy_stable = 1
        error = 0
        pi_prev = np.copy(pi)
        for s in range(n_states):
            # Vs = V[s]
            Q = np.zeros((n_actions,1))
            for a in range(n_actions):
               
                Prob = Prob_Transition[a,s,:]
                Prob = np.reshape(Prob, newshape = (1,-1))
                reward = Reward[a,s,:]
                reward = np.reshape(reward, newshape = (-1,1))
                Q[a] = Prob @ (reward + discount * V)
            # print(Prob.shape, reward.shape)
            # V[s] = Prob @ (reward + discount * V)
            # V[s] = np.max(Q)
            
            pi[s] = np.argmax(Q)
        

        error = np.sum(pi_prev != pi)
        if error > 0 :
            policy_stable = 0

        itr += 1
        # when there is no policy improvement we say the policy iteration algorithm
        # has converged
        if policy_stable == 1:
            print("Policy iteration has converged")
            break
        
    return(V, pi)

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

# %%

# General Setting
discount = 0.99 #discount factor
error_threshold = 0.00001 # threshold to check the convergence
        
# instantiate the grid world problem
mdp_maze = MDP_Maze(maze, terminal_state= (7,7))

PTM, reward_matrix = mdp_maze.ProbReward_Matrices()


state_dict = dict(enumerate(mdp_maze.free_cells))
n_states = len(state_dict)

# Solve the problem using Value iteration
start = time.time()
V, pi = Value_Iteration(PTM, reward_matrix, discount, error_threshold)
end = time.time()
value_iteration = {'V' : V, 'pi' : pi, 'algorithm' : 'Value Iteration'}
print(f"The time it took value iteration to run is : {end-start:.3f}")
show(maze, mdp_maze, value_iteration)

# Solve the MDP problem using Policy iteration
start = time.time()
V, pi = Policy_Iteration(PTM, reward_matrix, discount, error_threshold)
end = time.time()
policy_iteration ={'V' : V, 'pi' : pi, 'algorithm' : 'Policy Iteration'}
print(f"The time it took policy iteration to run is : {end-start:.3f}")
show(maze, mdp_maze, policy_iteration)

# A particular trajectory with starting state : 'start_pos'
start_pos = [0,0]
Trajectory(start_pos, maze, mdp_maze, value_iteration)


        
# %%

"""
Tasks yet to be completed :
    1. Plot the maze with the values and policies (if possible the arrows) (1hr) (Done)
    2. Check the time requirements for VI and PI (30 min) (Done)
    3. Make a report for VI and PI (2 hrs) 
    4. Vary the discount factor and check how the policy changes (30 min) 
    (Done... Better to have discount factor close to 1 )
    5. Vary the probability of error in taking the action (10 min) (not required)
    6. Check the variations of PI and VI and also CS234 homework (2hr)
    (Check modified policy iteration when you are free )
"""
        
        
        
        
            

        
        
        
        
        
        
        
        
        
        
        