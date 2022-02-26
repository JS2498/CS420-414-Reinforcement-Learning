# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 20:05:04 2022

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# %%
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',}
pylab.rcParams.update(params)

#%%
class MAB:
    
    def __init__ (self, reward_dist, bandits_mean, stdv = 0.01):
        
        self.reward_dist  = reward_dist
        self.bandits_mean = bandits_mean
        self.stdv         = stdv
        
    def pull_arm (self, k):

        '''
        pull the arm k and obtain the reward based on gaussian distribution
        return the reward and the regret associated with that arm
        '''
        if self.reward_dist == 'Bernoulli':
          return np.random.binomial(1,self.bandits_mean[k]), (np.max(self.bandits_mean) - self.bandits_mean[k])
        elif self.reward_dist == 'Gaussian':
          return np.random.normal(loc = self.bandits_mean[k], scale = self.stdv), (np.max(self.bandits_mean) - self.bandits_mean[k])

# class Gaussian_MAB:
    
#     def __init__(self, bandits_mean, bandits_variance):
#         self.bandits_mean = bandits_mean
#         if (len(bandits_variance) == 1):
#             self.bandits_variance = bandits_variance*len(self.bandits_mean)
#         else:
#             self.bandits_variance = bandits_variance
        
        
#     def pull_arm(self,k):
#         #pull the arm k and obtain the reward based on binomial distribution
#         #return the reward and the regret associated with that arm
#         mean = self.bandits_mean[k]
#         variance = self.bandits_variance[k]
#         return np.random.normal(loc = mean, scale = variance), (np.max(self.bandits_mean) - self.bandits_mean[k])

class Epsilon_Greedy:   
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon
    
    def action(self, reward_estimates, num_arms_played, n_bandits):
        # success_count = reward_array.sum(axis=1)
        # total_count = k_array.sum(axis = 1)
        # reward_estimate = success_count/(total_count+1)
        return np.argmax(reward_estimates) if (np.random.uniform(0,1)>self.epsilon) else np.random.randint(0,n_bandits)
        
class Variable_Epsilon_Greedy:
    
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon #initial epsilon value
        self.c = 0.2
        self.d = 0.05 #min. gap between the average reward of the optimal arm and a sub-optimal arm
    
    def action(self, reward_estimates, num_arms_played, n_bandits):
        # success_count = reward_array.sum(axis=1)
        total_count = np.sum(num_arms_played)
        epsilon = min(((n_bandits*self.c)/((self.d**2)*(max(total_count,1)))), 1)  #decrease the epsilon value as the time progresses
        
        # reward_estimate = success_count/(total_count+1)
        
        return np.argmax(reward_estimates) if (np.random.uniform(0,1)>epsilon) else np.random.randint(0,n_bandits)
        

class Softmax:
    def __init__(self,temp = 1):
        self.temp = temp
        
    def action(self, reward_estimates, num_arms_played, n_bandits):
        # success_count = reward_array.sum(axis = 1)
        # total_count = k_array.sum(axis = 1)
        
        # total_count = np.sum(num_arms_played)
        # reward_estimate = success_count / (total_count + 1)
        softmax_prob = np.exp(reward_estimates/ self.temp)
        softmax_prob = softmax_prob/(np.sum(softmax_prob))
        
        return np.random.choice(n_bandits, p = softmax_prob[0,:])
        
        
        
class UCB:
    
    def __init__(self):
        pass
    
    def action(self, reward_estimates, num_arms_played, n_bandits):
        
        # success_count = reward_array.sum(axis=1)
        # total_count = k_array.sum(axis = 1) #no. of times each arm is played
        # print(total_count)
        # reward_estimate = success_count/(total_count + 1e-10)
        
        #confidence term for each arm
        if (np.sum(num_arms_played) < n_bandits):
          for i in range(n_bandits): 
            if (num_arms_played[0,i] == 0):
              return i 
           

        confidence_term = np.sqrt((2*np.log(np.sum(num_arms_played)))/(num_arms_played))
    
        return np.argmax(reward_estimates + confidence_term)

class UCB1_Normal:
    
    def __init__(self, n_bandits):
        self.q = np.zeros((1,n_bandits))
    
    def action(self, reward_estimates, num_arms_played, n_bandits, **kargs):
        
        # success_count = reward_array.sum(axis=1)
        # total_count = k_array.sum(axis = 1) #no. of times each arm is played
        # print(total_count)
        # reward_estimate = success_count/(total_count + 1e-10)
        
        #confidence term for each arm
        for i in range(n_bandits): 
          if (num_arms_played[0,i] <= np.ceil(8*np.log(np.sum(num_arms_played)+1))):
            return i 
           
        # print(self.q- num_arms_played*(reward_estimates**2))
        confidence_term = np.sqrt((16*np.abs(self.q- num_arms_played*(reward_estimates**2))*np.log(np.sum(num_arms_played)-1))/(num_arms_played*(num_arms_played-1)))
    
        return np.argmax(reward_estimates + confidence_term)
    
    def update_q(self, reward, a_t, num_plays):
        self.q[0,a_t] += (reward**2 - self.q[0,a_t])/(num_plays)
        # print(self.q)


class TS :

    def __init__(self, reward_dist):
        self.reward_dist = reward_dist
       
        
    def action(self,  reward_estimates, num_arms_played, n_bandits):
        
        """
        num_arms_played   : Contains the number of times each arm is played
        reward_estimates  : contains the reward estimate for each arm
        n_bandits         : Contains the indexes of the arms
        """
        
        if self.reward_dist == 'Bernoulli':

          # Increase the success count based on the reward obtained
          success_count = num_arms_played[0,:] * reward_estimates[0,:]
          failure_count = num_arms_played[0,:] - success_count
          
          samples_list = [np.random.beta(success_count[bandit_id]+1, failure_count[bandit_id]+1) for bandit_id in range(n_bandits)]

        elif self.reward_dist == 'Gaussian':  
            
          #here the update rule we have assumed is for bandits with variance = 1.
          #Need to update the rule for the case with different variance

          samples_list = [np.random.normal(loc = reward_estimates[0,bandit_id], scale = 1/(num_arms_played[0,bandit_id]+1)) for bandit_id in range(n_bandits)]
        
        # select the arm without any tie-breaking
        a_t = np.argmax(samples_list) 
        
        
        return a_t
    
    
class TS_Gaussian :

    def __init__(self, n_bandits):
        # self.reward_dist = reward_dist
        
        #prior values of parameters for gaussian distribution to estimate the mean
        self.prior_mean = np.zeros((1,n_bandits))
        self.nu = np.ones((1,n_bandits))
        
        #prior values of parameters for inverse gamma distribution to estimate the variance
        self.init_alpha = 1*np.ones((1,n_bandits))
        self.init_beta = 1*np.ones((1,n_bandits))
        
        self.alpha = self.init_alpha
        self.beta = self.init_beta
        
        self.s = np.zeros((1,n_bandits))
    
    def action(self,  reward_estimates, num_arms_played, n_bandits):
        
        """
        num_arms_played : Contains the number of times each arm is played
        reward_estimates = contains the reward estimate for each arm
        n_banits : Contains the indexes of the arms
        """
        
        var_est = [stats.invgamma.rvs(a = self.alpha[0,bandit_id], scale = self.beta[0, bandit_id]) for bandit_id in range(n_bandits)]
        
        mu_var = var_est/(num_arms_played + self.nu)
        rho = (self.nu*self.prior_mean + num_arms_played*reward_estimates)/(self.nu + num_arms_played)
        
        samples_list = [np.random.normal(loc = rho[0,bandit_id], scale = mu_var[0,bandit_id]) for bandit_id in range(n_bandits)]

        a_t = np.argmax(samples_list) # select the arm without any tie-breaking
        
        return a_t
    
    def update(self, reward_estimate, reward, a_t, num_plays):
        
        self.s[0,a_t]  += reward**2 + (num_plays) * ((reward_estimate +  (( reward_estimate - reward)/(num_plays)))**2) - (num_plays+1) * (reward_estimate**2)
        self.beta[0,a_t] = self.init_beta[0,a_t] + 0.5*self.s[0,a_t] + (num_plays*self.nu[0,a_t]*((reward_estimate - self.prior_mean[0,a_t])**2))/(2*(num_plays + self.nu[0,a_t]))
        self.alpha[0, a_t] = 0.5*num_plays + self.init_alpha[0,a_t]
        
        # print(self.s[0,a_t], self.beta[0,a_t], self.alpha[0,a_t])
         # print(self.q)
        
class Reinforce:

  def __init__(self, n_bandits, learning_rate = 0.1):
    self.w = np.random.random(size = (1,n_bandits))
    self.lr = learning_rate

    self.probs = np.ones(shape = (1,n_bandits))/(n_bandits) # equal probability for all the arms in the beginning

  def action(self, reward_estimtes, num_arms_played, n_bandits):

     """
        num_arms_played : Contains the number of times each arm is played
        reward_estimates = contains the reward estimate for each arm
        n_bandits : Contains the indexes of the arms
    """
     return np.random.choice(n_bandits, p = self.probs[0,:])

  def update_weights(self, a_t, reward_t, avg_reward):

    #update the weight parameters using the stochastic gradient ascent update rule
    self.w += - self.lr*(reward_t)*self.probs
    self.w[0,a_t] +=  self.lr*(reward_t) 
    
    #update the probabilities after updating the weights

    self.probs = (np.exp(self.w)/(np.sum(np.exp(self.w))))


    
class Reinforce_Baseline:

  def __init__(self, n_bandits, learning_rate = 0.1):
    """
    Input:
      n_bandits : No. of bandits 

    Task Performed:
      Initialize the parameters randomly
      Initialize the learning rate
      Initialize the probability of picking an arm to be uniformly random

    """
    self.w = np.random.random(size = (1,n_bandits))
    self.lr = learning_rate

    self.probs = np.ones(shape = (1,n_bandits))/(n_bandits) # equal probability for all the arms in the beginning

  def action(self, reward_estimtes, num_arms_played, n_bandits):

     """
        num_arms_played : Contains the number of times each arm is played
        reward_estimates = contains the reward estimate for each arm
        n_bandits : Contains the indexes of the arms

        Note : Here we don't need the information of reward estimates and no. of times each arm is played
    """
     return np.random.choice(n_bandits, p = self.probs[0,:])

  def update_weights(self, a_t, reward_t, avg_reward):
    """
    Update the weight parameters using the update rule
    Update the probabilities of picking the arm after updating the weights
    """
    
    self.w += - self.lr*(reward_t - avg_reward)*self.probs
    self.w[0,a_t] +=  self.lr*(reward_t - avg_reward) 
    
    self.probs = (np.exp(self.w)/(np.sum(np.exp(self.w))))

# %%

# Different number of arms simulated
K_list            = [2,5,10]
# bandits_mean_list = [0.2, 0.3, 0.5, 0.25, 0.45, 0.6, 0.62, 0.54, 0.1, 0.58]

T        = 20000  # No. of time steps we need to pull the arm
num_runs = 20     # Number of times to run the simulation
num_algorithms = 7

# Parameters
epsilon = 0.1
# For softmax we select the temperature parameter to be 0.05 that gave us the lowest regret.
softmax_temp = 0.05

algo_labels = [f"Epsilon Greedy with epsilon = {epsilon}", "Variable-Epsilon Greedy", f"Softmax with temp = {softmax_temp}", "Upper Confidence Bound", "Thompson Sampling", "Reinforce Algorithm", "Reinforce With Baseline"]

for K in K_list:
    
  
  mab_list = [] #list of instantiation of MAB 
  for _ in range(num_runs):
        bandits_mean = np.random.uniform(low = 0, high = 1, size = K) #bandits_mean_list[0:K]
        opt_arm      = np.argmax(bandits_mean)
        # Instantiate the MAB object with the bandit means ans standard deviation
        mab = MAB('Bernoulli', bandits_mean) 
        mab_list.append(mab)


  fig1, ax1 = plt.subplots(1,1, figsize=(20, 10))
  fig2, axs = plt.subplots(1,2, figsize=(20, 10))
  print("====================================================================================================================================================================================")
  print("====================================================================================================================================================================================\n")
  print(f"\n******************************** NUMBER OF ARMS : {K} ********************************\n")
  print(f"Each algorithm was played {num_runs} times and the episode length was {T}\n")
  
  for algo_number in range(num_algorithms):
      
      # Stores the cummulative regret averaged over 'num_runs'
      cummulative_avg_regret  = np.zeros((1, T+1))  
      
      # Stores the regret per turn averaged over 'num_runs'
      avg_regret_per_turn     = np.zeros((1,T)) 

      avg_percent_optimal_arm = np.zeros((1,T+1))

      running_avg_regret_per_turn = np.zeros((1,T+1))

      for run in range(num_runs): 
            
          mab = mab_list[run]
          opt_arm = np.argmax(mab.bandits_mean)
            
          # The algorithms are instantiated for each run
          algorithms = [Epsilon_Greedy(epsilon), Variable_Epsilon_Greedy(1), Softmax(temp = softmax_temp), UCB(), TS('Bernoulli'), Reinforce(K), Reinforce_Baseline(K)]

          # pick the bandit algorithm that you wish to run
          algorithm  = algorithms[algo_number]  
      
          # To store the no. of times an arm gets pulled
          no_arms_pulled  = np.zeros((1,K))

          # To store the reward estimate of each arm 
          reward_estimate = np.zeros((1,K)) 
          
          # To store the reward obtained at each time step
          rewards_obtained = np.zeros((1, T)) 
          
          # To store the regret obtained at each time step
          regrets_achieved   = np.zeros((1,T)) 

          # To store the cummulative regret obtained at each time step
          cummulative_regret = np.zeros((1,T+1)) 

          avg_regret_over_T  = np.zeros((1,T+1)) 

          # Percentage of optimal arm played at each time step 
          percent_optimal_arm = np.zeros((1,T+1)) 
          
          avg_reward = np.zeros((1,T+1))

          for t in range(T): # At each time step pick an arm and obtain it's reward
              
              a_t  = algorithm.action(reward_estimate, no_arms_pulled, K)
              no_arms_pulled[0,a_t] += 1

              percent_optimal_arm[0,t+1] = percent_optimal_arm[0,t] + ((a_t == opt_arm) - percent_optimal_arm[0,t])/(t+1)

              reward, regret          = mab.pull_arm(a_t)
              reward_estimate[0,a_t] += ((reward - reward_estimate[0,a_t])/no_arms_pulled[0,a_t]) 
              
              avg_reward = np.sum(reward_estimate * no_arms_pulled)/(t+1)

              if (algo_labels[algo_number] == "Reinforce Algorithm" or algo_labels[algo_number] == "Reinforce With Baseline"):
                algorithm.update_weights(a_t, reward, avg_reward)
              
              rewards_obtained[0,t] = reward
              
              regrets_achieved[0,t] = regret

              cummulative_regret[0,t+1] = regret + cummulative_regret[0,t]

              avg_regret_over_T[0,t+1] = avg_regret_over_T[0,t] + ((regret- avg_regret_over_T[0,t])/(t+1)) 

          cummulative_avg_regret = cummulative_avg_regret + ((cummulative_regret - cummulative_avg_regret)/(run+1))   
          avg_regret_per_turn = avg_regret_per_turn + ((regrets_achieved - avg_regret_per_turn)/(run+1))
      
          running_avg_regret_per_turn += (avg_regret_over_T - running_avg_regret_per_turn)/(run+1)
          avg_percent_optimal_arm += (percent_optimal_arm - avg_percent_optimal_arm)/(run+1)
          
      ax1.plot(cummulative_avg_regret[0,1:], label = f'{algo_labels[algo_number]}')  
      axs[0].plot(running_avg_regret_per_turn[0,1:], label = f'{algo_labels[algo_number]}')
      axs[1].plot(avg_percent_optimal_arm[0,1:], label = f'{algo_labels[algo_number]}')

      print(f"The total regret accumulated over time for {algo_labels[algo_number]} is {cummulative_avg_regret[0,-1]:.3f}\n")

  ax1.legend()
  # ax1.set_title(f"The number of bandits considered is {K} with success probabilities {bandits_mean:.3f}")
  
  axs[0].legend()
  axs[1].legend()
  ax1.set(xlabel = "Time steps", ylabel = "Cumulative regret over time steps")
  axs[0].set(xlabel = "Time steps", ylabel = "Average regret per turn")
  axs[1].set(xlabel = "Time steps", ylabel = "Average Percentage of optimal arm plays", ylim = [0,1])
  plt.suptitle(f"The number of bandits considered is {K} with Bernoulli reward distribution", fontsize = '16')
  plt.show()

# %%

# Different number of arms simulated
K_list            = [2, 5, 10]
stdv_list         = [0.5, 1]
# bandits_mean_list = [0.2, 0.3, 0.5, 0.25, 0.45, 0.6, 0.62, 0.54, 0.1, 0.58]

T = 30000  # No. of time steps we need to pull the arm
num_runs = 50    # Number of times to run the simulation
num_algorithms = 7

# Parameters
epsilon      = 0.1
# For softmax we select the temperature parameter to be 0.05 that gave us the lowest regret.
softmax_temp = 0.05

algo_labels = [f"Epsilon Greedy with epsilon = {epsilon}", "Variable-Epsilon Greedy", f"Softmax with temp = {softmax_temp}", "Upper Confidence Bound", "Thompson Sampling", "Reinforce Algorithm", "Reinforce With Baseline"]    
# algo_labels = ["TS_Gaussian"]
# %%

for stdv in stdv_list:
  for K in K_list:

    # bandits_mean = bandits_mean_list[0:K]
    # opt_arm      = np.argmax(bandits_mean)

    # Instantiate the MAB object with the bandit means ans standard deviation
    # mab = MAB('Gaussian',bandits_mean,stdv) 

    mab_list = [] #list of instantiation of MAB 
    for _ in range(num_runs):
        bandits_mean = np.random.uniform(low = 0, high = 1, size = K) #bandits_mean_list[0:K]
       
        # Instantiate the MAB object with the bandit means ans standard deviation
        mab = MAB('Gaussian', bandits_mean,stdv) 
        mab_list.append(mab)


    # The algorithms are instantiated for each run
    # algorithms = [Epsilon_Greedy(epsilon), Variable_Epsilon_Greedy(1), Softmax(temp = softmax_temp), UCB(), TS_Gaussian(K), Reinforce(K), Reinforce_Baseline(K)]
    # algorithms = [TS_Gaussian(K)]
    
    fig2, axs = plt.subplots(1,2, figsize=(20, 10))
    fig1, ax1 = plt.subplots(1,1, figsize = (20,10))
    print("====================================================================================================================================================================================")
    print("====================================================================================================================================================================================\n")
    print(f"\n******************************** Standard Deviation of each arm : {stdv} ********************************\n")
    print(f"\n******************************** NUMBER OF ARMS : {K} ********************************\n")
    print(f"Each algorithm was played {num_runs} times and the episode length was {T}\n")
    

    for algo_number in range(num_algorithms):
        
        # Stores the cummulative regret averaged over 'num_runs'
        cummulative_avg_regret  = np.zeros((1, T+1))  
        
        # Stores the regret per turn averaged over 'num_runs'
        avg_regret_per_turn     = np.zeros((1,T)) 

        avg_percent_optimal_arm = np.zeros((1,T+1))

        running_avg_regret_per_turn = np.zeros((1,T+1))

        for run in range(num_runs):
            
            mab = mab_list[run]
            opt_arm = np.argmax(mab.bandits_mean)
            
            #we reinstantiate the algorithms for each run
            algorithms = [Epsilon_Greedy(epsilon), Variable_Epsilon_Greedy(1), Softmax(temp = softmax_temp), UCB(), TS_Gaussian(K), Reinforce(K), Reinforce_Baseline(K)]
    
            
            # pick the bandit algorithm that you wish to run
            algorithm  = algorithms[algo_number]  
        
            # To store the no. of times an arm gets pulled
            no_arms_pulled  = np.zeros((1,K))

            # To store the reward estimate of each arm 
            reward_estimate = np.zeros((1,K)) 
            
            # To store the reward obtained at each time step
            rewards_obtained = np.zeros((1, T)) 
            
            # To store the regret obtained at each time step
            regrets_achieved   = np.zeros((1,T)) 

            # To store the cummulative regret obtained at each time step
            cummulative_regret = np.zeros((1,T+1)) 

            avg_regret_over_T  = np.zeros((1,T+1)) 

            # Percentage of optimal arm played at each time step 
            percent_optimal_arm = np.zeros((1,T+1)) 
            
            avg_reward = np.zeros((1,T+1))

            for t in range(T): # At each time step pick an arm and obtain it's reward
                

                a_t  = algorithm.action(reward_estimate, no_arms_pulled, K)
                no_arms_pulled[0,a_t] += 1

                percent_optimal_arm[0,t+1] = percent_optimal_arm[0,t] + ((a_t == opt_arm) - percent_optimal_arm[0,t])/(t+1)

                reward, regret          = mab.pull_arm(a_t)
                reward_estimate[0,a_t] += ((reward - reward_estimate[0,a_t])/no_arms_pulled[0,a_t]) 
                
                avg_reward = np.sum(reward_estimate * no_arms_pulled)/(t+1)

                if (algo_labels[algo_number] == "Thompson Sampling" ):
                  algorithm.update(reward_estimate[0,a_t], reward, a_t, no_arms_pulled[0,a_t])


                if (algo_labels[algo_number] == "Reinforce Algorithm" or algo_labels[algo_number] == "Reinforce With Baseline"):
                  algorithm.update_weights(a_t, reward, avg_reward)
                
                rewards_obtained[0,t] = reward
                
                regrets_achieved[0,t] = regret

                cummulative_regret[0,t+1] = regret + cummulative_regret[0,t]

                avg_regret_over_T[0,t+1] = avg_regret_over_T[0,t] + ((regret- avg_regret_over_T[0,t])/(t+1)) 

            cummulative_avg_regret = cummulative_avg_regret + ((cummulative_regret - cummulative_avg_regret)/(run+1))   
            avg_regret_per_turn = avg_regret_per_turn + ((regrets_achieved - avg_regret_per_turn)/(run+1))
        
            running_avg_regret_per_turn += (avg_regret_over_T - running_avg_regret_per_turn)/(run+1)
            avg_percent_optimal_arm += (percent_optimal_arm - avg_percent_optimal_arm)/(run+1)

        print(f"The total regret accumulated over time for {algo_labels[algo_number]} is {cummulative_avg_regret[0,-1]:.3f}\n")

        # if algo_labels[algo_number] == algo_labels[0] or algo_labels[algo_number] == algo_labels[2] :
        #     continue
        ax1.plot(cummulative_avg_regret[0,1:], label = f'{algo_labels[algo_number]}')  
        axs[0].plot(running_avg_regret_per_turn[0,1:], label = f'{algo_labels[algo_number]}')
        axs[1].plot(avg_percent_optimal_arm[0,1:], label = f'{algo_labels[algo_number]}')

    ax1.legend()
    # ax1.set_title(f"The number of bandits (with rewards from Gaussian distribution) considered is {K} with average rewards {bandits_mean:.3f} with variance = {stdv**2}")
    axs[0].legend()
    axs[1].legend()
    ax1.set(xlabel = "Time steps", ylabel = "Cumulative regret over time steps")
    axs[0].set(xlabel = "Time steps", ylabel = "Average regret per turn")
    axs[1].set(xlabel = "Time steps", ylabel = "Average Percentage of optimal arm plays", ylim = [0,1])
    plt.suptitle(f"The number of bandits (with rewards from Gaussian distribution) considered is {K} with standard deviation = {stdv}", fontsize = '16')
    plt.show()
    
    
