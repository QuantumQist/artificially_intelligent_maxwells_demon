
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:44:49 2023

@author: bbhandari, rczupryn
"""


import sys
import os
sys.path.append(os.path.join('..','src'))
import sac_tri
import sac_tri_envs_dis
import numpy as np



a_val = 1.0
k_val = 0.9

env_params = {
    "g0": 0.8,                        #\Gamma of the bath
    "b0": 1,                        #inverse temperature \beta of the bath
    "min_u": 0.,                    #minimum value of action u
    "max_u": 1.0 ,
    "kappa": k_val,                    #maximum value of action u
    "e0": 0.5,                         #E_0
    "dt": 1.,                       #timestep \Delta t
    "a": a_val,                        # characteristic time
    "gamma": 0.995,
    "pow_coeff": 35.,
    "diss_coeff": 4.,         #the reward is multiplied by this factor
} 
training_hyperparams = {
    "BATCH_SIZE": 128,              #batch size
    "LR": 0.0008,                    #learning rate   
    "ALPHA_LR": 0.001,              # MAKE IT 0.001
    "H_D_START": np.log(3.),        #the exploration coeff
    "H_D_END": 0.01,                #the exploration coeff
    "H_D_DECAY": 100000,            #the exploration coeff, in "units" of steps   
    "H_C_START": 0.8,              #the exploration coeff
    "H_C_END": -3.,                #3.5 INSTEAD OF -7 SINCE HERE 1 ACTION, WHILE -7 2 ACTIONS
    "H_C_DECAY": 100000,           #the exploration coeff, in "units" of steps     
    "REPLAY_MEMORY_SIZE": 80000,  #IT WAS 2000 MAKE IT 100000
    "POLYAK": 0.995,                #polyak coefficient
    "LOG_STEPS": 1000,              #save logs and display training every number of steps
    "GAMMA": 0.998,                 #RL discount factor
    "HIDDEN_SIZES": (128,128),      #size of hidden layers 
    "SAVE_STATE_STEPS": 50000,      #saves complete state of trainig every number of steps
    "INITIAL_RANDOM_STEPS": 5000,   #number of initial uniformly random steps
    "UPDATE_AFTER": 1000,           #start minimizing loss function after initial steps
    "UPDATE_EVERY": 50,             #performs this many updates every this many steps
    "USE_CUDA": False,               #use cuda for computation
    "MIN_COV_EIGEN": 1.e-8
}
log_info = {
    "log_running_reward": True,     #log running reward 
    "log_running_loss": True,       #log running loss
    "log_actions": True,            #log chosen actions
    "extra_str": f"_testrun" #extra string to append to training folder
}

train = sac_tri.SacTrain()
train.initialize_new_train(sac_tri_envs_dis.TwoLevelDemonDisPowDiss, env_params, training_hyperparams, log_info)

train.train(250000)

##Evaluate deterministic policy and collect sigma logs

##_____________________________
##Evaluate deterministic policy

log_dir = "/Users/robertc/Desktop/artificial_demon-main/data/2024_02_28-17_10_21_a=1.0_k=0.6"


# # # # # # __________________________________

loaded_train = sac_tri.SacTrain()
loaded_train.load_train(log_dir, no_train=True)
#evaluate the deterministic policy
loaded_train.evaluate_current_policy(deterministic=True, steps=10000, gamma=0.9999,actions_to_plot=80,
                                      save_policy_to_file_name="det_policy.txt",actions_ylim=[-0.05,1.05])


