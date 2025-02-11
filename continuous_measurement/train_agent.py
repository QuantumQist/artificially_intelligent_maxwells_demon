# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:15:49 2023

@author: bbhandari, rczupryn
"""


import sys
import os
sys.path.append(os.path.join('..','src'))
import sac_tri
import sac_tri_envs_con
import numpy as np
import csv


EPISODES = 300000 # Number of episodes used to train the agent
### RL environment parameters
env_params = {
    "g0": 1.,                        #\Gamma of the bath
    "b0": 1.,                        #inverse temperature \beta of the bath
    "min_u": -0.8,                   #minimum value of action u
    "max_u": 0.8,                    #maximum value of action u
    "e0": 5.,                        # qubit energy gap                   
    "dt": 0.05,                      #timestep \Delta t
    "tau": 0.001,                  # characteristic measurement time
    "a": 1.,                         # pow-diss trade-off
    "pow_coeff": 35.,                # the reward is multiplied by this factor
    "diss_coeff": 4.,                # the d issipation is multiplied by this factor
}  
### Training parameters
training_hyperparams = {
    "BATCH_SIZE": 256,              #batch size
    "LR": 0.0003,                   #learning rate
    "ALPHA_LR": 0.001,              
    "H_D_START": np.log(3.),        #the exploration coeff
    "H_D_END": 0.01,                #the exploration coeff
    "H_D_DECAY": 80000,            #the exploration coeff, in "units" of steps    MAKE THEM 2 OR 3 TIMES LARGER
    "H_C_START": 0.8,               #the exploration coeff
    "H_C_END": -3.,                 #3.5 INSTEAD OF -7 SINCE HERE 1 ACTION, WHILE -7 2 ACTIONS
    "H_C_DECAY": 80000,            #the exploration coeff, in "units" of steps     MAKE THEM 2 OR 3 TIMES LARGER
    "REPLAY_MEMORY_SIZE": 80000,   
    "POLYAK": 0.995,                #polyak coefficient
    "LOG_STEPS": 2000,              #save logs and display training every number of steps
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
    "extra_str": "_test_run" #extra string to append to training folder
}

# Instantiate and train the agent
train = sac_tri.SacTrain()
train.initialize_new_train(sac_tri_envs_con.TwoLevelBosonicFeedbackDemonPowDissContMeas, env_params, training_hyperparams, log_info)

train.train(EPISODES)



















