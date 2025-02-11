#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:49:26 2025

@author: robertc

This file can be executed to train RL agent for the discrete measurement
cases.

The number of episodes and the dictionaries containing training parameters 
can be modified to match the parameters in the paper.

The results will be saved in the `data` folder in the main directory.
"""

### Import dependencies
import sys
import os
sys.path.append(os.path.join('..','src'))
import sac_tri
import sac_tri_envs_dis
import numpy as np

### Set up number of episodes
EPISODES = 250000

### Set up environmnet parameters, i.e., parameters of the qubit+reservoir system
env_params = {
    "g0": 0.8,                       #\Gamma of the bath
    "b0": 1,                         #inverse temperature \beta of the bath
    "min_u": 0.,                     #minimum value of action u
    "max_u": 1.0 ,                   #maximum value of action u
    "kappa": 0.99,                   #measurement strength, must be >0.5 and <1. to avoid errors
    "e0": 0.5,                       #E_0 - qubit energy gap
    "gamma": 0.995,
    "dt": 1.,                        #timestep \Delta t
    "a": 1.0,                        # power-dissipation trade-off
    "pow_coeff": 35.,                #the reward is multiplied by this factor
    "diss_coeff": 4.,                #the penalty is multiplied by this factor
} 

### Set up training parameters
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
    "extra_str": "_custom_run_2" #extra string to append to training folder
}


### Perform training
train = sac_tri.SacTrain()
train.initialize_new_train(
    sac_tri_envs_dis.TwoLevelDemonDisPowDiss, # Class defining RL Environment
    env_params, training_hyperparams, 
    log_info)

train.train(250000)

