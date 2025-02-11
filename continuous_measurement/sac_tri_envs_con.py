# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:35:51 2023

@author: bbhandari
"""

from __future__ import print_function
import gym
import numpy as np
import dataclasses
import random
import scipy.linalg as sci
from scipy.special import xlogy
import os


"""
This module contains gym.Env environments that can be trained using sac.SacTrain. We implemented a single bath
diagonal qubit (operated e.g. as a heater), and the coherent qubit fridge.
These environments, besides being proper gym.Env, MUST satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that state

If the environment is multiobjective, the reward will be the convex combination that is intended to be 
optimized. However, the individual objectives must be returd in the info dictionry of step() (see gym.Env), as a numpy
array with key "multi_obj".
"""

#1
class TwoLevelDemonConPowDissX(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with continuous measurement of sigmaX. We study power dissipation
    trade off with no continuous action 
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of the bath
        "b0" (float): inverse temperature
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): E_0
        "dt" (float): timestep \Delta t
        "tau" (float): Measurement characteristic time
        "a" (float): power dissipation trade off hyperparameter
        "pow_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonConPowDissX, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
        self.tau = env_params["tau"]
        self.a = env_params["a"]
        self.pow_coeff = env_params["pow_coeff"]
        self.diss_coeff = env_params["diss_coeff"]
        self.state = self.State()

        #set the observation and action spaces
        self.observation_space = (gym.spaces.Box( low=np.array([0., 0., self.min_u],dtype=np.float32),
                                    high=np.array([1.,1.,self.max_u],dtype=np.float32), dtype=np.float32))
        self.action_space = (gym.spaces.Discrete(3), gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                              high=np.array([self.max_u],dtype=np.float32), dtype=np.float32))
 
        #reset the state of the environment
        self.reset_internal_state_variables()
        
    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
    
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action
        Args:
            action (type specificed by self.action_space): the action to perform on the environment
        Raises:
            Exception: action out of bound
        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the average heat flux absorbed
                from both baths during the current timestep
            end(bool): whether the episode ended (these environments never end)
            additional_info: required by gym.Env, but we don't use it
        """
   
        #check if action in bound
        if not self.action_space[0].contains(action[0]) or not self.action_space[1].contains(action[1]):
            raise Exception(f"Action {action} out of bound")
            
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
        iden = np.array([[1,0],[0,1]])
        
        
        ham = np.array([[self.e0,0],[0,0]])
        gam_eg = self.g0 * (1 + (-1+np.exp(self.b0 * self.e0))**(-1))
        gam_ge = self.g0 * (-1+np.exp(self.b0 * self.e0))**(-1)
        gsig = gam_eg + gam_ge
        
        #load action
        d_act = action[0]
        u_act = action[1][0]
    

        #initialize null multi objectives
        poww = 0.
        diss = 0.
        
        # # Uncomment this section to save logs
        # file = open("X_tau50.txt", "a")  # append mode
        # file.write(str(d_act)+"\t"+str(self.state.rhofx)+"\t"+str(self.state.rhofz)+"\n")
        # file.close()
        
        #Thermalization
        if d_act == 0:
            peg = self.rho[0,1]
            peg_dt = peg * np.exp(-(gsig/2+1j*self.e0)*self.dt)
            pge_dt = np.conjugate(peg_dt)
            pe = np.exp(-gsig * self.dt)*((-1+np.exp(gsig * self.dt))*gam_ge + gsig * self.rho[0,0])/gsig
            pg = 1 - pe
            rho_th = np.array([[pe,peg_dt],[pge_dt,pg]])
            
            poww +=  (self.pow_coeff/self.dt)*(np.real(np.trace(ham.dot(rho_th))\
                                                                     -np.trace(ham.dot(self.rho))))
            
            self.rho=rho_th
            

        #Measurment
        elif d_act == 1:
            sx_av = np.trace(sigmax.dot(self.rho))
            prb0 = (1-sx_av)/2
            prb1 = (1+sx_av)/2
            if prb0 >= 0 and prb1 >= 0:
                dis1 = np.random.normal(-1,np.sqrt(self.tau/self.dt),10000)
                dis2 = np.random.normal(1,np.sqrt(self.tau/self.dt),10000)
                elements = [dis1,dis2]
                rr = np.random.choice(elements[np.random.choice([0,1],p=[prb0,prb1])])
                rr_pow = np.linalg.matrix_power((rr*iden-sigmax),2)
                mpl = ((self.dt/(2*np.pi*self.tau))**(1/4))*sci.expm(-0.25*(self.dt/self.tau)*rr_pow)
                prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl))    
                
                self.rho = (mpl.dot(self.rho).dot(mpl))/prob_mpl

                diss += 0
 
            else:
                pass
        
        #Feedback
        else:
            xx = np.trace(sigmax.dot(self.rho))
            zz = np.trace(sigmaz.dot(self.rho))
            z3 = np.sqrt(xx**2+zz**2)
            self.rho = 0.5*(iden - z3*sigmaz)
        
        #complete state evolution
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = u_act

        #compute reward
        reward = self.a*poww - (1.-self.a)*diss

        #return        
        return self.current_state(), reward, False, {"multi_obj": 
            np.array([poww, -diss], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.state)

    def calculate_measurement_penalty(self, p):
        """
        Calculates the measurement penalty accorting to the Landauer's rule
        """
        return - p * np.log(p) - (1-p)*np.log(1-p)
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.
        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.rhofx, self.state.rhofz, self.state.u = state

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.rhofx,self.state.rhofz, self.state.u] , dtype=np.float32)
           
    def reset_internal_state_variables(self):
        """ sets initial values for the state """
        random_u =  self.action_space[1].sample()[0]
        ham = np.array([[self.e0,0],[0,0]])
        exp_ham = sci.expm(-self.b0*ham)
        Zz = np.trace(exp_ham)
        self.rho = exp_ham/Zz
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
        
        #set the 3 state variables
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = random_u
        
# 2
class TwoLevelDemonConPowDissZ(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with continuous measurement of sigmaZ. We study power dissipation
    trade off with no continuous action 
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of the bath
        "b0" (float): inverse temperature
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): E_0
        "dt" (float): timestep \Delta t
        "tau" (float): Measurement characteristic time
        "a" (float): power dissipation trade off hyperparameter
        "pow_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonConPowDissZ, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
        self.tau = env_params["tau"]
        self.a = env_params["a"]
        self.pow_coeff = env_params["pow_coeff"]
        self.diss_coeff = env_params["diss_coeff"]
        self.state = self.State()

        #set the observation and action spaces
        self.observation_space = (gym.spaces.Box( low=np.array([0., 0., self.min_u],dtype=np.float32),
                                    high=np.array([1.,1.,self.max_u],dtype=np.float32), dtype=np.float32))
        self.action_space = (gym.spaces.Discrete(3), gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                              high=np.array([self.max_u],dtype=np.float32), dtype=np.float32))
 
        #reset the state of the environment
        self.reset_internal_state_variables()
        
    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
    
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action
        Args:
            action (type specificed by self.action_space): the action to perform on the environment
        Raises:
            Exception: action out of bound
        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the average heat flux absorbed
                from both baths during the current timestep
            end(bool): whether the episode ended (these environments never end)
            additional_info: required by gym.Env, but we don't use it
        """
   
        #check if action in bound
        if not self.action_space[0].contains(action[0]) or not self.action_space[1].contains(action[1]):
            raise Exception(f"Action {action} out of bound")
            
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
        iden = np.array([[1,0],[0,1]])
        
        
        ham = np.array([[self.e0,0],[0,0]])
        gam_eg = self.g0 * (1 + (-1+np.exp(self.b0 * self.e0))**(-1))
        gam_ge = self.g0 * (-1+np.exp(self.b0 * self.e0))**(-1)
        gsig = gam_eg + gam_ge
        
        #load action
        d_act = action[0]
        u_act = action[1][0]
    

        #initialize null multi objectives
        poww = 0.
        diss = 0.
        
        # file = open("Z_g005_tau50.txt", "a")  # append mode
        # file.write(str(d_act)+"\t"+str(self.state.rhofx)+"\t"+str(self.state.rhofz)+"\n")
        # file.close()
        
        #Thermalization
        if d_act == 0:
            peg = self.rho[0,1]
            peg_dt = peg * np.exp(-(gsig/2+1j*self.e0)*self.dt)
            pge_dt = np.conjugate(peg_dt)
            pe = np.exp(-gsig * self.dt)*((-1+np.exp(gsig * self.dt))*gam_ge + gsig * self.rho[0,0])/gsig
            pg = 1 - pe
            rho_th = np.array([[pe,peg_dt],[pge_dt,pg]])
            
            poww +=  (self.pow_coeff/self.dt)*(np.real(np.trace(ham.dot(rho_th))\
                                                                     -np.trace(ham.dot(self.rho))))
            
            self.rho=rho_th
            

        #Measurment
        elif d_act == 1:
            sx_av = np.trace(sigmaz.dot(self.rho))
            prb0 = (1-sx_av)/2
            prb1 = (1+sx_av)/2
            if prb0 >= 0 and prb1 >= 0:
                dis1 = np.random.normal(-1,np.sqrt(self.tau/self.dt),10000)
                dis2 = np.random.normal(1,np.sqrt(self.tau/self.dt),10000)
                elements = [dis1,dis2]
                rr = np.random.choice(elements[np.random.choice([0,1],p=[prb0,prb1])])
                rr_pow = np.linalg.matrix_power((rr*iden-sigmaz),2)
                mpl = ((self.dt/(2*np.pi*self.tau))**(1/4))*sci.expm(-0.25*(self.dt/self.tau)*rr_pow)
                prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl))   
                
                self.rho = (mpl.dot(self.rho).dot(mpl))/prob_mpl
               
            else:
                pass
        
        #Feedback
        else:
            xx = np.trace(sigmax.dot(self.rho))
            zz = np.trace(sigmaz.dot(self.rho))
            z3 = np.sqrt(xx**2+zz**2)
            self.rho = 0.5*(iden - z3*sigmaz)
        
        #complete state evolution
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = u_act

        #compute reward
        reward = self.a*poww - (1.-self.a)*diss

        #return        
        return self.current_state(), reward, False, {"multi_obj": 
            np.array([poww, -diss], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.state)
        
    def calculate_measurement_penalty(self, p):
        """
        Calculates the measurement penalty accorting to the Landauer's rule
        """
        return - p * np.log(p) - (1-p)*np.log(1-p)
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.
        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.rhofx, self.state.rhofz, self.state.u = state

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.rhofx,self.state.rhofz, self.state.u] , dtype=np.float32)
           
    def reset_internal_state_variables(self):
        """ sets initial values for the state """
        random_u =  self.action_space[1].sample()[0]
        ham = np.array([[self.e0,0],[0,0]])
        exp_ham = sci.expm(-self.b0*ham)
        Zz = np.trace(exp_ham)
        self.rho = exp_ham/Zz
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
        
        #set the 3 state variables
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = random_u
                 
# 3
class TwoLevelDemonConPowDissTheta(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with continuous measurement of sigmaX. We study power dissipation
    trade off with no continuous action 
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of the bath
        "b0" (float): inverse temperature
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): E_0
        "dt" (float): timestep \Delta t
        "tau" (float): Measurement characteristic time
        "a" (float): power dissipation trade off hyperparameter
        "pow_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonConPowDissTheta, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
        self.tau = env_params["tau"]
        self.a = env_params["a"]
        self.pow_coeff = env_params["pow_coeff"]
        self.diss_coeff = env_params["diss_coeff"]
        self.state = self.State()

        #set the observation and action spaces
        self.observation_space = (gym.spaces.Box( low=np.array([0., 0., self.min_u],dtype=np.float32),
                                    high=np.array([1.,1.,self.max_u],dtype=np.float32), dtype=np.float32))
        self.action_space = (gym.spaces.Discrete(3), gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                              high=np.array([self.max_u],dtype=np.float32), dtype=np.float32))
 
        #reset the state of the environment
        self.reset_internal_state_variables()
        
    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
    
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action
        Args:
            action (type specificed by self.action_space): the action to perform on the environment
        Raises:
            Exception: action out of bound
        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the average heat flux absorbed
                from both baths during the current timestep
            end(bool): whether the episode ended (these environments never end)
            additional_info: required by gym.Env, but we don't use it
        """
   
        #check if action in bound
        if not self.action_space[0].contains(action[0]) or not self.action_space[1].contains(action[1]):
            raise Exception(f"Action {action} out of bound")
            
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
        iden = np.array([[1,0],[0,1]])
        
        
        ham = np.array([[self.e0,0],[0,0]])
        gam_eg = self.g0 * (1 + (-1+np.exp(self.b0 * self.e0))**(-1))
        gam_ge = self.g0 * (-1+np.exp(self.b0 * self.e0))**(-1)
        gsig = gam_eg + gam_ge
        
        #load action
        d_act = action[0]
        u_act = action[1][0]
    

        #initialize null multi objectives
        poww = 0.
        diss = 0.
        
        # #________________________________
        # #Uncomment this section to get sigmaLogs
        # file = open("R_test.txt", "a")  # append mode
        # file.write(str(d_act) + "\t"+ str(u_act * 3.6 - 0.3) +"\t"+str(self.state.rhofx)+"\t"+str(self.state.rhofz)+"\n")
        # file.close()
        # #________________________________
        
        #Thermalization
        if d_act == 0:
            peg = self.rho[0,1]
            peg_dt = peg * np.exp(-(gsig/2+1j*self.e0)*self.dt)
            pge_dt = np.conjugate(peg_dt)
            pe = np.exp(-gsig * self.dt)*((-1+np.exp(gsig * self.dt))*gam_ge + gsig * self.rho[0,0])/gsig
            pg = 1 - pe
            rho_th = np.array([[pe,peg_dt],[pge_dt,pg]])
            
            poww +=  (self.pow_coeff/self.dt)*(np.real(np.trace(ham.dot(rho_th))\
                                                                     -np.trace(ham.dot(self.rho))))
            
            self.rho=rho_th
            

        #Measurment
        elif d_act == 1:
            #Define measurement angle to be in range from -0.3 to 3.5 (1.8 ~= pi + 0.3)
            theta = u_act * 3.6 - 0.3
            
            #Define measurement operator
            sigma = np.array( [ [np.cos(theta), np.sin(theta)],
                    [ np.sin(theta), -np.cos(theta)]] )
            
            sx_av = np.trace(sigma.dot(self.rho))
            prb0 = (1-sx_av)/2
            prb1 = (1+sx_av)/2
            if prb0 >= 0 and prb1 >= 0:
                dis1 = np.random.normal(-1,np.sqrt(self.tau/self.dt),10000)
                dis2 = np.random.normal(1,np.sqrt(self.tau/self.dt),10000)
                elements = [dis1,dis2]
                rr = np.random.choice(elements[np.random.choice([0,1],p=[prb0,prb1])])
                rr_pow = np.linalg.matrix_power((rr*iden-sigma),2)
                mpl = ((self.dt/(2*np.pi*self.tau))**(1/4))*sci.expm(-0.25*(self.dt/self.tau)*rr_pow)
                prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl))    
                
                self.rho = (mpl.dot(self.rho).dot(mpl))/prob_mpl

                
            else:
                pass
        
        #Feedback
        else:
            xx = np.trace(sigmax.dot(self.rho))
            zz = np.trace(sigmaz.dot(self.rho))
            z3 = np.sqrt(xx**2+zz**2)
            self.rho = 0.5*(iden - z3*sigmaz)
        
        #complete state evolution
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = u_act

        #compute reward
        reward = self.a*poww - (1.-self.a)*diss

        #return        
        return self.current_state(), reward, False, {"multi_obj": 
            np.array([poww, -diss], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.state)
        
    def calculate_measurement_penalty(self, p):
        """
        Calculates the measurement penalty accorting to the Landauer's rule
        """
        return - p * np.log(p) - (1-p)*np.log(1-p)
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.
        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.rhofx, self.state.rhofz, self.state.u = state

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.rhofx,self.state.rhofz, self.state.u] , dtype=np.float32)
           
    def reset_internal_state_variables(self):
        """ sets initial values for the state """
        random_u =  self.action_space[1].sample()[0]
        ham = np.array([[self.e0,0],[0,0]])
        exp_ham = sci.expm(-self.b0*ham)
        Zz = np.trace(exp_ham)
        self.rho = exp_ham/Zz
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
        
        #set the 3 state variables
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = random_u

# Paolo original + cont measurement  
class TwoLevelBosonicFeedbackDemonPowDissContMeas(gym.Env):
    """
    this is like "TwoLevelBosonicDemonPowDiss", but instead of putting the system in G.S. after every measurement,
    it stochastically chooses the right state. So the agent has to learn to choose the right value of u
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        p: float = 0. #probability of being in the excited state.
        u: float = 0. #last chosen action.
    
    def __init__(self, env_params):
        super().__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g = env_params["g0"]
        self.b = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
        self.a = env_params["a"]
        self.tau = env_params["tau"]
        self.pow_coeff = env_params["pow_coeff"]
        self.diss_coeff = env_params["diss_coeff"]
        self.state = self.State()
        

        #set the observation and action spaces
        self.observation_space = (gym.spaces.Box( low=np.array([0., self.min_u],dtype=np.float32),
                                    high=np.array([1.,self.max_u],dtype=np.float32), dtype=np.float32))
        self.action_space = (gym.spaces.Discrete(3), gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                              high=np.array([self.max_u],dtype=np.float32), dtype=np.float32))
 
        #reset the state of the environment
        self.reset_internal_state_variables()
        
    def reset(self):
        """ resets the state of the environment """
        self.reset_internal_state_variables()
        return self.current_state()
    
    def step(self, action):
        """ Evolves the state for a timestep depending on the chosen action
        Args:
            action (type specificed by self.action_space): the action to perform on the environment
        Raises:
            Exception: action out of bound
        Returns:
            state(np.Array): new state after the step
            reward(float): the reward, i.e. the average heat flux absorbed
                from both baths during the current timestep
            end(bool): whether the episode ended (these environments never end)
            additional_info: required by gym.Env, but we don't use it
        """
   
   
        #check if action in bound
        if not self.action_space[0].contains(action[0]) or not self.action_space[1].contains(action[1]):
            raise Exception(f"Action {action} out of bound")
        
        #load action 
        d_act = action[0]
        u_act = action[1][0]

        #initialize null multi objectives
        pow = 0.
        diss = 0.
        
        iden = np.array([[1,0],[0,1]])
        sigmaz = np.array([[1,0],[0,-1]])            

        #thermalization
        if d_act == 0:
            #evolve the state according to the master equation
            prev_p = self.state.p
            peq = self.peq(self.de(u_act),self.b)
            self.state.p = (prev_p - peq)*np.exp(-self.g*self.dt/np.tanh(0.5*np.abs(self.de(u_act))*self.b)) + peq
            #compute power reward
            pow += self.pow_coeff* self.de(u_act)*(self.state.p - prev_p)/self.dt
        
        #if it's a measurement (d_act=0), I compute effect of the measreument
        if d_act == 1:
            self.rho = np.array([[self.state.p,0],[0,1-self.state.p]])
            
            sx_av = np.trace(sigmaz.dot(self.rho))
            prb0 = (1-sx_av)/2
            prb1 = (1+sx_av)/2
            if prb0 >= 0 and prb1 >= 0:
                dis1 = np.random.normal(-1,np.sqrt(self.tau/self.dt),10000)
                dis2 = np.random.normal(1,np.sqrt(self.tau/self.dt),10000)
                elements = [dis1,dis2]
                rr = np.random.choice(elements[np.random.choice([0,1],p=[prb0,prb1])])
                rr_pow = np.linalg.matrix_power((rr*iden-sigmaz),2)
                mpl = ((self.dt/(2*np.pi*self.tau))**(1/4))*sci.expm(-0.25*(self.dt/self.tau)*rr_pow)
                prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl))   
                
                self.rho = (mpl.dot(self.rho).dot(mpl))/prob_mpl
                
                self.state.p = self.rho[0,0]
               
            else:
                pass
        
        #complete state evolution
        self.state.u = u_act

        #compute reward
        reward = self.a*pow - (1.-self.a)*diss

        #return        
        return self.current_state(), reward, False, {"multi_obj": 
            np.array([pow, -diss], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.state)
    
    
    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.
        Args:
            state (type specificed by self.observation_space): state of the environment
        """
        self.state.p, self.state.u = state

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        return np.array([self.state.p, self.state.u] , dtype=np.float32)
           
    def reset_internal_state_variables(self):
        """ sets initial values for the state """
        #set initial population to average temperature and choose random action b
        random_u =  self.action_space[1].sample()[0]
        
        #set the 2 state variables
        self.state.p = self.peq( self.de(random_u) ,self.b)
        self.state.u = random_u

    def peq(self, eps, b):
        """
        Equilibrium probability of being in excited state at energy gap eps and inverse temperature b
        Args:
            eps (float): energy gap of the qubit
            b (float): inverse temperature
        Returns:
            peq (float): equilibrium probability of being in excited state
        """
        return 1. / (1. + np.exp(b*eps) )

    def de(self, u):
        """
        Energy gap of the qubit.
        Args:
            u (float): value of the control
        
        Returns:
            de (float): energy gap of the qubit
        """
        return self.e0 * u
    
    def entropy(self, p):
        return   -xlogy(p,p) - xlogy(1.-p,1.-p)
    
    def get_z_coordinate(self, p):
        #here I use as convention that negative rz is the ground state of the positive u hamiltonian
        return 2.*p-1.
    
    
    
    
    
    
    