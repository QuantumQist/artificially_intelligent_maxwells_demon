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


"""
This module contains gym.Env environments that can be trained using sac_tri.SacTrain(). We implemented a single bath
diagonal qubit fridge.
These environments, besides being proper gym.Env, MUST satisfy these additional requirements:
    1) __init__ must accept a single dict with all the parameters necessary to define the environment.
    2) implement set_current_state(state). Functions that takes a state as input, and sets the environment to that state

If the environment is multiobjective, the reward will be the convex combination that is intended to be 
optimized. However, the individual objectives must be returned in the info dictionry of step() (see gym.Env), as a numpy
array with key "multi_obj".

This module contains the following classes:
    1. `TwoLevelDemonDisPowDiss` - cooling cycle powered by sigmaX measurement
    2. `TwoLevelDemonDisPowDiss2` - cooling cycle powered by sigmaZ measurement
    3. `TwoLevelDemonDisPowDissTheta` - cooling cycle powered by measurement
        with an arbitrary angle
"""

#1
class TwoLevelDemonDisPowDiss(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with discrete measurement. We study power-dissipation trade-off with the feedback
    angle being the continuous action. Demon is allowed to perform sigmaX measurement
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of bath 0
        "b0" (float): inverse temperature \beta of bath 0
        "kappa" (float): measurement strength
        "a"  (float): power dissipation trade off hyperparameter
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): qubit gap
        "dt" (float): timestep \Delta t
        "power_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonDisPowDiss, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.kappa = env_params["kappa"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
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

        #In this step, we perform thermalization
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
            

        #In this step, we perform measurement
        elif d_act == 1:
            kap = self.kappa
            mpl = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden + 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigmax
            mmi = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden - 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigmax
            prob_mpl = np.abs(np.trace(mpl.dot(self.rho).dot(mpl))) #Probability of "plus" result
            test_rand = random.random()
            if prob_mpl < test_rand:
                meas = mpl
            else:
                meas = mmi
            
            if meas is mpl:
                prob_for = np.trace(mpl.dot(self.rho).dot(mpl))
            else:
                prob_for = np.trace(mmi.dot(self.rho).dot(mmi))
                
            self.rho = (meas.dot(self.rho).dot(meas)) / prob_for
            
            
            diss += np.real((self.diss_coeff)*self.calculate_measurement_penalty(prob_for))
        
        #In this step, we perform feedback
        else:
            #Scale u_act 
            u_act_new = u_act * 2 * np.pi
            
            unit = np.array([[np.cos(u_act_new),-np.sin(u_act_new)],[np.sin(u_act_new),np.cos(u_act_new)]])
            unit_dag = np.array([[np.cos(u_act_new),np.sin(u_act_new)],[-np.sin(u_act_new),np.cos(u_act_new)]])
            self.rho = unit.dot(self.rho).dot(unit_dag)
            
        
        #complete state evolution
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = np.real(u_act)

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
        

    
#2
class TwoLevelDemonDisPowDiss2(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with discrete measurement. We study power-dissipation trade-off with the feedback
    angle being the continuous action.  Demon is allowed to perform sigmaZ measurement
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of bath 0
        "b0" (float): inverse temperature \beta of bath 0
        "kappa" (float): measurement strength
        "a"  (float): power dissipation trade off hyperparameter
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): E_0
        "dt" (float): timestep \Delta t
        "power_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonDisPowDiss2, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.kappa = env_params["kappa"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
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
        # file = open("logs_k099.txt", "a")  # append mode
        # file.write(str(d_act)+"\t"+str(self.state.rhofx)+"\t"+str(self.state.rhofz)+"\n")
        # file.close()
        # #________________________________

        #In this step, we perform thermalization
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
            

        #In this step, we perform measurement
        elif d_act == 1:
            kap = self.kappa
            mpl = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden + 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigmaz
            mmi = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden - 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigmaz
            prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl)) #Probability of "plus" result
            test_rand = random.random()
            if prob_mpl < test_rand:
                meas = mpl
            else:
                meas = mmi
            if meas is mpl:
                prob_for = np.trace(mpl.dot(self.rho).dot(mpl))
            else:
                prob_for = np.trace(mmi.dot(self.rho).dot(mmi))  
                
            self.rho = (meas.dot(self.rho).dot(meas))/prob_for
            
            
            diss += np.real((self.diss_coeff)*self.calculate_measurement_penalty(prob_for))
        
        #In this step, we perform feedback
        else:
            #Scale u_act to be between 0 and 2*pi
            u_act_new = 2 * u_act * np.pi 
            
            unit = np.array([[np.cos(u_act_new),-np.sin(u_act_new)],[np.sin(u_act_new),np.cos(u_act_new)]])
            unit_dag = np.array([[np.cos(u_act_new),np.sin(u_act_new)],[-np.sin(u_act_new),np.cos(u_act_new)]])
            self.rho = unit.dot(self.rho).dot(unit_dag)
            
        
        #complete state evolution
        #self.state.dt = self.dt
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = np.real(u_act)

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
        
    
    
#1.3
class TwoLevelDemonDisPowDissTheta(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with discrete measurement. 
    We study power-dissipation trade-off with the measurement angle being the 
    continuous action
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of bath 0
        "b0" (float): inverse temperature \beta of bath 0
        "kappa" (float): measurement strength
        "a"  (float): power dissipation trade off hyperparameter
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "e0" (float): E_0
        "dt" (float): timestep \Delta t
        "power_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonDisPowDissTheta, self).__init__()
        self.load_env_params(env_params)

    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.kappa = env_params["kappa"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
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
        
        # # Choose range of measurement angles
        theta = u_act * 3.6 - 0.2
    

        #initialize null multi objectives
        poww = 0.
        diss = 0.

        # #________________________________
        # #Uncomment this section to get sigmaLogs
        # file = open("logs_a100.txt", "a")  # append mode
        # file.write(str(d_act) + "\t"+ str(theta) +"\t"+str(self.state.rhofx)+"\t"+str(self.state.rhofz)+"\n")
        # file.close()
        # #________________________________

        #In this step, we perform thermalization
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
            

        #In this step, we perform measurement
        elif d_act == 1:
            #Assume the continuous action can take values from 0 to 1
            
            #Define measurement operator
            sigma = np.array( [ [np.cos(theta), np.sin(theta)],
                    [ np.sin(theta), -np.cos(theta)]] )
            
            kap = self.kappa
            mpl = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden + 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigma
            mmi = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden - 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigma
            prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl)) #Probability of "plus" result
            test_rand = random.random()
            if prob_mpl < test_rand:
                meas = mpl
            else:
                meas = mmi
            if meas is mpl:
                prob_for = np.trace(mpl.dot(self.rho).dot(mpl))
            else:
                prob_for = np.trace(mmi.dot(self.rho).dot(mmi))  
                
            self.rho = (meas.dot(self.rho).dot(meas))/(np.trace(meas.dot(self.rho).dot(meas)))
            
            # Calculate the penalty
            diss += np.real((self.diss_coeff)*self.calculate_measurement_penalty(prob_for))
        
        #In this step, we perform feedback - rotation to negative z axis
        else:
            xx = np.trace(sigmax.dot(self.rho))
            zz = np.trace(sigmaz.dot(self.rho))
            z3 = np.sqrt(xx**2+zz**2)
            self.rho = 0.5*(iden - z3*sigmaz)
            
        
        #complete state evolution
        #self.state.dt = self.dt
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = np.real(u_act)

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
        
# 2. Environment with variable qubit gap
class TwoLevelDemonDisPowDisse0(gym.Env):
    """
    Gym.Env representing a qubit based Maxwell demon with discrete measurement. We study power-dissipation
    trade-off with qubit gap being the continuous action
    Args:
        env_params is a dictionary that must contain the following: 
        "g0" (float): \Gamma of bath 0
        "b0" (float): inverse temperature \beta of bath 0
        "gm" (float): measurement strength
        "a"  (float): power dissipation trade off hyperparameter
        "min_u" (float): minimum value of action u
        "max_u" (float): maximum value of action u
        "gamma"  (float): hyperparameter to calculate the average time, since meas time and
          thermal time can be different
        "dt" (float): time allocated for measurement and thermalization
        "power_coeff" (float): the power is multiplied by this factor
        "diss_coeff" (float): the dissipation is multiplied by this factor
    """

    @dataclasses.dataclass
    class State:
        """ data object representing the state of the environment """
        rhofx: float = 0. #probability of being in the x state
        rhofz: float = 0. #probability of being the z state
        u: float = 0. # last chosen action
    
    def __init__(self, env_params):
        super(TwoLevelDemonDisPowDisse0, self).__init__()
        self.load_env_params(env_params)
    
    def load_env_params(self, env_params):
        """
        Initializes the environment
        
        Args:
            env_params: environment parameters as passed to __init__()
        """
        #load the environment parameters
        self.g0 = env_params["g0"]
        self.gm = env_params["gm"]
        self.b0 = env_params["b0"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.dt = env_params["dt"]
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
        
        #load action
        d_act = action[0]
        u_act = action[1][0]
        
        
        ham = np.array([[u_act,0],[0,0]])
        gam_eg = self.g0 * (1 + (-1+np.exp(self.b0 * u_act))**(-1))
        gam_ge = self.g0 * (-1+np.exp(self.b0 * u_act))**(-1)
        gsig = gam_eg + gam_ge
        
        #initialize null multi objectives
        poww = 0.
        diss = 0.
    
        #In this step, we perform thermalization
        if d_act == 0:
            peg = self.rho[0,1]
            peg_dt = peg * np.exp(-(gsig/2+1j*u_act)*self.dt)
            pge_dt = np.conjugate(peg_dt)
            pe = np.exp(-gsig * self.dt)*((-1+np.exp(gsig * self.dt))*gam_ge + gsig * self.rho[0,0])/gsig
            pg = 1 - pe
            rho_th = np.array([[pe,peg_dt],[pge_dt,pg]])
            
            poww +=  (self.pow_coeff/self.dt)*(np.real(np.trace(ham.dot(rho_th))\
                                                                     -np.trace(ham.dot(self.rho))))
            
            self.rho=rho_th
            
    
        #In this step, we perform measurement
        elif d_act == 1:
            kap = 0.5-np.sqrt(2.0*self.gm*self.dt)
            mpl = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden + 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigmaz
            mmi = 0.5*(np.sqrt(kap)+np.sqrt(1-kap))*iden - 0.5*(np.sqrt(kap)-np.sqrt(1-kap))*sigmaz
            prob_mpl = np.trace(mpl.dot(self.rho).dot(mpl)) #Probability of "plus" result
            test_rand = random.random()
            if prob_mpl < test_rand:
                meas = mpl
            else:
                meas = mmi
            if meas is mpl:
                prob_for = np.trace(mpl.dot(self.rho).dot(mpl))
            else:
                prob_for = np.trace(mmi.dot(self.rho).dot(mmi))  
            self.rho = (meas.dot(self.rho).dot(meas))/(np.trace(meas.dot(self.rho).dot(meas)))
            
            
            diss += np.real((self.diss_coeff)*self.calculate_measurement_penalty(prob_for))
        
        #In this step, we perform feedback
        else:
            xx = np.trace(sigmax.dot(self.rho))
            zz = np.trace(sigmaz.dot(self.rho))
            z3 = np.sqrt(xx**2+zz**2)
            self.rho = 0.5*(iden - z3*sigmaz)
            
        
        #complete state evolution
        #self.state.dt = self.dt
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = np.real(u_act)
    
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
        ham = np.array([[random_u,0],[0,0]])
        exp_ham = sci.expm(-self.b0*ham)
        Zz = np.trace(exp_ham)
        self.rho = exp_ham/Zz
        sigmax = np.array([[0,1],[1,0]])
        sigmaz = np.array([[1,0],[0,-1]])
    
        #set the 3 state variables
        self.state.rhofx = np.real(np.trace(self.rho.dot(sigmax)))
        self.state.rhofz = np.real(np.trace(self.rho.dot(sigmaz)))
        self.state.u = random_u
