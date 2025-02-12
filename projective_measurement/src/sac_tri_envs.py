from __future__ import print_function
import gym
import numpy as np
import dataclasses
import random
import types
import qutip as qt
from scipy.special import xlogy


"""
This module contains gym.Env environments corresponding to the different physical setups
"""


class TwoLevelBosonicFeedbackDemonPowDiss(gym.Env):
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
        self.g = env_params["g"]
        self.b = env_params["b"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.e0 = env_params["e0"]
        self.dt = env_params["dt"]
        self.a = env_params["a"]
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

        #if it's a measurement (d_act=0), I compute effect of the measreument
        if d_act == 0:
            #compute the entropy of the classical bit storing the outcome of the measurement
            s = self.entropy(self.state.p)
            #now I measure, and collapse the state 
            if np.random.rand() < self.state.p:
                self.state.p = 1.
            else:
                self.state.p = 0.
            #compute the objectives
            diss += self.diss_coeff * s / self.b / self.dt

        #thermalization (d_act=1) or measurement case (d_act=0)
        # (if I measure, I also perform a thermalization right after)
        if d_act == 1:
            #evolve the state according to the master equation
            prev_p = self.state.p
            peq = self.peq(self.de(u_act),self.b)
            self.state.p = (prev_p - peq)*np.exp(-self.g*self.dt/np.tanh(0.5*np.abs(self.de(u_act))*self.b)) + peq
            #compute power reward
            pow += self.pow_coeff* self.de(u_act)*(self.state.p - prev_p)/self.dt
        
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
    
    def state_to_block_sphere(self, state):
        p0 = state[0]
        #here I use as convention that negative rz is the ground state of the positive u hamiltonian
        rz = 2.*p0-1.
        return np.array([0., 0., rz])

    
class TwoQubitResonantFeedbackDemonPowDiss(gym.Env):
    """
    represents two qubits, each one with a ~ sigma_plus*sigma_minus local H, and an interaction term which is sigma_x * sigma_x or just the rotating terms
    """

    @dataclasses.dataclass
    class State:
        """
        Data object representing the state of the environment. It consists of a qutip 
        density matrix rho of two qubits, and the last chosen action u
        """
        rho = 0.
        u  = 0.
        

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
        self.e0 = env_params["e0"]
        self.g = env_params["g"]
        self.b = env_params["b"]
        self.gamma = env_params["gamma"]
        self.min_u = env_params["min_u"]
        self.max_u = env_params["max_u"]
        self.dt = env_params["dt"]
        self.a = env_params["a"]
        self.pow_coeff = env_params["pow_coeff"]
        self.diss_coeff = env_params["diss_coeff"]
        self.counter_rot = env_params["counter_rot"]
        #for backward compatibility
        if "mesolver_nsteps" in env_params:
            self.mesolver_nsteps =  env_params["mesolver_nsteps"]
        else:
            self.mesolver_nsteps = None
        
        self.state = self.State()

        #prepare the observation space (each of the 16 coefficients of the
        # density matrix is between -1 and +1)
        obs_low = np.zeros(17, dtype=np.float32) - 1. 
        obs_low[-1] = self.min_u
        obs_high = obs_low + 2.
        obs_high[-1] = self.max_u
        
        #set the observation and action spaces
        self.observation_space = gym.spaces.Box( low=obs_low, high= obs_high, dtype=np.float32)
        self.action_space = (gym.spaces.Discrete(3),
                            gym.spaces.Box(low=np.array([self.min_u],dtype=np.float32),
                                        high=np.array([self.max_u],dtype=np.float32), dtype=np.float32))
 
        #initialize the Hamiltonian
        self.init_qt_vars()

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

        #if it's a measurement (d_act=0), I compute effect of the measreument
        if d_act == 0:
            #i compute the probability of measuring the ground state
            p0 = qt.expect(self.gs_proj, self.state.rho) 
            
            #numerically this p0 can become (within numerical accuracy) slightly bigger than 1. So i have to clip it
            p0 = np.clip(p0, 0.00001, 0.99999)

            #i compute the entropy associated with the classical memory storing the outcome of the measurement
            s = self.entropy(p0)
            #now i compute the evolution of the state due to the measurement outcome
            if np.random.rand() < p0:
                # i am in the GS
                self.state.rho = self.gs_proj * self.state.rho * self.gs_proj / p0
            else:
                #i am in the excited state
                self.state.rho = self.ex_proj * self.state.rho * self.ex_proj / (1.-p0)
            
            #compute the dissipation
            diss += self.diss_coeff * s / self.b / self.dt

        # if it's NOT  a measurement
        else:
            #if it's a thermalization
            if d_act == 1:
                #switch off the coupling between qubits
                g_act = 0.
                #add the creation/destruction operators
                x = self.b * self.e0 * u_act
                c_ops = [np.sqrt(self.gamma*np.abs(self.bose(x))) * self.c_up,
                        np.sqrt(self.gamma*np.abs(1.+self.bose(x))) * self.c_down]
     
            #if it's a unitary evolution
            elif d_act == 2:
                #switch on the coupling
                g_act = self.g
                #set no dissipation
                c_ops = []

            #compute the hamiltonian
            h = self.hamiltonian(u_act, g_act)

            #compute initial energy
            initial_energy = qt.expect(h, self.state.rho)

            #perform the state evolution                
            self.state.rho = qt.mesolve(h, self.state.rho, [0.,self.dt], c_ops=c_ops,
                                         options=self.mesolver_options).states[1]
        
            #compute final energy
            final_energy = qt.expect(h, self.state.rho)

            #the heat is the energy difference (if unitary, this should be zero, since the initial
            # energy is computed *after* the quench, since we use the new value of u in the hamiltonian
            pow += self.pow_coeff*(final_energy-initial_energy)/self.dt if d_act == 1 else 0.

        #complete state evolution
        self.state.u = u_act

        #compute reward
        reward = self.a*pow - (1.-self.a)*diss

        #return        
        return self.current_state(), reward, False, {"multi_obj": 
            np.array([pow, -diss], dtype=np.float32)}
    
    def render(self):
        """ Required by gym.Env. Prints the current state."""
        print(self.current_state())

    def reset_internal_state_variables(self):
        """ sets initial values for the state """

        #set initial state of qubit zero to 50% gs and excited
        rho_0 = 0.5*self.gs*self.gs.dag() + 0.5*self.ex*self.ex.dag()

        #set initial state of qubit 1 to pure state (g.s.)
        rho_1 = self.gs*self.gs.dag()

        #initialize state as product of these two (uncorrelated)
        self.state.rho = qt.tensor(rho_0, rho_1)
        
        #set initial u to half way
        self.state.u = 0.5*(self.min_u+self.max_u)

    def set_current_state(self, state):
        """ 
        Allows to set the current state of the environment. This function must be implemented in order
        for sac_tri.SacTrain.load_full_state() to properly load a saved training session.

        Args:
            state (type specificed by self.observation_space): state of the environment
        """

        self.state = self.array_state_to_state_obj(state)

    def current_state(self):
        """ Returns the current state as the type specificed by self.observation_space"""
        
        self.im_part_indices 

        rho_np = self.state.rho.full()
        rho_re_np = np.real(rho_np)
        rho_im_np = np.imag(rho_np)
        re_array = rho_re_np[self.re_part_indices]
        im_array = rho_im_np[self.im_part_indices]
        u_array = np.array([self.state.u])

        return np.concatenate([re_array, im_array, u_array])
           
    def init_qt_vars(self):
        """ initialize qutip object to speed up the construction of the hamiltonian """

        sp = qt.operators.sigmap()
        sm = qt.operators.sigmam()
        sx = qt.operators.sigmax()
        id2 = qt.operators.identity(2)

        #define gs and excited state for a single qubit
        self.gs = qt.states.basis(2,1)
        self.ex = qt.states.basis(2,0)

        #initialize hamiltonian terms used by self.hamiltonian(u,g)
        self.h_0 = qt.tensor(self.e0*sp*sm, id2 )
        self.h_1 = qt.tensor(id2, self.e0*sp*sm )
        if self.counter_rot:
            self.h_int = qt.tensor(sx,sx) 
        else:
            self.h_int = qt.tensor(sp,sm) + qt.tensor(sm,sp)

        #initialize the indices to set and extract state. If I change this, update also state_to_block_sphere
        self.re_part_indices = np.triu_indices(4,0)
        self.im_part_indices = np.triu_indices(4,1)

        #initialize the collapse operators (that still have to be multiplied by the rates)
        self.c_up = qt.tensor(sp, id2)
        self.c_down = qt.tensor(sm, id2)

        #projectors for the measurement operators
        self.gs_proj = qt.tensor(id2, self.gs*self.gs.dag())
        self.ex_proj = qt.tensor(id2, self.ex*self.ex.dag())

        #setup the mesolver options
        if self.mesolver_nsteps is None:
            self.mesolver_options = qt.solver.Options()
        else:
            self.mesolver_options = qt.solver.Options(nsteps=self.mesolver_nsteps)

    def hamiltonian(self, u, g):
        """
        returns the Hamiltonian given the control u

        Args:
            u (float): value of the control
        """
        return u*(self.h_0 + self.h_1) + g*self.h_int

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

    def entropy(self, p):
         return -xlogy(p,p) - xlogy(1.-p,1.-p)

    def bose(self, x):
        return 1./np.expm1(x)
    
    def array_state_to_state_obj(self, array_state):
        #parse the state
        re_array = array_state[:10]
        im_array = array_state[10:16]
        u_val = array_state[-1]
        
        #create the state matrix
        re_mat = np.zeros((4,4), dtype=np.complex)
        im_mat = np.zeros((4,4), dtype=np.complex)

        #create the real part
        re_mat[self.re_part_indices] = re_array
        re_mat += re_mat.T
        np.fill_diagonal(re_mat, re_mat.diagonal()/2.)

        #create the imaginary part
        im_mat[self.im_part_indices] = 1.j*im_array
        im_mat += im_mat.T.conjugate()

        #create the object state
        obj_state =  TwoQubitResonantFeedbackDemonPowDiss.State()

        #load the qutip state
        obj_state.rho = qt.Qobj(re_mat+im_mat, dims=[[2,2],[2,2]],shape=(4,4))
        obj_state.u = u_val

        return obj_state

    def state_to_block_sphere(self, state):
        #this takes the vector form of the state of input, and returns the bloch vector of qubit 0 (the thermalizing one)
        #here I use as convention that negative rz is the ground state of the positive u hamiltonian.
        
        #create a "fake self" to make array_state_to_state_obj work without initializing this object
        fake_self = types.SimpleNamespace(re_part_indices = np.triu_indices(4,0), im_part_indices = np.triu_indices(4,1))
        
        #i reconstruct the qutip state from the array representation
        obj_state = TwoQubitResonantFeedbackDemonPowDiss.array_state_to_state_obj(fake_self, state)

        #compute the reduced density matrix
        rho_reduced = obj_state.rho.ptrace(0)

        #compute the 3 components of the block vector
        rx = qt.expect(qt.operators.sigmax(), rho_reduced)
        ry = qt.expect(qt.operators.sigmay(), rho_reduced)
        rz = qt.expect(qt.operators.sigmaz(), rho_reduced)

        #return it as a vector
        return np.array([rx, ry, rz])

    def state_to_concurrence(self, state):
        #this takes the vector form of the state of input, and returns the concurrence between the two qubits
        #here I use as convention that negative rz is the ground state of the positive u hamiltonian.
        
        #create a "fake self" to make array_state_to_state_obj work without initializing this object
        fake_self = types.SimpleNamespace(re_part_indices = np.triu_indices(4,0), im_part_indices = np.triu_indices(4,1))
        
        #i reconstruct the qutip state from the array representation
        obj_state = TwoQubitResonantFeedbackDemonPowDiss.array_state_to_state_obj(fake_self, state)

        #return the concurrence
        return qt.entropy.concurrence(obj_state.rho)


