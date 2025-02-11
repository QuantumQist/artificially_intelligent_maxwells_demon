import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical


"""
This module defines the NNs used to parameterize the value and policy function when we have
a single continuous action, and one discrete action that can take values (0,1,2).
"""

#Constants used by the NNs to prevent numerical instabilities
PROBS_MIN = 2e-5
PROBS_MAX = 1.
UNIFORM_SIGMA =3.46

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    return a sequential net of fully connected layers

    Args:
        sizes(tuple(int)): sizes of all the layers
        activation: activation function for all layers except for the output layer
        output_activation: activation to use for output layer only. It can also be the
            string "soft_max", in which case the soft_max will be performed

    Returns:
        stacked fully connected layers
    """
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes)-2:
            act = activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        else:
            if output_activation == "soft_max":
                layers += [nn.Linear(sizes[j], sizes[j+1]), nn.Softmax(dim=-1)]
            else:
                act = output_activation
                layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    """counts the variables of a module """
    return sum([np.prod(p.shape) for p in module.parameters()])

class RescaleInput(nn.Module):
    def __init__(self, lower_bounds, upper_bounds):
        """
        I assume lower_bounds and upper_bouds are 1D arrays with the bounds. I will try to properly 
        handle batched and unbatched data
        """
        super().__init__()
        self.register_buffer("lower_bounds", torch.tensor(lower_bounds, dtype=torch.float32))
        self.register_buffer("upper_bounds", torch.tensor(upper_bounds, dtype=torch.float32))
    
    def forward(self, x):
        return ((x-self.lower_bounds)/(self.upper_bounds-self.lower_bounds)-0.5)*UNIFORM_SIGMA
        
class MLPActorCritic(nn.Module):
    """
    Module that contains two value functions, self.q1 and self.q2, and the policy self.pi.
    Both are parameterizes using an arbitrary number of fully connected layers.
    WARNING: it only works for 1 continuous action. Should be easy to generalize to more.
    WARNING: it only works for a discrete action with 3 possibilities: (0,1,2)

    Args:
        observation_space: observation space of the environment
        action_space: action space of sac_tri_envs environments
        hidden_sizes (tuple(int)): size of each of the hidden layers that will be addded to 
            both the value and policy functions
        activation: activation function for all layers except for the output layer
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, min_cov_eigen = 1.e-8):
        super().__init__()

        #raise error if the size isn't supported
        if action_space[0].n != 3:
            raise NameError(f"Only works for 3 discrete actions, not for {action_space[0].n}.")
        
        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation,
                    min_cov_eigen=min_cov_eigen)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.alpha_d = Alpha()
        self.alpha_c = Alpha()
        
        #create tensors that will be used to determine the discrete action
        self.zero_float = torch.tensor(0., dtype=torch.float32)
        self.one_float = torch.tensor(1., dtype=torch.float32)

    def act(self, obs, deterministic=False):
        """ return the action, chosen according to deterministic, given a single unbatched observation obs """
        with torch.no_grad():
            b_action, pi0_action, pi1_action, pi2_action, _, _, _, _, _= self.pi(obs.view(1,-1), deterministic, False)
            if torch.isclose(b_action[0],self.zero_float):
                return (b_action[0], pi0_action[0])
            elif torch.isclose(b_action[0],self.one_float):
                return (b_action[0], pi1_action[0])
            else:
                return (b_action[0], pi2_action[0])
    
    def alpha_d_no_grad(self):
        """
        returns the current value of alpha without gradients
        """
        with torch.no_grad():
            return self.alpha_d()

    def alpha_c_no_grad(self):
        """
        returns the current value of alpha without gradients
        """
        with torch.no_grad():
            return self.alpha_c()

class MLPQFunction(nn.Module):
    """
    Class representing a q-value function, implemented with fully connected layers
    that stack the state-action as input, and output the value of each discrete action (3 heads)

    Args:
        obs_dim(int): number of continuous state variables
        act_dim(int): number of continuous actions
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation):
        super().__init__()

        #determine action and observation spaces
        obs_dim = observation_space.shape[0]
        act_dim = action_space[1].shape[0]
        obs_act_lower_bounds = np.concatenate([observation_space.low, action_space[1].low ])
        obs_act_upper_bounds = np.concatenate([observation_space.high, action_space[1].high ])
        
        #define the input layer that places data in a bound
        self.rescale_input = RescaleInput(obs_act_lower_bounds, obs_act_upper_bounds)
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [3], activation)


    def forward(self, obs, act):
        """
        Args:
            obs(torch.Tensor): batch of observations
            act(torch.Tensor): batch of continuous actions

        Returns:
            (torch.Tensor): 3 elements representing the value of each of the 3 discrete actions
        """
        out = self.rescale_input(torch.cat([obs, act], dim=-1))
        out = self.q(out)
        return out

class SquashedGaussianMLPActor(nn.Module):
    """
    Class representing the policy, implemented with fully connected layers
    that take the state as input, and output the marginal probability of each discrete action,
    and the average and sigma of the conditional density probability of chosing a continuous
    action, given each of the 3 discrete actions. The density probability for the conditional
    continuous action is a squashed gaussian policy.

    Args:
        obs_dim(int): number of continuous state variables
        act_dim(int): number of continuous actions
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
        act_lower_limit (float): minimum value of the single continuous action
        act_upper_limit (float): maximum value of the single continuous action
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation,
                    min_cov_eigen=1.e-8):
        super().__init__()

        #load state and action dimensions and bounds
        obs_dim = observation_space.shape[0]
        act_dim = action_space[1].shape[0]
        obs_lower_bounds = observation_space.low
        obs_upper_bounds = observation_space.high
        self.register_buffer("act_lower_bounds", torch.tensor(action_space[1].low, dtype=torch.float32))
        self.register_buffer("act_upper_bounds", torch.tensor(action_space[1].high, dtype=torch.float32))
        self.act_dim = act_dim

        #define the input layer that places data in a bound
        self.rescale_input = RescaleInput(obs_lower_bounds, obs_upper_bounds)
        
        #main network taking the state as input, and passing it through all hidden layers
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        
        #output layer producing the average of the three conditional probability densities
        self.mu0_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.mu1_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.mu2_layer = nn.Linear(hidden_sizes[-1], act_dim)
        #output layer producing log(sigma) of the three conditional probability densities
        self.m0_layer = nn.Linear(hidden_sizes[-1], act_dim*act_dim)
        self.m1_layer = nn.Linear(hidden_sizes[-1], act_dim*act_dim)
        self.m2_layer = nn.Linear(hidden_sizes[-1], act_dim*act_dim)
        #I create the matrix l_id = lambda *Id for the covariance matrix
        self.register_buffer("l_id", torch.eye(act_dim)*min_cov_eigen)
        #i compute the quantity to subtract to the log_probs because of the action rescaling
        #NEW: i don't do this anymore, since it's just an additive quantity that depends on the
        #details of the interval
        # self.register_buffer("log_prob_rescaling",
        #              torch.log(0.5*(self.act_upper_bounds-self.act_lower_bounds)).sum())
        #output layer producing the marginal probability of each of the 3 discrete actions
        p_activation = "soft_max"
        self.p_layer = mlp(sizes=[hidden_sizes[-1], 3], activation=p_activation, output_activation=p_activation)
        
    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs(torch.Tensor): batch of observations
            deterministic(bool): if the actions should be chosen deterministally or not
            with_logprob(bool): if the log of the probability should be computed and returned

        Returns:
            b_action(torch.Tensor): the chosen discrete action (0,1,2)
            pi0_action(torch.Tensor): the chosen continuous action if discrete action 0 is chosen
            pi1_action(torch.Tensor): the chosen continuous action if discrete action 1 is chosen
            pi2_action(torch.Tensor): the chosen continuous action if discrete action 2 is chosen
            p(torch.Tensor): the probability of chosing each of the 3 discrete actions
            logp_pi0(torch.Tensor): total log probability of chosing discrete action 0 and the 
                corresponding continuous action
            logp_pi1(torch.Tensor): total log probability of chosing discrete action 1 and the 
                corresponding continuous action
            logp_pi2(torch.Tensor): total log probability of chosing discrete action 2 and the 
                corresponding continuous action
        """
        #run the state through the network
        net_out = self.rescale_input(obs)
        net_out = self.net(net_out)
        mu0 = self.mu0_layer(net_out)
        mu1 = self.mu1_layer(net_out)
        mu2 = self.mu2_layer(net_out)
        m0 = self.m0_layer(net_out).view(-1,self.act_dim,self.act_dim)
        m1 = self.m1_layer(net_out).view(-1,self.act_dim,self.act_dim)
        m2 = self.m2_layer(net_out).view(-1,self.act_dim,self.act_dim)
        cov0_mat = (torch.transpose(m0,1,2) @ m0) + self.l_id
        cov1_mat = (torch.transpose(m1,1,2) @ m1) + self.l_id
        cov2_mat = (torch.transpose(m2,1,2) @ m2) + self.l_id
        p = self.p_layer(net_out)
        #added this clamp to prevent the argument of softmax from exploding
        p = torch.clamp(p, PROBS_MIN,PROBS_MAX)

        #after clamping, I renormalize the probabilities
        p = p / p.sum(-1, keepdim=True)

        # Pre-squash (tanh) distribution and sample
        b_distribution = Categorical(probs=p)
        pi0_distribution = MultivariateNormal(mu0, cov0_mat)
        pi1_distribution = MultivariateNormal(mu1, cov1_mat)
        pi2_distribution = MultivariateNormal(mu2, cov2_mat)
        if deterministic:
            b_action = torch.argmax(p, dim=-1).type(torch.float32)
            pi0_action = mu0
            pi1_action = mu1
            pi2_action = mu2
        else:
            b_action = b_distribution.sample().type(torch.float32)
            pi0_action = pi0_distribution.rsample()
            pi1_action = pi1_distribution.rsample()
            pi2_action = pi2_distribution.rsample()
            
        #if necessary, compute the log of the probabilities
        if with_logprob:
            logp_pi0 = pi0_distribution.log_prob(pi0_action)
            logp_pi1 = pi1_distribution.log_prob(pi1_action)
            logp_pi2 = pi2_distribution.log_prob(pi2_action)
            #change of distribution when going from gaussian to Tanh
            logp_pi0 -= (2*(np.log(2) - pi0_action - F.softplus(-2*pi0_action))).sum(axis=1)
            logp_pi1 -= (2*(np.log(2) - pi1_action - F.softplus(-2*pi1_action))).sum(axis=1)
            logp_pi2 -= (2*(np.log(2) - pi2_action - F.softplus(-2*pi2_action))).sum(axis=1)
            #change of distribution when going from Tanh to the interval [act_lower_limit,act_upper_limit]
            #NEW: i don't do this anymore, since it just introduces a constant that depends on the details
            #of the interval
            # logp_pi0 -= self.log_prob_rescaling
            # logp_pi1 -= self.log_prob_rescaling 
            # logp_pi2 -= self.log_prob_rescaling
            #NOTICE: as opposed to previous code, now i DO NOT include the log of the discrete probability
            #these log probs are just of the continuous part

            #compute entropy of the discrete action alone
            p_entropy = b_distribution.entropy()
        else:
            logp_pi0 = None
            logp_pi1 = None
            logp_pi2 = None
            p_entropy = None

        #Apply Tanh to the sampled gaussian
        pi0_action = torch.tanh(pi0_action)
        pi1_action = torch.tanh(pi1_action)
        pi2_action = torch.tanh(pi2_action)
        #Apply the shift of the action to the correct interval
        pi0_action = self.act_lower_bounds + 0.5*(pi0_action + 1.)*(self.act_upper_bounds-self.act_lower_bounds)
        pi1_action = self.act_lower_bounds + 0.5*(pi1_action + 1.)*(self.act_upper_bounds-self.act_lower_bounds)
        pi2_action = self.act_lower_bounds + 0.5*(pi2_action + 1.)*(self.act_upper_bounds-self.act_lower_bounds)

        return b_action, pi0_action, pi1_action, pi2_action, p, p_entropy, logp_pi0, logp_pi1, logp_pi2

class Alpha(nn.Module):
    """

    Args:
       
    """
    def __init__(self):
        super().__init__()
        #initialize it to some "large" initial value
        self.log_alpha_val = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        
    def forward(self):
        """
        Args:

        Returns:
        """
        return torch.exp(self.log_alpha_val)

    