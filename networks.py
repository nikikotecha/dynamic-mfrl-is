import argparse
import copy
from typing import Union

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from sequential_social_dilemma_games.social_dilemmas.envs.cleanup import CleanupEnvMultiType, CleanupEnvReward
#from sequential_social_dilemma_games.social_dilemmas.envs.harvest import HarvestEnvReward
#from utils.utils_all import init_weights

from env3rundivproduct import MultiAgentInvManagementDiv
from utils import init_weights

class Actor(nn.Module):
    """
    Actor network based on MLP structure.
    """
    def __init__(self, observation_size: int, action_size: int, hidden_dims: list):
        """
        Create a new actor network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        Lastly, we use softmax to return the action probabilities.

        Parameters
        ----------
        observation_size: int
        action_size: int
        hidden_dims: list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128, 64], [64, 32], [32, action_size]].
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.observation_size: int = observation_size
        self.action_size: int = action_size
        layer_dims: list = self.make_layer_dims(hidden_dims)
        for layer_dim in layer_dims:
            fc_i = nn.Linear(layer_dim[0], layer_dim[1])
            self.layers.append(fc_i)
        self.num_layers: int = len(self.layers)

        self.mu_layer = nn.Linear(1, 1)
        self.log_std_layer = nn.Linear(1, 1)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20


    def make_layer_dims(self, hidden_dims: list) -> list:
        """
        Make a list which contains dimensions of layers.
        Each element denotes the dimension of a linear layer.

        Parameters
        ----------
        hidden_dims: list

        Returns
        -------
        layer_dims: list
        """
        layer_dims = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [self.observation_size, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], self.action_size] # x2 to get mean and std
        layer_dims.append(layer_dim)
        return layer_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        Input of the actor network will be the batches of individual observations.

        Parameters
        ----------
        x: torch.Tensor
            Batches of individual observations which size should be (N, observation_size).
            ex. observation_size = 15 * 15 if self.one_hot_obs is False.

        Returns
        -------
        x: torch.Tensor
            Return the action probability using softmax (action).
            The shape will be (N, output_size: action_size).
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        mean = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        #dist = torch.distributions.Normal(mean, std)
        #x = dist.rsample()

        #x = F.tanh(x)

        #x = F.softmax(x, dim=-1) this is for discrete, changing so its continuous 
        
        return mean, std 


class Critic(nn.Module):
    """
    Critic network based on MLP structure.
    """
    def __init__(self, observation_size: int, action_size: int, mean_action_size: int, hidden_dims: list):
        """
        Create a new critic network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.

        Parameters
        ----------
        observation_size: int
        action_size: int
        mean_action_size: int
        hidden_dims: list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128 + mean_action_size, 64], [64, 32], [32, action_size]].
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.observation_size: int = observation_size
        self.action_size: int = action_size
        self.mean_action_size: int = mean_action_size
        layer_dims: list = self.make_layer_dims(hidden_dims)
        for layer_dim in layer_dims:
            fc_i = nn.Linear(layer_dim[0], layer_dim[1])
            self.layers.append(fc_i)
        self.num_layers: int = len(self.layers)

    def make_layer_dims(self, hidden_dims: list) -> list:
        """
        Make a list which contains dimensions of layers.
        Each element denotes the dimension of a linear layer.

        Parameters
        ----------
        hidden_dims: list

        Returns
        -------
        layer_dims: list
        """
        layer_dims = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [self.observation_size, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + self.mean_action_size, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], self.action_size]
        layer_dims.append(layer_dim)

        return layer_dims

    def forward(self, observation: torch.Tensor, mean_action: list) -> torch.Tensor:
        """
        Forward propagation.
        Inputs of the critic network are batches of individual observations and mean actions.
        Critic will return the q values for all actions.

        Parameters
        ----------
        observation: torch.Tensor
            Batches of individual observations which size should be (N, observation_size).
            ex. observation_size = 15 * 15 if self.one_hot_obs is False.
        mean_action: list
            Each element is batches of mean actions for each "action" type.

        Returns
        -------
        x: torch.Tensor
            q values of all actions for observation and mean_action.
            The shape will be (N, action_size).
            ex. action_size = 6.
        """
        x = self.layers[0](observation)
        x = F.relu(x)
        m_act = copy.deepcopy(mean_action)
        m_act.insert(0, x)
        x = torch.cat(m_act, dim=-1)
        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        x.view(-1, self.action_size)

        return x


class Psi(nn.Module):
    """
    Psi network based on MLP structure.
    """
    def __init__(self, observation_size: int, action_size: int, mean_action_size: int, feature_size: int, hidden_dims: list):
        """
        Create a new psi (successor feature) network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        We will reshape the last outcome to show the features for all actions.

        Parameters
        ----------
        observation_size: int
        action_size: int
        mean_action_size: int
        feature_size: int
        hidden_dims: list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128 + mean_action_size, 64], [64, 32], [32, action_size * feature_size]].
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.observation_size: int = observation_size
        self.action_size: int = action_size
        self.mean_action_size: int = mean_action_size
        self.feature_size: int = feature_size
        layer_dims: list = self.make_layer_dims(hidden_dims)
        for layer_dim in layer_dims:
            fc_i = nn.Linear(layer_dim[0], layer_dim[1])
            self.layers.append(fc_i)
        self.num_layers: int = len(self.layers)

    
    def make_layer_dims(self, hidden_dims: list) -> list:
        """
        Make a list which contains dimensions of layers.
        Each element denotes the dimension of a linear layer.

        Parameters
        ----------
        hidden_dims: list

        Returns
        -------
        layer_dims: list
        """
        layer_dims = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [self.observation_size, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + self.mean_action_size, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], self.action_size * self.feature_size]
        layer_dims.append(layer_dim)
        return layer_dims

    def forward(self, observation: torch.Tensor, mean_action: list) -> torch.Tensor:
        """
        Forward propagation.
        Inputs of the psi network are batches of individual observations and mean actions.
        Psi will return the successor features (psi) for all actions.
        Therefore, the shape of the outcome of Psi will be (N, action_size, feature_size).

        Parameters
        ----------
        observation: torch.Tensor
            Batches of individual observations which size should be (N, observation_size).
            ex. observation_size = 15 * 15 if self.one_hot_obs is False.
        mean_action: list
            Each element is batches of mean actions for each "action" type.

        Returns
        -------
        x: torch.Tensor
            Return the psi values of all actions for observation and mean_action.
            The shape will be (N, action_size, feature_size).
            ex. action_size = 6.
        """
        x = self.layers[0](observation)
        x = F.relu(x)
        m_act = copy.deepcopy(mean_action)
        m_act.insert(0, x)
        x = torch.cat(m_act, dim=-1)
        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        x = x.view(-1, self.action_size, self.feature_size)

        return x


class Networks(object):
    """
    Define networks (actor-critic / actor-psi / critic / psi).
    """
    def __init__(self, env, args):
        """

        Parameters
        ----------
        env: CleanupEnvReward | CleanupEnvMultiType | HarvestEnvReward
        args: argparse.Namespace
        """
        self.args: argparse.Namespace = args
        self.observation_num_classes: int = int(env.observation_space.high.max() + 1)
        self.observation_size: int = self.get_observation_size(env)
        self.num_types: int = env.num_types
        self.num_agents: list = env.num_agents_type
        self.agent_types = env.agent_types
        self.action_size: list = self.get_action_size(env)
        self.mean_action_size: int = sum(self.action_size)
        self.feature_size: int = int(np.prod(env.observation_space.shape))
        self.w: torch.Tensor = self.get_w()
        self.actor, self.actor_target = self.get_network('actor')  # list, list
        self.critic, self.critic_target = self.get_network('critic')  # list, list
        self.psi, self.psi_target = self.get_network('psi')  # list, list
        self.reuse_networks()
        self.actor_opt, self.actor_skd = self.get_opt_and_skd('actor')  # list, list
        self.critic_opt, self.critic_skd = self.get_opt_and_skd('critic')  # list, list
        self.psi_opt, self.psi_skd = self.get_opt_and_skd('psi')  # list, list

    
    def get_observation_size(self, env: Union[MultiAgentInvManagementDiv]) -> int:
        if self.args.mode_one_hot_obs:
            observation_size = np.prod(env.observation_space.shape) * self.observation_num_classes
        else:
            observation_size = np.prod(env.observation_space.shape)
        return int(observation_size)

    def get_action_size(self, env: Union[MultiAgentInvManagementDiv]) -> list:
        action_size = []
        for agent_type in range(env.num_types):
            #action_size.append(env.action_space[agent_type].n)
            action_size.append(env.action_space.shape[0])
        return action_size

    def get_w(self) -> torch.Tensor:
        if "cleanup" in self.args.env:
            w = torch.tensor([1 - self.args.lv_penalty, self.args.lv_incentive], dtype=torch.float)
        elif "harvest" in self.args.env:
            w = torch.tensor([1, self.args.lv_penalty, self.args.lv_incentive], dtype=torch.float)
        else:
            raise NotImplementedError
        return w

    def get_network(self, mode: str) -> (list, list):
        network = []
        is_actor = (self.args.mode_ac and mode == 'actor')
        is_critic = (not self.args.mode_psi and mode == 'critic')
        is_psi = (self.args.mode_psi and mode == 'psi')
        for agent_type in range(self.num_types):
            net = None
            if is_actor:
                net = Actor(observation_size=self.observation_size,
                            action_size=self.action_size[agent_type],
                            hidden_dims=self.args.h_dims_a)
                net.apply(init_weights)
            if is_critic:
                net = Critic(observation_size=self.observation_size,
                             action_size=self.action_size[agent_type],
                             mean_action_size=self.mean_action_size,
                             hidden_dims=self.args.h_dims_c)
                net.apply(init_weights)
            if is_psi:
                net = Psi(observation_size=self.observation_size,
                          action_size=self.action_size[agent_type],
                          mean_action_size=self.mean_action_size,
                          feature_size=self.feature_size,
                          hidden_dims=self.args.h_dims_p)
                net.apply(init_weights)
            network.append(net)
        network_target = copy.deepcopy(network)
        return network, network_target

    def get_opt_and_skd(self, mode: str) -> (list, list):
        optimizer = []
        scheduler = []
        is_actor = (self.args.mode_ac and mode == 'actor')
        is_critic = (not self.args.mode_psi and mode == 'critic')
        is_psi = (self.args.mode_psi and mode == 'psi')
        for agent_type in range(self.num_types):
            opt = None
            skd = None
            if is_actor:
                opt = optim.Adam(self.actor[agent_type].parameters(), lr=self.args.lr_a)
            if is_critic:
                opt = optim.Adam(self.critic[agent_type].parameters(), lr=self.args.lr_c)
            if is_psi:
                opt = optim.Adam(self.psi[agent_type].parameters(), lr=self.args.lr_p)
            if self.args.mode_lr_decay and (is_actor or is_psi or is_critic):  # opt is not None
                skd = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.999)
            optimizer.append(opt)
            scheduler.append(skd)
        return optimizer, scheduler

    def reuse_networks(self):
        if self.args.mode_reuse_networks:
            prev_dict = torch.load(self.args.file_path)
            for agent_type in range(self.num_types):
                # TODO: check whether we have to add opt_params or not.
                if self.args.mode_ac:
                    self.actor[agent_type].load_state_dict(prev_dict['actor'][agent_type])
                    self.actor_target[agent_type].load_state_dict(prev_dict['actor'][agent_type])
                if self.args.mode_psi:
                    self.psi[agent_type].load_state_dict(prev_dict['psi'][agent_type])
                    self.psi_target[agent_type].load_state_dict(prev_dict['psi'][agent_type])
                else:
                    self.critic[agent_type].load_state_dict(prev_dict['critic'][agent_type])
                    self.critic_target[agent_type].load_state_dict(prev_dict['critic'][agent_type])

    def get_boltzmann_policy(self, q_values: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Get Boltzmann policy using batches of q values.

        Parameters
        ----------
        q_values: torch.Tensor
            Q values of possible actions for the specific type.
            In other words, q_values = critic[agent_type](obs_input, m_act_input).
            q_values = Tensor: (N, action_size) (action_size depends on the type).
        
        beta: torch.Tensor
            Temperature (boltzmann parameter) for the specific type.
            In other words, beta = beta_t[i].
            beta = Tensor: (N,).

        Returns
        -------
        actions_probs: torch.Tensor
            Probabilities for possible actions for the specific type.
            actions_probs = Tensor: (N, action_size) (action_size depends on the type).
        """
        softmax_func = torch.nn.Softmax(dim=1)
        actions_probs = softmax_func(q_values / beta[:, None])

        return actions_probs

    def get_actions(self, obs: dict, prev_m_act: list, beta: float, is_target: bool = True):
        """
        Get actions.
        If env.args.mode_ac is True, it will use self.actor to get actions.
        Otherwise, it will use self.critic or self.psi to get actions based on Boltzmann policy.

        Parameters
        ----------
        obs: dict
            dict of observations of agents.
            ex. obs['agent-0'] = np.array(15,15)
        prev_m_act: list
            List of dict.
            Each dict contains previous mean actions of agents for each "action" type.
            ex. prev_m_act[0]['agent-0'] = np.zeros(env.action_space[0].n)
        beta: float
        is_target: bool
            True if it uses the target network.
            We might only use the target network when we use this function.

        Returns
        -------
        actions: dict
            dict of actions of "all" agents.
            ex. actions['agent-0'] = 3 (int)
        actions_probs: dict
            dict of action probabilities of "all" agents.
            It might be only used for denoting the agents' probabilities in the video.
        """
    
        agent_ids = list(obs.keys())
        agent_types = list(self.agent_types.values())  # env.agents_types = {agent_id: int}
        obs_t = [[] for _ in range(self.num_types)]
        prev_m_act_t = [[[] for _ in range(self.num_types)] for _ in range(self.num_types)]
        beta_t = [[] for _ in range(self.num_types)]
        for idx, agent_id in enumerate(agent_ids):
            agent_type = agent_types[idx]
            obs_t[agent_type].append(obs[agent_id])
            for action_type in range(self.num_types):
                prev_m_act_t[agent_type][action_type].append(prev_m_act[action_type][agent_id])
            beta_t[agent_type].append(beta)

        for agent_type in range(self.num_types):
            obs_t[agent_type] = np.array(obs_t[agent_type])
            for action_type in range(self.num_types):
                prev_m_act_t[agent_type][action_type] = np.array(prev_m_act_t[agent_type][action_type])
            beta_t[agent_type] = np.array(beta_t[agent_type])

        with torch.no_grad():
            tensors = self.to_tensors(obs_t=obs_t, m_act_t=prev_m_act_t, beta_t=beta_t)
            actions_probs = {}
            actions = {}
            idx = 0
            for agent_type in range(self.num_types):
                obs_input = tensors['obs'][agent_type]  # obs_input = (n_t, observation_size)
                m_act_input = tensors['m_act'][agent_type]  # action_type별 tensor 존재
                beta_input = tensors['beta'][agent_type]
                # TODO: check whether mode_ac and mode_psi properly work.
                if self.args.mode_ac:  # based on actor.
                    if is_target:
                        mean, std = self.actor_target[agent_type](obs_input)
                    else:
                        mean, std = self.actor[agent_type](obs_input)
                else:
                    if self.args.mode_psi:  # based on psi.
                        if is_target:
                            psi_target = self.psi_target[agent_type](obs_input, m_act_input)
                            q_target = torch.tensordot(psi_target, self.w, dims=([2], [0]))
                            action_probs = self.get_boltzmann_policy(q_target, beta_input)
                        else:
                            psi = self.psi[agent_type](obs_input, m_act_input)
                            q = torch.tensordot(psi, self.w, dims=([2], [0]))
                            action_probs = self.get_boltzmann_policy(q, beta_input)
                    else:  # based on critic.
                        if is_target:
                            q_target = self.critic_target[agent_type](obs_input, m_act_input)
                            action_probs = self.get_boltzmann_policy(q_target, beta_input)
                        else:
                            q = self.critic[agent_type](obs_input, m_act_input)
                            action_probs = self.get_boltzmann_policy(q, beta_input)
                action_dists = distributions.Normal(mean, std)
                action = action_dists.rsample()
                action = F.tanh(action)  #ensures its within -1 and 1
                for i in range(self.num_agents[agent_type]):  # n_t
                    #actions_probs[agent_ids[idx]] = action_probs[i]
                    actions[agent_ids[idx]] = action[i].item()
                    idx += 1
        actions_probs = {}
        return actions, actions_probs

    def preprocess(self, samples: list):
        """
        The purpose of this function is to collect data for each "category (obs, act, ...)", "agent_type",
        and (additionally) "action_type".
        It will return the list of ndarray for each category.

        Parameters
        ----------
        samples: list
            ex. [(obs, act, rew, m_act, n_obs, fea, beta), (obs, act, rew, m_act, n_obs, fea, beta), ...]
            ex. obs = {'agent-0': np.array(15,15), 'agent-1': np.array(15,15), ...}
                act = {'agent-0': 1, 'agent-1': 0, ...}
                rew = {'agent-0': 1, 'agent-1': 0, ...}
                m_act = [{'agent-0': np.zeros(env.action_space[0].n), 'agent-1': np.zeros(env.action_space[0].n),...},
                         {'agent-0': np.zeros(env.action_space[1].n), 'agent-1': np.zeros(env.action_space[1].n),...}]
                n_obs = {'agent-0': np.array(15,15), 'agent-1': np.array(15,15), ...}
                fea = {'agent-0': np.array(feature_size), 'agent-1': np.array(feature_size), ...}
                beta = 0.9

        Returns
        -------
        obs_t: list
            Each element of obs_t is a numpy.ndarray of individual observations for each type.
            obs_t[i] = ndarray: (N * num_agents[i], 15, 15) (15 can be changed, N is the number of samples).
        act_t: list
            Each element of act_t is a numpy.ndarray of individual actions for each type.
            act_t[i] = ndarray: (N * num_agents[i],) (N is the number of samples).
            ex. act_t[i] = np.array([3,2,1,1,1,0,5,4,...]).
        rew_t: list
            Each element of rew_t is a numpy.ndarray of individual rewards for each type.
            rew_t[i] = ndarray: (N * num_agents[i],) (N is the number of samples).
            ex. rew_t[0] = np.array([0,0,0,1,0,1,0,1,1,...]).
        m_act_t: list
            Each element of m_act_t is also a list (denoted as l_t (_t means the "agent" type)).
            Each element of the list l_t is a numpy.ndarray of individual mean actions for each "action" type.
            m_act_t[i][j] = ndarray: (N * num_agents[i], env.action_space[j].n) (N is the number of samples).
        n_obs_t: list
            Each element of n_obs_t is a numpy.ndarray of individual next observations for each type.
            n_obs_t[i] = ndarray: (N * num_agents[i], 15, 15) (15 can be changed, N is the number of samples).
        fea_t: list
            Each element of fea_t is a numpy.ndarray of individual features for each type.
            fea_t[i] = ndarray: (N * num_agents[i], feature_size) (N is the number of samples).
        beta_t: list
            Each element of beta_t is a numpy.ndarray of betas for each type.
            beta_t[i] = ndarray: (N * num_agents[i],) (N is the number of samples).
        """
        obs_t, act_t, rew_t, n_obs_t, fea_t, beta_t = [[[] for _ in range(self.num_types)] for _ in range(6)]
        m_act_t = [[[] for _ in range(self.num_types)] for _ in range(self.num_types)]
        for sample in samples:
            obs, act, rew, m_act, n_obs, fea, beta = sample
            agent_ids = list(obs.keys())
            agent_types = list(self.agent_types.values())  # env.agents_types = {agent_id: int}
            for idx, agent_id in enumerate(agent_ids):
                agent_type = agent_types[idx]
                obs_t[agent_type].append(obs[agent_id])
                act_t[agent_type].append(act[agent_id])
                rew_t[agent_type].append(rew[agent_id])
                for action_type in range(self.num_types):
                    m_act_t[agent_type][action_type].append(m_act[action_type][agent_id])
                n_obs_t[agent_type].append(n_obs[agent_id])
                fea_t[agent_type].append(fea[agent_id])
                beta_t[agent_type].append(beta)

        for agent_type in range(self.num_types):
            obs_t[agent_type] = np.array(obs_t[agent_type])
            act_t[agent_type] = np.array(act_t[agent_type], dtype=np.int64)
            rew_t[agent_type] = np.array(rew_t[agent_type])
            for action_type in range(self.num_types):
                m_act_t[agent_type][action_type] = np.array(m_act_t[agent_type][action_type])
            n_obs_t[agent_type] = np.array(n_obs_t[agent_type])
            fea_t[agent_type] = np.array(fea_t[agent_type])
            beta_t[agent_type] = np.array(beta_t[agent_type])

        return obs_t, act_t, rew_t, m_act_t, n_obs_t, fea_t, beta_t

    def to_tensors(self, obs_t=None, act_t=None, rew_t=None, m_act_t=None, n_obs_t=None, fea_t=None, beta_t=None):
        """
        For each input, this function make a list of ndarrays to a list of tensors.
        If args.mode_one_hot_obs, observations will be changed into one-hot encoded version.

        Parameters
        ----------
        obs_t: list
            Each element of obs_t is a numpy.ndarray of individual observations for each type.
            obs_t[i] = ndarray: (N[i], 15, 15) (15 can be changed, N[i] is the number of rows for type i).
        act_t: list
            Each element of act_t is a numpy.ndarray of individual actions for each type.
            act_t[i] = ndarray: (N[i],) (N[i] is the number of rows for type i).
            ex. act_t[i] = np.array([3,2,1,1,1,0,5,4,...]).
        rew_t: list
            Each element of rew_t is a numpy.ndarray of individual rewards for each type.
            rew_t[i] = ndarray: (N[i],) (N[i] is the number of rows for type i).
            ex. rew_t[0] = np.array([0,0,0,1,0,1,0,1,1,...]).
        m_act_t: list
            Each element of m_act_t is also a list (denoted as l_t (_t means the "agent" type)).
            Each element of the list l_t is a numpy.ndarray of individual mean actions for each "action" type.
            m_act_t[i][j] = ndarray: (N[i], env.action_space[j].n) (N[i] is the number of rows for type i).
        n_obs_t: list
            Each element of n_obs_t is a numpy.ndarray of individual next observations for each type.
            n_obs_t[i] = ndarray: (N[i], 15, 15) (15 can be changed, N[i] is the number of rows for type i).
        fea_t: list
            Each element of fea_t is a numpy.ndarray of individual features for each type.
            fea_t[i] = ndarray: (N[i], feature_size) (N[i] is the number of rows for type i).
        beta_t: list
            Each element of beta_t is a numpy.ndarray of betas for each type.
            beta_t[i] = ndarray: (N[i],) (N[i] is the number of rows for type i).

        Returns
        -------
        tensors: dict
            Dict of lists.
            Each element of the list is a tensor of something for each type.
            ex. tensors['obs'][0] = torch.Tensor: (N, observation_size: 15 * 15 * 6)
        """
        def get_tensor(x, size=None):
            """
            Return the float and reshaped tensor.

            Parameters
            ----------
            x: torch.Tensor or numpy.ndarray
            size: int

            Returns
            -------
            x_tensor: torch.Tensor
            """
            if type(x) is torch.Tensor:
                x_tensor = x.type(dtype=torch.float)
            else:  # numpy.ndarray
                x_tensor = torch.tensor(x, dtype=torch.float)
            x_tensor = x_tensor.view(-1, size)
            return x_tensor

        tensors = {i: None for i in ['obs', 'act', 'rew', 'm_act', 'n_obs', 'fea', 'beta']}
        with torch.no_grad():
            if obs_t is not None:
                obs_t_tensor = []
                for agent_type in range(self.num_types):
                    if self.args.mode_one_hot_obs:
                        # F.one_hot takes tensor with index values of shape (*) and returns a tensor of shape (*, num_classes)
                        obs_tensor = torch.tensor(obs_t[agent_type], dtype=torch.int64)
                        obs_tensor = F.one_hot(obs_tensor, num_classes=self.observation_num_classes)
                        obs_tensor = get_tensor(obs_tensor, self.observation_size)  # Shape: (N, observation_size)
                    else:
                        obs_tensor = torch.tensor(obs_t[agent_type], dtype=torch.int64)
                        obs_tensor = get_tensor(obs_tensor, self.observation_size)  # Shape: (N, observation_size)
                    obs_t_tensor.append(obs_tensor)
                tensors['obs'] = obs_t_tensor
            if act_t is not None:
                act_t_tensor = []
                for agent_type in range(self.num_types):
                    act_tensor = torch.tensor(act_t[agent_type], dtype=torch.int64)  # Shape should be (N,)
                    act_t_tensor.append(act_tensor)
                tensors['act'] = act_t_tensor
            if rew_t is not None:
                rew_t_tensor = []
                for agent_type in range(self.num_types):
                    rew_tensor = get_tensor(rew_t[agent_type], 1)  # Shape should be (N, 1)
                    rew_t_tensor.append(rew_tensor)
                tensors['rew'] = rew_t_tensor
            if m_act_t is not None:
                m_act_t_tensor = []  # for agent_type
                for agent_type in range(self.num_types):
                    m_act_t_t_tensor = []  # for action_type
                    for action_type in range(self.num_types):
                        m_act_tensor = get_tensor(m_act_t[agent_type][action_type], self.action_size[action_type])  # Shape should be (N, action_size[type])
                        m_act_t_t_tensor.append(m_act_tensor)
                    m_act_t_tensor.append(m_act_t_t_tensor)
                tensors['m_act'] = m_act_t_tensor
            if n_obs_t is not None:
                n_obs_t_tensor = []
                for agent_type in range(self.num_types):
                    if self.args.mode_one_hot_obs:
                        # F.one_hot takes tensor with index values of shape (*) and returns a tensor of shape (*, num_classes)
                        n_obs_tensor = torch.tensor(n_obs_t[agent_type], dtype=torch.int64)
                        n_obs_tensor = F.one_hot(n_obs_tensor, num_classes=self.observation_num_classes)
                        n_obs_tensor = get_tensor(n_obs_tensor, self.observation_size)  # Shape should be (N, observation_size)
                    else:
                        n_obs_tensor = torch.tensor(n_obs_t[agent_type], dtype=torch.int64)
                        n_obs_tensor = get_tensor(n_obs_tensor, self.observation_size)  # Shape should be (N, observation_size)
                    n_obs_t_tensor.append(n_obs_tensor)
                tensors['n_obs'] = n_obs_t_tensor
            if fea_t is not None:
                fea_t_tensor = []
                for agent_type in range(self.num_types):
                    fea_tensor = get_tensor(fea_t[agent_type], self.feature_size)    # Shape should be (N, feature_size)
                    fea_t_tensor.append(fea_tensor)
                tensors['fea'] = fea_t_tensor
            if beta_t is not None:
                beta_t_tensor = []
                for agent_type in range(self.num_types):
                    beta_tensor = torch.tensor(beta_t[agent_type], dtype=torch.float)  # Shape should be (N,)
                    beta_t_tensor.append(beta_tensor)
                tensors['beta'] = beta_t_tensor

        return tensors

    def calculate_losses(self, tensors):
        """
        Calculate losses given settings (self.args.mode_ac, self.args.mode_psi).

        Parameters
        ----------
        tensors: dict

        Returns
        -------
        actor_loss: list
        psi_loss: list
        critic_loss: list
        """
        actor_loss, psi_loss, critic_loss = [None for _ in range(3)]
        if self.args.mode_ac:
            actor_loss = self.calculate_actor_loss(tensors)
        if self.args.mode_psi:
            psi_loss = self.calculate_psi_loss(tensors)
        else:
            critic_loss = self.calculate_critic_loss(tensors)

        return actor_loss, psi_loss, critic_loss

    def calculate_actor_loss(self, tensors):
        """
        Calculate actor loss.

        Parameters
        ----------
        tensors: dict

        Returns
        -------
        actor_loss: list
        """
        actor_loss = []
        obs: list = tensors['obs']  # [torch.Tensor: (N[agent_type], observation_size) for agent_type in range(self.num_types)]
        act: list = tensors['act']  # [torch.Tensor: (N[agent_type],) for agent_type in range(self.num_types)]
        m_act: list = tensors['m_act']  # [[torch.Tensor: (N[agent_type], action_size[action_type]) for action_type in range(self.num_types)] for agent_type in range(self.num_types)]
        for agent_type in range(self.num_types):
            with torch.no_grad():
                # Get q values from the psi/critic target network.
                if self.args.mode_psi:
                    psi_target = self.psi_target[agent_type](obs[agent_type], m_act[agent_type])  # Shape: (N[agent_type], action_size[agent_type], feature_size)
                    q_target = torch.tensordot(psi_target, self.w, dims=([2], [0]))  # Shape: (N[agent_type], action_size[agent_type])
                else:
                    q_target = self.critic_target[agent_type](obs[agent_type], m_act[agent_type])  # Shape: (N[agent_type], action_size[agent_type])
                # Get action probabilities from the actor target network.
                #act_probs_target = self.actor_target[agent_type](obs[agent_type])
                #v_target = torch.sum(q_target * act_probs_target, dim=-1).view(-1, 1)
                act_means_target, act_stds_target = self.actor_target[agent_type](obs[agent_type])
                act_dist_target = distributions.Normal(act_means_target, act_stds_target)
                sampled_actions_target = act_dist_target.sample()  # Sample actions from the target distribution
                act_probs_target = act_dist_target.log_prob(sampled_actions_target).exp()  # Convert log probs to probs

                # Value estimation
                v_target = torch.sum(q_target * act_probs_target, dim=-1).view(-1, 1)

            # Get action probabilities from the actor network.
            #act_probs = self.actor[agent_type](obs[agent_type])
            #act_dist = distributions.Categorical(act_probs)
            mu, std = self.actor[agent_type](obs[agent_type])
            act_dist = distributions.Normal(mu, std)
            
            # Log probability of the taken actions
            log_probs = act_dist.log_prob(act[agent_type]).sum(dim=-1, keepdim=True)

            # Get q_target for the taken actions
            #q_target = q_target.gather(1, act[agent_type].long())

            # Get actor loss using values and probabilities
            #print("actor loss1", q_target, q_target.shape)
            #q_target = q_target[torch.arange(q_target.size(0)), act[agent_type]].view(-1, 1)
            #loss = - (q_target - v_target) * act_dist.log_prob(act[agent_type]).view(-1, 1)
            loss = - (q_target - v_target) * act_dist.log_prob(act[agent_type])
            loss = torch.mean(loss)
            actor_loss.append(loss)

        return actor_loss

    def calculate_psi_loss(self, tensors):
        """
        Calculate actor loss.

        Parameters
        ----------
        tensors: dict

        Returns
        -------
        psi_loss: list
        """
        psi_loss = []
        obs: list = tensors['obs']
        act: list = tensors['act']
        m_act: list = tensors['m_act']
        n_obs: list = tensors['n_obs']
        fea: list = tensors['fea']
        beta: list = tensors['beta']

        for agent_type in range(self.num_types):
            with torch.no_grad():
                # Get psi values of next observations from the psi target network.
                psi_target_n = self.psi_target[agent_type](n_obs[agent_type], m_act[agent_type])  # (N[agent_type], action_size[agent_type], feature_size)

                # Get action probabilities from the actor target network or the Boltzmann policy. (N[agent_type], action_size[agent_type])
                if self.args.mode_ac:
                    act_probs_target_n = self.actor_target[agent_type](n_obs[agent_type])
                else:
                    q_target_n = torch.tensordot(psi_target_n, self.w, dims=([2], [0]))
                    act_probs_target_n = self.get_boltzmann_policy(q_target_n, beta[agent_type])

                # Get expected psi using psi and action probabilities.
                expected_psi_target_n = torch.bmm(act_probs_target_n.unsqueeze(1), psi_target_n)  # (N[agent_type], 1, feature_size)
                expected_psi_target_n = expected_psi_target_n.view(-1, self.feature_size)  # (N[agent_type], feature_size)

            # Get psi loss
            psi = self.psi[agent_type](obs[agent_type], m_act[agent_type])  # (N[agent_type], action_size[agent_type], feature_size)
            psi = psi[torch.arange(psi.size(0)), act[agent_type]]  # (N[agent_type], feature_size)
            loss = (fea[agent_type] + self.args.gamma * expected_psi_target_n - psi) ** 2  # (N[agent_type], feature_size)
            loss = torch.mean(loss, dim=0)  # (feature_size,)
            psi_loss.append(loss)

        return psi_loss

    def calculate_critic_loss(self, tensors):
        """
        Calculate critic loss.

        Parameters
        ----------
        tensors: dict

        Returns
        -------
        critic_loss: list
        """
        critic_loss = []
        obs: list = tensors['obs']
        act: list = tensors['act']
        rew: list = tensors['rew']
        m_act: list = tensors['m_act']
        n_obs: list = tensors['n_obs']
        beta: list = tensors['beta']

        for agent_type in range(self.num_types):
            with torch.no_grad():
                # Get q values of next observations from the q target network.
                q_target_n = self.critic_target[agent_type](n_obs[agent_type], m_act[agent_type])

                # Get action probabilities from the actor target network or the Boltzmann policy.
                if self.args.mode_ac:
                    means_target, stds_target = self.actor_target[agent_type](n_obs[agent_type])
                    act_dist_target_n = distributions.Normal(means_target, stds_target)
                    sampled_actions_target = act_dist_target_n.sample()
                    log_probs_target_n = act_dist_target_n.log_prob(sampled_actions_target).sum(dim=-1, keepdim=True)
                    act_probs_target_n = act_dist_target_n.log_prob(act[agent_type]).exp()

                else:
                    act_probs_target_n = self.get_boltzmann_policy(q_target_n, beta[agent_type])

                # Get v values using q values and action probabilities.
                #print("q_target_n", q_target_n.shape, act_probs_target_n.shape)
                # Compute the value target using sampled actions and log probabilities
                #v_target_n = torch.exp(log_probs_target_n) * q_target_n
                #v_target_n = torch.sum(q_target_n * act_probs_target_n, dim=-1).view(-1, 1)
                v_target_n = torch.sum(q_target_n * torch.exp(log_probs_target_n), dim=-1, keepdim=True)


            # Get critic loss using values.
            q = self.critic[agent_type](obs[agent_type], m_act[agent_type])

            #q = q[torch.arange(q.size(0)), act[agent_type]].view(-1, 1)

            loss = (rew[agent_type] + self.args.gamma * v_target_n - q) ** 2
            loss = torch.mean(loss)
            critic_loss.append(loss)

        return critic_loss

    def update_networks(self, samples: list):
        obs_t, act_t, rew_t, m_act_t, n_obs_t, fea_t, beta_t = self.preprocess(samples)
        tensors = self.to_tensors(obs_t=obs_t,
                                  act_t=act_t,
                                  rew_t=rew_t,
                                  m_act_t=m_act_t,
                                  n_obs_t=n_obs_t,
                                  fea_t=fea_t,
                                  beta_t=beta_t)
        actor_loss, psi_loss, critic_loss = self.calculate_losses(tensors)
        for agent_type in range(self.num_types):
            if self.args.mode_ac:  # actor
                self.actor_opt[agent_type].zero_grad()
                actor_loss[agent_type].backward()
                self.actor_opt[agent_type].step()
                self.actor_skd[agent_type].step() if self.args.mode_lr_decay else None
            if self.args.mode_psi:  # psi
                self.psi_opt[agent_type].zero_grad()
                psi_loss[agent_type].backward(torch.ones(self.feature_size))
                self.psi_opt[agent_type].step()
                self.psi_skd[agent_type].step() if self.args.mode_lr_decay else None
            else:  # critic
                self.critic_opt[agent_type].zero_grad()
                critic_loss[agent_type].backward()
                self.critic_opt[agent_type].step()
                self.critic_skd[agent_type].step() if self.args.mode_lr_decay else None

    def update_target_network(self, network: list, target_network: list):
        for agent_type in range(self.num_types):
            for param, target_param in zip(network[agent_type].parameters(), target_network[agent_type].parameters()):
                target_param.data.copy_(param.data * self.args.tau + target_param.data * (1.0 - self.args.tau))

    def update_target_networks(self):
        if self.args.mode_ac:
            self.update_target_network(self.actor, self.actor_target)
        if self.args.mode_psi:
            self.update_target_network(self.psi, self.psi_target)
        else:
            self.update_target_network(self.critic, self.critic_target)