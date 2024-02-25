import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/model.py

def hidden_init(layer):
    import numpy as np
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):

    def __init__(self, state_size: int, action_size: int, random_seed: int):
        """
        Actor (Policy) Model.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param random_seed: random seed
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Build an actor (policy) network that maps states -> actions.

        :param states: states vector
        :return: actions vector
        """
        x = self.bn0(states)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int, random_seed: int,
                 num_agents: int = 2):
        """
        Critic (Q-value) Model. The Critic network in the MADDPG algorithm receives states and actions from all
        agents.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param random_seed: random seed
        :param num_agents: number of agents in a Multi Agent environment
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        self.bn0 = nn.BatchNorm1d(state_size * num_agents)
        self.fc1 = nn.Linear(state_size * num_agents, 256)
        self.fc2 = nn.Linear(256 + (action_size*num_agents), 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Build a critic network that maps states -> Q values.

        :param states: states vector
        :param actions: actions vector
        :return: Q value for each pair state-action
        """
        states = self.bn0(states)
        x_state = F.leaky_relu(self.fc1(states))
        x = torch.cat((x_state, actions), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
