import numpy as np
import random
import copy
from collections import namedtuple, deque
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

import collaboration_competition.conf as conf
from collaboration_competition.model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, state_size: int, action_size: int, random_seed=conf.RANDOM_SEED):
        """
        Interacts with and learns from the environment.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=conf.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=conf.LR_CRITIC)

        # Noise process
        self.noise = OUNoise((1, action_size), random_seed)
        self.noise_scalar_decay = conf.NOISE_DECAY
        # Replay memory
        self.memory = ReplayBuffer(conf.BUFFER_SIZE, conf.BATCH_SIZE, random_seed)

        self.timestep = 0

    def step(self, states: np.ndarray, actions: np.ndarray, reward: np.ndarray,
             next_states: np.ndarray, done: np.ndarray):
        """
        Saves experience in replay memory, and use random sample from buffer to learn. An agent
        will only learn every conf.LEARN_EVERY steps, and will perform conf.LEARN_NUM learning iterations
        in each step.

        :param states: state vectors of all agents
        :param actions: action vectors of all agents
        :param reward: reward of agent
        :param next_states: next state vectors of all agents
        :param done: integer indicating whether the task has finished
        """
        self.timestep += 1
        # Save experience
        self.memory.add(states, actions, reward, next_states, done)
        # Learn, if enough samples are available in memory and at learning interval settings
        if len(self.memory) > conf.BATCH_SIZE and self.timestep % conf.LEARN_EVERY == 0:
            # Multiple learning iterations in the same step
            for _ in range(conf.LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, conf.GAMMA)

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Chooses an action given a state based on the agent's policy

        :param state: env state
        :param add_noise: whether to add noise to the action to encourage exploration
        :return: action
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (self.noise_scalar_decay * self.noise.sample())
            self.noise_scalar_decay *= self.noise_scalar_decay
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences: Tuple, gamma: float):
        """
        Updates policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Critic will be updated using the actions and states from ALL agents

        :param experiences: tuple of (s, a, r, s', done) Torch tensors
        :param gamma: discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        n_agents = states.shape[1]

        # ---------------------------- update critic ---------------------------- #
        # Get next actions using agent's policy for all agents states
        next_actions = [self.actor_target(next_states[:, i]) for i in range(n_agents)]
        next_actions = torch.cat(next_actions, dim=1)

        # Reshape next_states from (, num_agents, state_size) to (, num_agents * state_size)
        next_states = next_states.view(-1, n_agents * next_states.shape[-1])
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, next_actions)

        Q_targets = rewards.view(-1, 1) + (gamma * Q_targets_next * (1 - dones.view(-1, 1)))
        # States and actions must be reshaped since critic uses all agents info
        # States: from (, num_agents, state_size) to (, num_agents * state_size)
        # Actions: from (, num_agents, action_size) to (, num_agents * action_size)
        all_states = states.view(-1, n_agents * states.shape[-1])
        all_actions = actions.view(-1, n_agents * actions.shape[-1])
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Predict actions using agent's policy for all agents states
        actions_pred = [self.actor_local(states[:, i]) for i in range(n_agents)]
        actions_pred = torch.cat(actions_pred, dim=1)

        # Compute actor loss
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, conf.TAU)
        self.soft_update(self.actor_local, self.actor_target, conf.TAU)

    @staticmethod
    def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:

    def __init__(self, size: Tuple, seed: int, mu: float = 0.0,
                 theta: float = conf.OU_THETA, sigma: float = conf.OU_SIGMA):
        """
        Ornstein-Uhlenbeck process. Initialize parameters and noise process.

        :param size: size of the resulting vector (in case there are more than 1 agent)
        :param seed: random seed
        :param mu: ong-running mean
        :param theta: speed of mean reversion
        :param sigma: volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """
        Fixed-size buffer to store experience tuples. Initialize a ReplayBuffer object.

        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param seed: random seed
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, states: np.ndarray, actions: np.ndarray, reward: np.ndarray,
            next_states: np.ndarray, done: np.ndarray):
        """Add a new experience to memory."""
        e = self.experience(states, actions, reward, next_states, done)
        self.memory.append(e)

    def sample(self) -> Tuple:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
