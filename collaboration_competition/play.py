"""
Script that can be used to play the Unity Tennis environment with a pretrained agent.
"""

import argparse
from typing import List

import numpy as np
import torch
from unityagents import UnityEnvironment

from collaboration_competition.agent import Agent
from collaboration_competition.utils import get_actions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path',
                        dest='env_path',
                        help='Unity environment local path')
    parser.add_argument('--experiment-id',
                        dest='experiment_id',
                        help='Experiment id')
    parser.add_argument('--weights-path',
                        dest='weights_path',
                        help='Path to store the agents NN weights',
                        default='./weights')
    args = parser.parse_args()
    return args


def play_with_agents(env: UnityEnvironment, brain_nm: str, agents: List[Agent]):
    """
    Uses pretrained agents to play a Unity environment.

    :param env: Unity environment
    :param brain_nm: brain name
    :param agents: list of Agent instances
    """
    env_info = env.reset(train_mode=False)[brain_nm]  # reset the environment
    states = env_info.vector_observations  # get the current state for each agent
    score = np.zeros(len(agents))  # initialize the scores for each agent
    while True:
        actions = get_actions(agents, states, add_noise=False)  # select action for each agent
        env_info = env.step(actions)[brain_nm]  # send actions to the environment
        next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
        score += rewards  # update the score
        states = next_states  # roll over the states to next time step
        if any(dones):  # exit loop if episode finished
            break

    print("Score: {}".format(np.mean(score)))


def main():
    args = parse_args()
    # Create env
    unity_env = UnityEnvironment(file_name=args.env_path, seed=10)
    # Get the default brain
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    # Init agent
    environment_info = unity_env.reset(train_mode=False)[brain_name]
    # number of agents
    num_agents = len(environment_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # States
    states = environment_info.vector_observations
    state_size = states.shape[1]
    # Agents
    agents = [
        Agent(state_size=state_size, action_size=action_size)
        for _ in range(num_agents)
    ]
    # Load weights
    weights_path = f'{args.weights_path}/{args.experiment_id}'
    for i, agent in enumerate(agents):
        agent.actor_local.load_state_dict(torch.load(f'{weights_path}/checkpoint_actor_{i}.pth'))
        agent.critic_local.load_state_dict(torch.load(f'{weights_path}/checkpoint_critic_{i}.pth'))

    # Play!
    play_with_agents(unity_env, brain_name, agents)


if __name__ == '__main__':
    main()
