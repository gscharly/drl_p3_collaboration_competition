"""
Script that can be used to train 2 agents to solve the Unity Tennis environment using the MADDPG algorithm
"""

import argparse
from collections import deque
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from collaboration_competition.agent import Agent
import collaboration_competition.conf as conf
from collaboration_competition.utils import get_actions

MIN_SCORE_SOLVED = 0.5


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


def train(env: UnityEnvironment, brain_nm: str, agents: List[Agent],
          weights_path: str, n_episodes: int = 1000,
          n_episodes_score: int = 100) -> Tuple[List, List, List]:
    """
    Trains a list of agents to solve the Unity env.

    :param env: Unity env
    :param brain_nm: Unit env name
    :param agents: list of Agent instances to train
    :param weights_path: path to store each agent's weights
    :param n_episodes: number of episodes to train the agents
    :param n_episodes_score: number of episodes to compute the average score that will be used
    to evaluate the agents
    :return: scores, average scores and trained agents
    """

    scores_deque = deque(maxlen=n_episodes_score)
    scores = []
    avg_scores = []
    solved_env = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_nm]
        states = env_info.vector_observations

        for agent in agents:
            agent.reset()

        score = np.zeros(len(agents))
        while True:
            # Each agent selects actions based on its actor network and states
            actions = get_actions(agents=agents, states=states, add_noise=True)
            env_info = env.step(actions)[brain_nm]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done

            for i, agent in enumerate(agents):
                agent.step(states, actions, rewards[i], next_states, dones[i])

            # Update states & rewards
            states = next_states
            score += rewards

            if np.any(dones):
                break

        max_score_episode = np.max(score)
        scores_deque.append(max_score_episode)
        scores.append(max_score_episode)
        avg_scores.append(np.mean(scores_deque))
        print('\rEpisode {}\t Score (max over agents): {:.2f}\tAverage Score: {:.2f}'.format(
            i_episode, max_score_episode, np.mean(scores_deque)))

        if not solved_env and i_episode >= n_episodes_score and np.mean(scores_deque) > MIN_SCORE_SOLVED:
            print(f'Environment has been solved in {i_episode} episodes')
            solved_env = 1

        if i_episode % n_episodes_score == 0:
            for i, ddpg_agent in enumerate(agents):
                torch.save(ddpg_agent.actor_local.state_dict(), f'{weights_path}/checkpoint_actor_{i}.pth')
                torch.save(ddpg_agent.critic_local.state_dict(), f'{weights_path}/checkpoint_critic_{i}.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores, avg_scores, agents


def plot_scores(scores: List, avg_scores: List, weights_path: str):
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(avg_scores)), avg_scores, color='orange')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'{weights_path}/scores.png')
    plt.show()


def main():
    args = parse_args()
    print(args)
    # Create env
    unity_env = UnityEnvironment(file_name=args.env_path, no_graphics=True)
    # Get the default brain
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    # Init env
    environment_info = unity_env.reset(train_mode=False)[brain_name]
    # number of agents
    num_agents = len(environment_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space
    states = environment_info.vector_observations
    state_size = states.shape[1]
    # Init agents
    agents = [
        Agent(state_size=state_size, action_size=action_size, random_seed=conf.RANDOM_SEED)
        for _ in range(num_agents)
    ]

    # Train agents
    weights_path = f'{args.weights_path}/{args.experiment_id}'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    result_scores, avg_scores, trained_agents = train(
        env=unity_env, brain_nm=brain_name, agents=agents,
        weights_path=weights_path, n_episodes=conf.EPISODES
    )
    plot_scores(result_scores, avg_scores, weights_path)

    # Store scores
    with open(f'{weights_path}/scores.pkl', 'wb') as f:
        pickle.dump(result_scores, f)


if __name__ == '__main__':
    main()
