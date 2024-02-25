from typing import List

import numpy as np

from collaboration_competition.agent import Agent


def get_actions(agents: List[Agent], states: np.ndarray, add_noise: bool) -> np.ndarray:
    """
    Each agent receives a state and returns the actions based on the agent's policy. The output
    contains the concatenated actions for all agents.

    :param agents: list of Agent instances
    :param states: state of each agent
    :param add_noise: whether to add noise to an agent's action
    :return: all agents actions
    """
    actions = [agent.act(state, add_noise) for agent, state in zip(agents, states)]
    return np.reshape(actions, (len(actions), actions[0].shape[-1]))
