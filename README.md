# Udacity Deep Reinforcement Learning course - Multi Agent RL - Collaboration and Competition

This repository contains code that train an agent to solve the environment proposed in the Multi Agent Reinforcement
Learning section  of the Udacity Deep Reinforcement Learning (DRL) course.

# Environment

The environment has 2 agents playing tennis. Each agent has a set of actions and states. If an agent hits the ball over
the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds,
it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The task is episodic, and
the environment will be considered solved when the average over 100 episodes of the maximum reward of the agents hits
+0.5.

Both the action and the state space are continuous. The state space consists of 8 variables corresponding to the
position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are
available, corresponding to movement toward (or away from) the net, and jumping.

# Getting started

## Unity environments

Unity doesn't need to be installed since the environment is already available. The environments can be downloaded from
the following links:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Python dependencies
The project uses Python 3.6 and relies on the [Udacity Value Based Methods repository](https://github.com/udacity/Value-based-methods#dependencies).
This repository should be cloned, and the instructions on the README should be followed to install the necessary
dependencies.

# Instructions
The repository contains 2 scripts under the collaboration_competition package: train.py and play.py.

## Train
The script train.py can be used to train the agents. The environment has been solved using the Multiple Agent Deep
Deterministic Policy Gradient (MADDPG) algorithm. More details can be found in ipynb/report.ipynb.

The script accepts the following arguments:
- env-path: path pointing to the Unity Tennis environment
- weights-path: path where the agents' NN weights will be stored
- experiment-id: path inside weights-path where weights and plots for an experiment will be stored

The algorithm hyperparameters are stored in conf.py to simplify experimentation.

Example:

```
python train.py --env-path /deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64
--weights-path /repos/deep-reinforcement-learning/p3_collab-compet/weights
--experiment-id maddpg_5
```

## Play
Agents can be used to play! To do so, the play.py script can be used, providing the Unity environment and
the agents' weights paths:

```
python play.py --env-path /home/carlos/cursos/udacity_rl_2023/repos/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_env2/Reacher.x86_64
--weights-path /home/carlos/cursos/udacity_rl_2023/projects/drl_p2_continous_control/weights
```
