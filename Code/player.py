import torch

from Networks import *

import sys
sys.path.insert(0, './Tasks')
from Tasks.t1 import *
from Tasks.t2 import *
from Tasks.t3 import *


# mai train method which contains training loop
def train(value_network, max_steps, task, r_seed, sample):
	if task == 1:
		Env = task1(r_seed+sample+1)
	if task == 2:
		Env = task2(r_seed+sample+1)
	if task == 3:
		Env = task3(r_seed+sample+1)

	steps = 0
	avg_reward = 0
	state= torch.Tensor([Env.state])
	while steps < max_steps:
		steps += 1

		# Take action a with ε-greedy policy based on Q(s, a; θ)
		action= get_egreedy_action(state, value_network)

		# Receive new state s′ and reward r
		new_observation, reward = Env.step(action)
		next_state = torch.Tensor([new_observation])
		avg_reward += (1/steps)*(reward-avg_reward)

		state = next_state
		
	return avg_reward


# computes and returns greedy action given the model and an observation
def get_egreedy_action(observation, q_model):
	q_preds = q_model.forward(observation).detach()
	action = q_preds.argmax()

	return action
