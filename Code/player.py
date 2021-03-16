import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import *
from torch.autograd import Variable
import sys
sys.path.insert(0, './Tasks')
from t1 import *
from t2 import *
from t3 import *


import random
import numpy as np
from torch import optim


def train(value_network,max_steps,task,path):
	if task==1:
		Env = task1()
	if task==2:
		Env = task2()
	if task==3:
		Env = task3()


	steps=0
	avg_reward=0
	state= torch.Tensor([Env.state])
	while steps< max_steps:
		steps+=1

		# Take action a with ε-greedy policy based on Q(s, a; θ)
		action= get_egreedy_action(state, value_network)

		# Receive new state s′ and reward r
		newObservation, reward= Env.step(action)
		next_state= torch.Tensor([newObservation])
		avg_reward+= (1/steps)*(reward-avg_reward)

		state = next_state
		
	return avg_reward

def get_egreedy_action(observation, Qmodel):
	Qpreds=Qmodel.forward(observation).detach()
	action = Qpreds.argmax()
	return action

	




