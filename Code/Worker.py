import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import *
from torch.autograd import Variable
import time
import sys
sys.path.insert(0, './Tasks')
from t1 import *
from t2 import *
from t3 import *
import random
import numpy as np
from torch import optim


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def train(idx, value_network, target_value_network, optimizer, lock, counter, max_steps, epsilon_min, task, num_action, path, alg, discounted, beta, discount_factor):
	DF=discount_factor
	I_update=5
	I_target=40000
	BETA=beta
	SAVE_EVERY=10**6
	crit = nn.MSELoss()

	if task==1:
		Env = task1()
	if task==2:
		Env = task2()
	if task==3:
		Env = task3()

	f=open(path+'/log'+str(idx),'w+')
	f.write("reward, avg\n")
	f.flush()


	t=0
	steps=0
	avg_reward=0
	state= torch.Tensor([Env.state])
	while steps< max_steps:
		steps+=1
		t+=1

		epsilon=set_epsilon(counter.value, epsilon_min)
		# Take action a with ε-greedy policy based on Q(s, a; θ)
		action= get_egreedy_action(state, value_network, epsilon,num_action)


		# Receive new state s′ and reward r
		newObservation, reward= Env.step(action)
		next_state= torch.Tensor([newObservation])

		if (alg=='SARSA' and steps>1) or (alg=='Q'):

			if alg=='SARSA':
				Q=computePrediction(past_state, past_action, value_network)
				Ut=computeTargets_SARSA(past_reward, state, action, DF, target_value_network, discounted,avg_reward)
			if alg=='Q':
				Q=computePrediction(state, action, value_network)
				Ut=computeTargets_Q(reward, next_state, DF, target_value_network, discounted,avg_reward)

			# Accumulate gradients wrt θ
			loss=crit(Q, Ut)			
			loss.backward()
			avg_reward+= float(BETA*(Ut-Q))


		with lock:
			counter.value += 1

			# Update the target network θ− ← θ
			if counter.value%I_target==0:
				hard_update(target_value_network, value_network)

			if counter.value%SAVE_EVERY==0:
				saveModelNetwork(target_value_network, path+'/params_'+str(int(counter.value/SAVE_EVERY)))

			if counter.value%100==0:
				logProgress(counter,SAVE_EVERY, 50)
			
	
			# Perform asynchronous update of θ using dθ. Clear gradients dθ ← 0.
			if (t%I_update==0):
				optimizer.step()
				optimizer.zero_grad()
		
			f.write("%.3f, %.2f\n" % (reward,avg_reward))
			f.flush() 

		past_action=action
		past_reward=reward
		past_state =state
		state = next_state

	saveModelNetwork(value_network, path+'/params_last')
	return 

def set_epsilon(frames, min_e):
	e= 1- frames/(4000000/(1-min_e))
	return max(min_e,e)

def get_egreedy_action(observation, Qmodel, epsilon,num_actions):
	random_action= random.random()<epsilon

	if random_action:
		action= random.randint(0,num_actions-1)
	else:
		Qpreds=Qmodel.forward(observation).detach()
		action = Qpreds.argmax()
	return action

def computeTargets_SARSA(reward, observation, action, discountFactor, targetNetwork,discounted, avg_reward):
    Qpred=targetNetwork.forward(observation).detach()
    if discounted:
    	target= reward + discountFactor * Qpred[0][action]
    else:
    	target= reward - avg_reward + Qpred[0][action]

    return target

def computeTargets_Q(reward, nextObservation, discountFactor, targetNetwork,discounted, avg_reward):
    Qpred=targetNetwork.forward(nextObservation).detach()
    if discounted:
    	target= reward + discountFactor * Qpred.max()
    else:
    	target= reward - avg_reward + Qpred.max()
    return target

def computePrediction(state, action, valueNetwork):
    out= valueNetwork.forward(state)
    return out[0][action]
	
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)


def logProgress(counter,saveIternval, barLength):
	save_iteration = int(counter.value/saveIternval)
	percent = counter.value%saveIternval * 100 / saveIternval
	makers   = '|' * int(percent/100 * barLength - 1)
	spaces  = ' ' * (barLength - len(makers))
	string = 'Parameter %d progress:\t |%s%s| %d %%' % (save_iteration, makers, spaces, percent)
	print(string.ljust(100), end='\r')
	




