#!/usr/bin/env python3
# encoding utf-8

import multiprocessing as mp
import numpy as np
import SharedAdam 
import Worker
import random
import Networks
import argparse
import torch
import os
import time
import player
import sys
sys.path.insert(0, './Tasks')

# Added in March 2021 to mute depreciation warnings. May be worth updating dependencies
import warnings
warnings.filterwarnings("ignore")


def setup_log(lr,args,beta=None,df=None):
	path='./logs/'+str(args.algorithm)+'/'+str(args.reward)+'/'+str(args.network)
	print('logging in:',path)
	runs=0

	while os.path.exists(path+'/'+str(runs)):
		runs+=1
	path+='/'+str(runs)

	if not os.path.exists(path):
		os.makedirs(path)

	f=open(path+'/hyper_param','w+')
	if beta == None:
		f.write("lr=%.5f\ndf=%.5f\n" % (lr,df))
	else:
		f.write("lr=%.5f\nbeta=%.5f\n" % (lr,beta))
	f.flush()
	f.close()
	return path

def get_scores(task,path,value_network):
	TOTAL_STEPS=50000
	SAMPLES=5

	param_files=[]
	for i in os.listdir(path):
	    if 'params' in i:
	        param_files.append(i.split('_')[1])

	for p in param_files:
		value_network.load_state_dict(torch.load(path+'/params_'+str(p)))
		score=[]
		for i in range(SAMPLES):
			s=player.train(value_network,TOTAL_STEPS,task,path)
			print(s)
			score.append(s)
		f=open(path+'/hyper_param','a')
		print("%s_score=%.5f\n" % (p,np.mean(score)))
		f.write("%s_score=%.5f\n" % (p,np.mean(score)))
		f.flush()
	f.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_processes', type=int, default=16)
	parser.add_argument('--steps', type=float, default=16) #total steps across all processes (in millions)
	parser.add_argument('--algorithm', type=str, default='Q') # Q or SARSA
	parser.add_argument('--network', type=str, default='linear') #linear or deep (deep is a CNN for task 3. See Networks.py)
	parser.add_argument('--reward', type=str, default='discounted') #discounted or average
	parser.add_argument('--task', type=int, default=1) #task ID: 1,2 or 3
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--beta', type=float, default=0.001)
	parser.add_argument('--df', type=float, default=0.99)
	args=parser.parse_args()

	EPSILON_MIN=[0.1]*4 + [0.01]*3 + [0.5]*3 #Distribution for the minimum value of epsilon 
	TOTAL_STEPS=args.steps * 10**6 
	DISCOUNT_FACTOR=args.df
	LR=args.lr
	BETA=args.beta

	if args.reward=='discounted':
		path=setup_log(lr=LR,args=args,df=DISCOUNT_FACTOR) 
	elif args.reward=='average':
		path=setup_log(lr=LR,args=args,beta=BETA) 

	if args.task==1:
		from t1 import state_size, num_action
	elif args.task==2:
		from t1 import state_size, num_action
	elif args.task==3:
		from t1 import state_size, num_action


	if args.network=='linear':
			value_network= Networks.Linear(state_size,num_action)
			target_value_network= Networks.Linear(state_size,num_action)
	elif args.network=='deep':
		if args.task==3:
			value_network= Networks.CNN(num_action)
			target_value_network= Networks.CNN(num_action)
		else:
			value_network= Networks.Deep(state_size,num_action)
			target_value_network= Networks.Deep(state_size,num_action)


	Worker.hard_update(target_value_network,value_network)
	Worker.saveModelNetwork(target_value_network, path+'/params_0')

	optimizer=SharedAdam.SharedAdam(value_network.parameters(), lr=LR)
	counter = mp.Value('i', 0)
	lock = mp.Lock()

	target_value_network.share_memory()
	value_network.share_memory()
	optimizer.share_memory()

	processes=[]

	start = time.time()

	for idx in range(0, args.num_processes):
		min_e=random.choice(EPSILON_MIN)
		trainingArgs = (idx, value_network, target_value_network,\
		optimizer, lock, counter,TOTAL_STEPS/args.num_processes,min_e,args.task,num_action,\
		path,args.algorithm,args.reward=='discounted',BETA,DISCOUNT_FACTOR)

		p = mp.Process(target=Worker.train, args=trainingArgs)
		p.start()
		processes.append(p)
	for p in processes:
		p.join()

	end = time.time()
	sec=end - start
	final_score=get_scores(args.task,path,value_network) 

	f=open(path+'/hyper_param','a')
	f.write("seconds=%.1f\n" % sec)
	f.flush()
	f.close()





if __name__ == "__main__" :
	main()

