import pylab as pl
import sys
import argparse
import random
sys.path.insert(0, './Tasks')
from Networks import *


def get_egreedy_action(observation, Qmodel):
    Qpreds=  Qmodel.forward(observation).detach()
    action=  Qpreds.argmax()
    return	action

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--network', type=str) #linear or deep
	parser.add_argument('--param', type=str, default='params_10')
	parser.add_argument('--task', type=int, default=1) #task ID: 1,2 or 3
	args=parser.parse_args()

	if args.task==3:
		from t3 import task3
		from task3 import state_size, num_action
		Env=task3(True)

	elif args.task==2:
		from t2 import task2
		from t2 import state_size, num_action
		Env=task2(True)

	elif args.task==1:
		from t1 import task1
		from t1 import state_size, num_action
		Env=task1(True)

	
	if args.task==3 and args.network=='deep':
		value_network= CNN(3)
	elif args.network=='linear':
		value_network= Linear(state_size,num_action)
	elif args.network=='deep':
		value_network= Deep(state_size,num_action)

	if args.network != None:
		value_network.load_state_dict(torch.load(args.param))

	steps=0
	avg_reward=0
	state=torch.Tensor([Env.state])
	while steps< 10000:
	    steps+=1

	    if args.network==None:
	    	action=random.randint(0,num_action-1)
	    else:
	    	action=get_egreedy_action(state,value_network)

	    newObservation, reward=  Env.step(action)
	    next_state=  torch.Tensor([newObservation])
	    avg_reward+=(1/steps)*(reward-avg_reward)
	    state  =  next_state

if __name__ == "__main__" :
	main()