import pylab as pl
import sys
import argparse
import random
import torch

sys.path.insert(0, './Tasks')
from Networks import *


def get_egreedy_action(observation, q_model):
    q_preds = q_model.forward(observation).detach()
    action = q_preds.argmax()
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)  # linear or deep
    parser.add_argument('--param', type=str, default='params_10')
    parser.add_argument('--task', type=int, default=1)  # task ID: 1,2 or 3
    args = parser.parse_args()

    if args.task == 3:
        from Tasks.t3 import task3
        from Tasks.t3 import state_size, num_action
        env = task3(render=True)
    elif args.task == 2:
        from t2 import task2
        from t2 import state_size, num_action
        env = task2(render=True)
    elif args.task == 1:
        from Tasks.t1 import task1
        from Tasks.t1 import state_size, num_action
        env = task1(render=True)

    # initialize agent VFA network that will be used for visualisation
    if args.task == 3 and args.network == 'deep':
        value_network = CNN(3)
    elif args.network == 'linear':
        value_network = Linear(state_size, num_action, 1)
    elif args.network == 'deep':
        value_network = Deep(state_size, num_action)

    # retrieve network parameters from path specified by the user
    if args.network is not None:
        value_network.load_state_dict(torch.load(args.param))

    # below we run 10,000 time-steps of the task for visualisation
    steps = 0
    avg_reward = 0
    state = torch.Tensor([env.state])
    while steps < 10000:
        steps += 1

        if args.network is None:  # if no network is given, random actions are selected
            action = random.randint(0, num_action - 1)
        else:
            action = get_egreedy_action(state, value_network)

        new_observation, reward = env.step(action)
        next_state = torch.Tensor([new_observation])
        avg_reward += (1 / steps) * (reward - avg_reward)
        state = next_state


if __name__ == "__main__":
    main()
