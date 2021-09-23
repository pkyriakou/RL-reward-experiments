import torch
import torch.nn as nn
import random

from Networks import *

import sys
sys.path.insert(0, './Tasks')
from Tasks.t1 import *
from Tasks.t2 import *
from Tasks.t3 import *


# method that copies the parameters of the value network to the target value network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# main training method which contains the training loop
def train(idx, value_network, target_value_network, optimizer, lock, counter, max_steps, epsilon_min, task,
          num_action, path, alg, discounted, beta, discount_factor, r_seed):

    DF = discount_factor
    I_update = 5
    I_target = 40000
    BETA = beta
    SAVE_EVERY = 10**5
    crit = nn.MSELoss()

    if task == 1:
        Env = task1(r_seed)
    if task == 2:
        Env = task2(r_seed)
    if task == 3:
        Env = task3(r_seed)

    f = open(path+'/log'+str(idx), 'w+')
    f.write("reward, avg\n")
    f.flush()

    t = 0
    steps = 0
    avg_reward = 0
    state = torch.Tensor([Env.state])

    # training loop
    while steps < max_steps:
        steps += 1
        t += 1

        epsilon = set_epsilon(counter.value, epsilon_min)
        # Take action a with ε-greedy policy based on Q(s, a; θ)
        action = get_egreedy_action(state, value_network, epsilon, num_action)

        # Receive new state s′ and reward r
        newObservation, reward = Env.step(action)
        next_state= torch.Tensor([newObservation])

        if (alg == 'SARSA' and steps > 1) or (alg == 'Q') or (alg == 'doubleQ'):

            if alg == 'SARSA':
                Q = computePrediction(past_state, past_action, value_network)
                Ut = computeTargets_SARSA(past_reward, state, action, DF, target_value_network, discounted,avg_reward)
            if alg == 'Q':
                Q = computePrediction(state, action, value_network)
                Ut = computeTargets_Q(reward, next_state, DF, target_value_network, discounted,avg_reward)
            if alg == 'doubleQ':
                Q = computePrediction(state, action, value_network)
                Ut = computeTargets_doubleQ(reward, next_state, DF, target_value_network, value_network, discounted,
                                            avg_reward)

            # Accumulate gradients wrt θ
            loss = crit(Q, Ut)
            loss.backward()
            avg_reward += float(BETA*(Ut-Q))

        # in this mutex lock we perform hard updates to the target value network, save the value network's
        # parameters and also log process. The lock is necessary as for the asynchronous agents to not
        # overwrite each other updates
        with lock:
            counter.value += 1

            # Update the target network θ− ← θ
            if counter.value % I_target == 0:
                hard_update(target_value_network, value_network)

            if counter.value % SAVE_EVERY == 0:
                saveModelNetwork(target_value_network, path+'/params_'+str(int(counter.value/SAVE_EVERY)))

            if counter.value % 100 == 0:
                logProgress(counter, SAVE_EVERY, 50)

            # Perform asynchronous update of θ using dθ. Clear gradients dθ ← 0.
            if t % I_update == 0:
                optimizer.step()
                optimizer.zero_grad()

            f.write("%.3f, %.2f\n" % (reward,avg_reward))
            f.flush()

        past_action = action
        past_reward = reward
        past_state = state
        state = next_state

    saveModelNetwork(value_network, path+'/params_last')
    return


# method which sets and returns epsilon parameter given the specific time-step
def set_epsilon(frames, min_e):
    e = 1 - frames/(4000000/(1-min_e))
    return max(min_e, e)


# method which performs epsilon-greedy exploration: epsilon probability of following a random action
# and 1-epsilon probability of following the best-rewarding action at this state.
def get_egreedy_action(observation, q_model, epsilon,num_actions):
    random_action= random.random()<epsilon

    if random_action:
        action= random.randint(0,num_actions-1)
    else:
        Qpreds=q_model.forward(observation).detach()
        action = Qpreds.argmax()
    return action


# method which computes and returns target update for the SARSA learning algorithm
def computeTargets_SARSA(reward, observation, action, discount_factor, target_network, discounted, avg_reward):
    q_pred = target_network.forward(observation).detach()
    if discounted:
        target = reward + discount_factor * q_pred[0][action]
    else:
        target = reward - avg_reward + q_pred[0][action]

    return target


# method which computes and returns the target update for the Q-learning algorithm
def computeTargets_Q(reward, next_observation, discount_factor, target_network, discounted, avg_reward):
    q_pred = target_network.forward(next_observation).detach()
    if discounted:
        target = reward + discount_factor * q_pred.max()
    else:
        target = reward - avg_reward + q_pred.max()
    return target


# method which computes and returns the target update for the double Q-learning algorithm
def computeTargets_doubleQ(reward, next_observation, discount_factor, target_network, value_network,
                           discounted, avg_reward):
    target_q_pred = target_network.forward(next_observation).detach()
    q_pred = value_network.forward(next_observation)[:, torch.argmax(target_q_pred)].detach()

    if discounted:
        target = reward + discount_factor * q_pred
    else:
        target = reward - avg_reward + q_pred
    return target


# method that returns the state-action value given a state, a value and a value network
def computePrediction(state, action, value_network):
    out = value_network.forward(state)
    return out[0][action]


# method saving the given model's parameters in the given directory
def saveModelNetwork(model, str_directory):
    torch.save(model.state_dict(), str_directory)


# method used to display a progress bar on the terminal
def logProgress(counter, saveIternval, barLength):
    save_iteration = int(counter.value/saveIternval)
    percent = counter.value%saveIternval * 100 / saveIternval
    makers   = '|' * int(percent/100 * barLength - 1)
    spaces  = ' ' * (barLength - len(makers))
    string = 'Parameter %d progress:\t |%s%s| %d %%' % (save_iteration, makers, spaces, percent)
    print(string.ljust(100), end='\r')





