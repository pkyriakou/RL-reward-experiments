# RL Reward Experiments

For documentation regarding the tasks used in these experiments refer to the following repository: [RL-Continuing-Tasks](https://github.com/Lucas-De/RL-Continuing-Tasks)

This is a fork of Lucas Descause's original repository. It is the codebase used for the experiments of my master's dissartation "Reinforcement Learning with Function Approximation in Continuing Tasks: Discounted Return or Average Reward?"

**Abstract**

Reinforcement learning is a machine learning sub-field, involving an agent performing sequential decision making and learning through trial and error inside a predefined environment. An important design decision for a reinforcement learning algorithm is the return formulation, which formulates the future expected returns that the agent receives after following any action in a specific environment state. In continuing tasks with value function approximation (VFA), average rewards and discounted returns can be used as the return formulation but it is unclear how the two formulations compare empirically. This dissertation aims at empirically comparing the two return formulations. We experiment with three continuing tasks of varying complexity, three learning algorithms and four different VFA methods. We conduct three experiments investigating the average performance over multiple hyperparameters, the performance with near-optimal hyperparameters and the hyperparameter sensitivity of each return formulation. Our results show that there is an apparent performance advantage in favour of the average rewards formulation because it is less sensitive to hyperparameters. Once hyperparameters are optimized, the two formulations seem to perform similarly.

## Training
 
### Usage
```python search.py [options]```

For documentation on algorithm parameters refer to [Thesis](https://github.com/Lucas-De/RL-reward-experiments/blob/main/MSc_Thesis.pdf)
 
|Option|Description|Default|
| --- | ---| ---|
|`--num_processes`| Number of asynchronous agents|`16`|
|`--steps`|Total number of steps distributed across all synchronous agents in millions|`16`|
|`--algorithm` | Algorithm: `Q` (for Q-Learning) or `SARSA` (for SARSA) or `doubleQ` (for Double Q-Learning)|`Q`|
|`--network` | Network specification: `linear` or `deep` (Architecture may depend on task. See [Networks.py](https://github.com/Lucas-De/RL-reward-experiments/blob/main/Code/Networks.py) for detailed architecture)|`linear`|
|`--degree` | Polynomial Degree to expand environment vector `1`, `2` or `3`|`1`|
|`--reward` | Type of Reward: `discounted` for discounted returns or `average` for average rewards |`discounted`|
|`--task` | The task ID: `1`, `2` or `3`|`1`|
|`--lr`| Learning Rate |`0.0001`|
|`--beta`|Beta: Used to calculate average reward when reward option is `average`|`0.001`|
|`--df`|Discount Factor: Used to weight future rewards when reward option is `discounted`|`0.99`|
|`--seed`| Random seed for experiment replication|`0`|

### Logging
The command above will generate logs in the following directory: `./Code/logs/{algorithm}/{reward}/{network}/{process id}`
The directory will contain:
- A log file for each asynchronous agent containing the reward and total avereage reward at each step
- The network parameters saved for each million steps (cumulated over all agents)
- A `hyper_params` file specifying: 
    - The parameters chosen for the run
    - The Average reward for each saved network (over 5 sample runs of 50,000 steps)

## Vizualization
After training the agent, it may be useful to observe its behavior in the environment. 
To do so use `python visualize.py [options]` 
|Option|Description|Default|
| --- | ---| ---|
|`--task` | The task ID: `1`, `2` or `3`|`1`|
|`--network` | Network specification: `linear` or `deep` |`linear`|
|`--degree` | Polynomial Degree to expand environment vector `1`, `2` or `3`|`1`|
|`--param` | Network parameters path ||

*Note: Vizalusation uses Matplotlib with Qt5Agg backend. Some issues have been identidied on some platforms*



  
