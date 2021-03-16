# RL Reward Experiments

For documentation regarding the tasks used in these experiments refer to the following repository: [RL-Continuing-Tasks](https://github.com/Lucas-De/RL-Continuing-Tasks)


## Training
 
### Usage
```python search.py [options]```
 
|Option|Description|Default|
| --- | ---| ---|
|`--num_processes`| Number of asynchronous agents|`16`|
|`--steps`|Total number of steps distributed across all synchronous agents in millions|`16`|
|`--algorithm` | Algorithm: `Q` (for Q-Learning) or `SARSA` (for SARSA)|`Q`|
|`--network` | Network specification: `linear` or `deep` (Architecture may depend on task. [Networks.py](https://github.com/Lucas-De/RL-reward-experiments/blob/main/Code/Networks.py)|`linear`|
|`--reward` | Type of Reward: `discounted` (for descounted returns) or `average` (for average rewards)|`discounted`|
|`--task` | The task ID: `1`, `2` or `3`|`1`|
|`--lr`| Learning Rate (Refer to [Thesis](https://github.com/Lucas-De/RL-reward-experiments/blob/main/MSc_Thesis.pdf))|`0.0001`|
|`--beta`|Beta: Used to calculate average reward when reward option is `average` (Refer to [Thesis](https://github.com/Lucas-De/RL-reward-experiments/blob/main/MSc_Thesis.pdf))|`0.001`|
|`--df`|Discount Factor: Used to weight future rewards when reward option is `discounted` (Refer to [Thesis](https://github.com/Lucas-De/RL-reward-experiments/blob/main/MSc_Thesis.pdf))|`0.99`|

### Logging
The command above will generate logs in the following directory: `./Code/logs/{algorithm}/{reward}/{network}`
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
|`--param` | Network parameters path ||

*Note: Vizalusation uses Matplotlib with Qt5Agg backend. Some issues have been identidied on some platforms*



  
