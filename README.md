#Safe Multi-Agent RL

To start the experiment simply run:
```bash
$ python main.py
```

To change the settings of the experiment, modify the params.json file. Below you can find a commented explanation of 
the keys in the dictionary:

```python
{"env": "Explore",  #environment name
  "n_meta_agent_learning_cycles": 75,  #how many times the lambdas get updated
  "n_agents_learning_cycles": 30,  #how many times the agents get updated within lambdas updates
  "batch_size": 50, 
  "max_t": 50,  #n of episodes steps
  "gamma": 0.999,
  "lr": 0.003,  #agents lr
  "meta_lr": 0.01,  #meta-agent lr
  "decay": 1.0,  # decay for the meta_agent lr
  "size": 5,  # size of the grid
  "n_agents": 3,  # n of agents
  "weights": [1,2,3],  # weights for the agents rewards (e.g. rew_agent_0 = reward[0]*weights[0])
  "thresholds": [25, 25, 25],  # thresholds for the constraints: the agent 0 shouldn't move more than thresholds[0] steps'
  "shuffle": False,  # this can be ignored for the moment
  "numpy_seed": 0,  # numpy random seed
  "torch_seed": 0,  # torch random seed
  "print_every": 1 
}
```

The results are saved at the end of each meta agent learning cycle, so you can stop the run whenever you want without loosing
the past results.