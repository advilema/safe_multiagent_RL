import numpy as np
import torch
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from safe_multi_agent_RL.buffer import Buffer
from safe_multi_agent_RL.util import make_env, print_info
from cli_parse import cli
from agent import A2CAgents


if __name__ == '__main__':
    params = cli()
    np.random.seed(params.numpy_seed)
    torch.manual_seed(params.torch_seed)

    env, continuous = make_env(params)
    agents = A2CAgents(params, env, continuous=continuous, hidden_size=16)
    buffer = Buffer(params, constrained=not params.unconstrained)

    for meta_agent_learning_cycle in range(params.n_meta_agent_learning_cycles):
        for agents_learning_cycle in range(params.n_agents_learning_cycles):
            for batch in range(params.batch_size):
                state = env.reset()
                print(batch)
                for t in range(params.max_t):
                    actions = agents.act(state)
                    next_state, reward, constraint, done = env.step(actions)
                    modified_reward = agents.train_model(state, actions, reward, next_state, constraint, done)
                    buffer.append(reward, modified_reward, constraint)

                    #if all agents are done ends the episode
                    if np.all(done):
                        break

                    state = next_state

                buffer.step()
                #[agent.step() for agent in agents]

            #[agent.update() for agent in agents]
            if agents_learning_cycle % params.print_every == 0:
                print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle)

        if params.checkpoints and meta_agent_learning_cycle % 100 == 0:
            buffer.save_results()

    buffer.save_results()

