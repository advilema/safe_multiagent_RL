import numpy as np
import torch
from safe_multi_agent_RL.agent import ReinforceAgent
from safe_multi_agent_RL.buffer import Buffer
from safe_multi_agent_RL.meta_agent import MetaAgent
from safe_multi_agent_RL.util import make_env, print_info
from cli_parse import cli
import time


if __name__ == '__main__':
    params = cli()

    #if args.constrained:
    #    with open('params.json', 'r') as file:
    #        params = json.load(file)
    #else:
    #    with open('params_unconstrained.json', 'r') as file:
    #        params = json.load(file)
    #params = namedtuple("params", params.keys())(*params.values())
    np.random.seed(params.numpy_seed)
    torch.manual_seed(params.torch_seed)

    env, continuous = make_env(params)
    agents = [ReinforceAgent(env, params.lr, params.gamma, continuous=continuous) for i in range(params.n_agents)]
    if not params.unconstrained:
        meta_agent = MetaAgent(env.constraint_space, params.gamma, params.meta_lr, params.thresholds,
                               start_learning_cycle=params.n_agents_learning_cycles-1, decay=params.decay)
    else:
        meta_agent = None
    buffer = Buffer(params, constrained=not params.unconstrained)

    for meta_agent_learning_cycle in range(params.n_meta_agent_learning_cycles):
        for agents_learning_cycle in range(params.n_agents_learning_cycles):
            for batch in range(params.batch_size):
                state = env.reset()
                for t in range(params.max_t):
                    log_probs = []
                    actions = []
                    for agent in agents:
                        action, log_prob = agent.act(np.array(state).flatten())
                        log_probs.append(log_prob)
                        actions.append(action)
                    state, reward, constraint, _ = env.step(actions)
                    if not params.unconstrained:
                        modified_reward = meta_agent.act(constraint, reward)
                    else:
                        modified_reward = reward
                    #if agents_learning_cycle > 0 and batch%100 == 0:
                    #    env.render()
                    #    time.sleep(0.1)

                    for agent, log_prob, rew, constr in zip(agents, log_probs, modified_reward, constraint):
                        agent.append(log_prob, rew)
                    buffer.append(reward, modified_reward, constraint)

                buffer.step()
                [agent.step() for agent in agents]
                if not params.unconstrained:
                    meta_agent.step()

            [agent.update() for agent in agents]
            if not params.unconstrained:
                meta_agent.increment_learning_cycle()
            if agents_learning_cycle % params.print_every == 0:
                print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle)

        if not params.unconstrained:
            meta_agent.update()
            print('Lambdas: {}'.format(meta_agent.lambdas))
            buffer.append_lambdas(meta_agent.lambdas)

        if meta_agent_learning_cycle % 5 == 0:
            buffer.save_results()
