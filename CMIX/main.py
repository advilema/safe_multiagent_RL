from config.arguments import get_arg
from envs import REGISTRY as env_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from learners import REGISTRY as learner_REGISTRY
from databuffers.replaybuffer import ReplayBuffer
from components.epsilon_schedules import DecayThenFlatSchedule
from utils.utils import cleanup_dir, print_info
from buffer import Buffer

import torch as th
import numpy as np
import os
import pickle
from torch.distributions import Categorical

if __name__ == '__main__':
    args = get_arg()

    # init logging directory and logging file
    log_dir = os.path.expanduser("./log/" + args.log_dir)
    cleanup_dir(log_dir)
    eval_log_dir = log_dir + "/eval"
    cleanup_dir(eval_log_dir)
    model_log_dir = log_dir + "/model"
    cleanup_dir(model_log_dir)
    
    global_reward_file = open("%s/global_reward.log" % (log_dir), "w", 1)
    loss_file = open("%s/loss.log" % (log_dir), "w", 1)
    if args.rl_model == "cql":
        tderror_loss_file = open("%s/tderror_loss.log" % (log_dir), "w", 1)
        gap_loss_file = open("%s/gap_loss.log" % (log_dir), "w", 1)
    env_file = open("%s/env.pickle" % (log_dir), "wb", 1)

    # setup/load Environment
    if args.env_path == None:
        env = env_REGISTRY[args.application](args)
        n_agent, state_space, observation_spaces, action_spaces, n_opponent_actions = env.setup() # obses: observation states for the agents; state: the global state for the mixer
    else:
        load_env_file = open(args.env_path, "rb")
        env = pickle.load(load_env_file)
        load_env_file.close()
        n_agent, state_space, observation_spaces, action_spaces, n_opponent_actions = env.get_rlinfo()
    # save the image of the environment
    pickle.dump(env, env_file)
    env_file.close()
    args.state_dim = state_space
    args.n_opponent_actions = n_opponent_actions 
    
    # init env parameters
    env.set_logger(log_dir)
    env.init_cql()
    env.set_scheme(args.rl_model)

    # init replay buffer
    buffer = ReplayBuffer(args.buffer_size)
    our_buffer = Buffer(args, constrained=not args.unconstrained)

    
    # init epsilon decay schedule
    schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, int(args.training_epochs * 0.9), decay=args.epsilon_scheduler)

    # init agents
    agents = []
    for i in range(n_agent):
        if args.rl_model == 'simple':
            agent = agent_REGISTRY['simple'](observation_spaces[i], action_spaces[i])
        else:
            agent = agent_REGISTRY['cql'](observation_spaces[i], action_spaces[i], n_opponent_actions)
        
        agents.append(agent)

    # init learner
    learner = learner_REGISTRY[args.rl_model](agents, args)
    if args.model_load_path != None:
        learner.load_models(args.model_load_path)
    
    for episode in range(args.training_episodes):
        buffer.clear()
        for epoch in range(args.training_epochs):
            state, obses = env.reset()
            for env_t in range(args.max_env_t):
                #print("episode:{} epoch:{} step: {}".format(episode, epoch, env_t))
                transition_data = {'states': obses, 'global_state': state}
                # make action
                with th.no_grad():
                    actions = []
                    for i in range(n_agent):
                        if args.rl_model == 'simple':
                            qs = agents[i](th.tensor(obses[i], dtype=th.float32))
                            if np.random.random() > schedule.eval(epoch):
                                action = th.argmax(qs).item()
                            else:
                                action = np.random.randint(action_spaces[i])
                        else:
                            qvals = agents[i](th.tensor(obses[i], dtype=th.float32).unsqueeze(0))
                            policys = agents[i].get_policy(th.tensor(obses[i], dtype=th.float32).unsqueeze(0)) 
                            dist = Categorical(probs=policys)
                            if np.random.random() > schedule.eval(epoch):
                                action = dist.sample().item()
                            else:
                                action = np.random.randint(action_spaces[i])
                        actions.append(action)
        
                state, obses, local_rewards, global_reward, done_mask, reward, constraint = env.step(actions)
                our_buffer.append([reward], global_reward, [constraint])
                
                if done_mask == 0:
                    break
                
                transition_data['actions'] = actions
                transition_data['local_rewards'] = local_rewards
                transition_data['global_reward'] = global_reward
                transition_data["states_new"] = obses
                transition_data["global_state_new"] = state
                transition_data["done_mask"] = done_mask
                
                print(global_reward, file=global_reward_file)
                buffer.add(transition_data)
            our_buffer.step()
                
            if len(buffer) >= args.batch_size:
                batch = buffer.sample_batch(args.batch_size)
                if args.rl_model == "simple":
                    loss = learner.train(batch)
                    print(loss, file=loss_file)
                else:
                    loss, tderror_loss, gap_loss = learner.train(batch)
                    print(loss, file=loss_file)
                    print(tderror_loss, file=tderror_loss_file)
                    print(gap_loss, file=gap_loss_file)
                print("training epoch:", epoch, "loss:", loss)
                buffer.clear()
            print_info(our_buffer, epoch, episode)
        learner.save_models(model_log_dir)

