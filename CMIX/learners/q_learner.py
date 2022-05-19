import copy
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, agents, args):
        self.args = args
        self.agents = agents
        self.n_agents = len(agents)
        self.params = []
        for i in range(self.n_agents):
            self.params += list(agents[i].parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args.state_dim, self.n_agents, args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        if args.optimizer == "RMSprop":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            self.optimiser = Adam(params=self.params, lr=args.lr)

        self.target_agents = copy.deepcopy(self.agents)

    def train(self, batch):
        # Get the relevant quantities
        local_rewards = th.tensor([d['local_rewards'] for d in batch], dtype=th.float32)
        global_reward = th.tensor([d['global_reward'] for d in batch], dtype=th.float32).view(-1, 1)
        done_mask = th.tensor([d['done_mask'] for d in batch], dtype=th.float32).view(-1, 1) 
        actions = th.tensor([d['actions'] for d in batch]).view(-1, self.n_agents, 1) 
        global_state = th.tensor([d['global_state'] for d in batch], dtype=th.float32)
        global_state_new = th.tensor([d['global_state_new'] for d in batch], dtype=th.float32)
        states = []
        states_new = []
        for i in range(self.n_agents):
            state = th.tensor([d['states'][i] for d in batch], dtype=th.float32)
            state_new = th.tensor([d['states_new'][i] for d in batch], dtype=th.float32)
            states.append(state)
            states_new.append(state_new)

        # Calculate estimated Q-Values
        chosen_action_qvals = []
        for i in range(self.n_agents):
            agent_out = self.agents[i](states[i]) 
            qvals_chosen = th.gather(agent_out, dim=1, index=actions[:, i]) 
            chosen_action_qvals.append(qvals_chosen) 
        chosen_action_qvals = th.cat(chosen_action_qvals, dim=1) 

        # Calculate the Q-Values for the target
        target_max_qvals = []
        for i in range(self.n_agents):
            agent_out = self.target_agents[i](states_new[i])
            max_qval = agent_out.max(dim=1)[0]
            target_max_qvals.append(max_qval)
        target_max_qvals = th.stack(target_max_qvals, -1) 

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, global_state)
            target_max_qvals = self.target_mixer(target_max_qvals, global_state_new)
            rewards = global_reward
        else:
            rewards = local_rewards

        # Calculate 1-step Q-Learning targets
        # rewards shape can be (batch, 1) or (batch, n_agents)
        targets = rewards + self.args.gamma * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        
        #QMIX loss with weight, using OW-QMIX
        if self.mixer is not None:
            ws = th.ones_like(td_error) * self.args.weight_alpha
            ws = th.where(td_error < 0, th.ones_like(td_error)*1, ws)
            loss = (ws.detach() * (done_mask * td_error ** 2)).mean()
        else:
            # Normal L2 loss
            loss = (done_mask * td_error ** 2).mean()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


        self.last_target_update_episode += 1
        if self.last_target_update_episode == self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = 0
        
        return loss.detach().item()

    def _update_targets(self):
        for i in range(self.n_agents):
            self.target_agents[i].load_state_dict(self.agents[i].state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        for i in range(self.n_agents):
            self.agents[i].cuda()
            self.target_agents[i].cuda()

        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        for i in range(self.n_agents):
            th.save(self.agents[i].state_dict(), "{}/agent{}.th".format(path, i))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        

    def load_models(self, path):
        for i in range(self.n_agents):
            self.agents[i].load_state_dict(th.load("{}/agent{}.th".format(path, i)))
            self.target_agents[i].load_state_dict(th.load("{}/agent{}.th".format(path, i)))
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path)))
