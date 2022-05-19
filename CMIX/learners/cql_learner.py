'''
for constrained Q learning, refer to CMIX paper
there are n_opponent_actions global rewards for the system,
CQL must have a mixer
'''
import copy
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.multi_qmix import MultiQMixer
import torch as th
from torch.optim import RMSprop, Adam


class CQLLearner:
    def __init__(self, agents, args):
        self.args = args
        self.agents = agents
        self.n_opponent_actions = args.n_opponent_actions
        self.state_dim = args.state_dim
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
                self.mixer = QMixer(self.state_dim, self.n_agents, args)
            elif args.mixer == "multi-qmix":
                self.mixer = MultiQMixer(self.state_dim, self.n_agents, self.n_opponent_actions, args)
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
        global_reward = th.tensor([d['global_reward'] for d in batch], dtype=th.float32)
        done_mask = th.tensor([d['done_mask'] for d in batch], dtype=th.float32)
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
        actions_expand = actions.unsqueeze(1).expand(-1, self.n_opponent_actions, -1, -1)
        chosen_action_qvals = []
        chosen_action_qvals_noise2 = []
        max_qvals = []
        max_lbound_qvals = []

        for i in range(self.n_agents):
            agent_out = self.agents[i](states[i]) # [batch, n_opponent_actions, n_actions]
            qvals_chosen = th.gather(agent_out, dim=2, index=actions_expand[:, :, i]) # batch * n_opponent_actions * 1 
            chosen_action_qvals.append(qvals_chosen)
            
            # add noise towards the action, for gap loss
            action_noise2 = th.randint(low=0, high=agent_out.shape[-1], size=actions[:, i].shape) 
            rands = th.rand(actions[:, i].shape)
            action_noise2 = th.where(rands < 1., action_noise2, actions[:, i]) # for testing
            
            action_noise2 = action_noise2.unsqueeze(1).expand(-1, self.n_opponent_actions, -1)
            qvals_chosen_noise2 = th.gather(agent_out, dim=2, index=action_noise2) 
            chosen_action_qvals_noise2.append(qvals_chosen_noise2)
            
        chosen_action_qvals = th.cat(chosen_action_qvals, dim=2) # batch * n_opponent_actions * n_agents
        #chosen_action_qvals_noise = chosen_action_qvals + th.normal(0, th.var(chosen_action_qvals).item(), chosen_action_qvals.shape) # reparameterization sampling, deprecated in CMIX
        chosen_action_qvals_noise2 = th.cat(chosen_action_qvals_noise2, dim=2) # batch * n_opponent_actions * n_agents

        # for lower bound margin loss 
        chosen_action_min_qvals_local = chosen_action_qvals.min(dim=1)[0]
        #chosen_action_min_qvals_local_noise = chosen_action_qvals_noise.min(dim=1)[0] #not used in CMIX
        chosen_action_min_qvals_local_noise2 = chosen_action_qvals_noise2.min(dim=1)[0] 
        
        # Calculate the Q-Values for the target
        target_max_qvals = []
        for i in range(self.n_agents):
            policy = self.target_agents[i].get_policy(states_new[i], self.args.policy_disc)
            agent_out = self.target_agents[i](states_new[i])
            max_qval = th.bmm(agent_out, policy.unsqueeze(-1)).squeeze(-1)
            target_max_qvals.append(max_qval)
        target_max_qvals = th.stack(target_max_qvals, -1) # batch * n_opponent_actions * n_agents

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, global_state, use_cql=True).squeeze(-1) 
            chosen_action_qvals_noise2 = self.mixer(chosen_action_qvals_noise2, global_state, use_cql=True).squeeze(-1)
            target_max_qvals = self.target_mixer(target_max_qvals, global_state_new, use_cql=True).squeeze(-1) # batch * n_opponent_actions
            rewards = global_reward
            
            chosen_action_min_qvals_noise2 = chosen_action_qvals_noise2.min(dim=1)[0] # batch
            # for lower bound gap loss
            if self.args.mixer in ["qmix", "vdn"]:
                chosen_action_min_qvals_lbound_noise2 = self.mixer(chosen_action_min_qvals_local_noise2, global_state, use_cql=False).squeeze(-1)
            else:
                # for CMIX-M
                chosen_action_min_qvals_local_expand_noise2 = chosen_action_min_qvals_local_noise2.unsqueeze(1).expand(-1, self.n_opponent_actions, -1)
                chosen_action_min_qvals_lbound_noise2 = self.mixer(chosen_action_min_qvals_local_expand_noise2, global_state, use_cql=True).squeeze(-1).min(dim=1)[0]  
            
            chosen_action_gap = chosen_action_min_qvals_noise2 - chosen_action_min_qvals_lbound_noise2  
            chosen_action_gap_loss = (done_mask * chosen_action_gap ** 2).mean()
        else:
            rewards = th.transpose(local_rewards, 1, 2)
            chosen_action_gap_loss = th.tensor(0.)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        
         
        #QMIX loss with weight, using OW-QMIX
         
        if self.mixer is not None:
            ws = th.ones_like(td_error) * self.args.weight_alpha
            ws = th.where(td_error < 0, th.ones_like(td_error)*1, ws)
            loss = (ws.detach() * (done_mask.view(-1, 1) * td_error ** 2)).mean()
        else:
            # Normal L2 loss, take mean over actual data
            loss = (done_mask.view(-1, 1, 1) * td_error ** 2).mean()
        
        # use gap loss
        if self.mixer is not None:
            total_loss = loss + self.args.loss_beta * chosen_action_gap_loss
        else:
            total_loss = loss

        # Optimise
        self.optimiser.zero_grad()
        total_loss.backward()
        self.optimiser.step()


        self.last_target_update_episode += 1
        if self.last_target_update_episode == self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = 0
        
        return total_loss.detach().item(), loss.detach().item(), chosen_action_gap_loss.detach().item()

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
