'''
Mixer for CMIX-S and plain QMIX algorithm
'''
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, args, mixing_embed_dim=32, hypernet_layers=2, hypernet_embed=64):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        self.hypernet_embed = hypernet_embed
        self.args = args

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers i.e. can be negative
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
    
    '''
    for using cql:
        the agent_qs has shape [batch, n_opponent_actions, n_agents]
    for not using cql:
        the agent_qs has shape [batch, n_agents]
    '''
    def forward(self, agent_qs, states, use_cql=False):
        bs = agent_qs.size(0)
        sz = agent_qs.size()

        states = states.view(-1, self.state_dim)
        if not use_cql:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        if self.args.policy_disc:
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        else:
            # for mixing strategy we apply a linear mixer
            # so combine the actions' Q values locally will not cause bias  
            hidden = th.bmm(agent_qs, w1) + b1 
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v

        # Reshape and return
        if len(sz) == 3:
            # for recurrent state or cql state
            q_tot = y.view(bs, -1, 1)
        elif len(sz) == 2:
            q_tot = y.view(bs, 1)
        else:
            raise Exception("Input with dimention larger than 3 is not implemented")
            
        return q_tot
