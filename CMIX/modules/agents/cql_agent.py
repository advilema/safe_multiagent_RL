import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize
import numpy as np


class CQLAgent(nn.Module):
    def __init__(self, input_shape, n_actions, n_opponent_actions, hidden_dim=64):
        super(CQLAgent, self).__init__()

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions * n_opponent_actions)
        self.n_actions = n_actions
        self.n_opponent_actions = n_opponent_actions

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        q = self.fc2(x)
        q = q.view(-1, self.n_opponent_actions, self.n_actions)
        return q

    '''
        @param:
            inputs: [batch, input_shape]
            policy_disc: whether to use the discrete policy
        @ retval: problem distributions of the action shape: [batch, n_actions] type:np.array
    '''
    def get_policy(self, inputs, policy_disc=True):
        qvals = self.forward(inputs)
        if policy_disc:
            qvals = th.min(qvals, axis=1)[0]
            actions = th.argmax(qvals, axis=1)
            policys = F.one_hot(actions, num_classes=self.n_actions).float().detach()
        else:
            policys = []
            qvals = qvals.detach().numpy()
            for qval_sample in qvals:
                c = np.array([0] * self.n_actions + [- 1]) # minimize the -lower_bound
                A_ub = np.concatenate((-qval_sample, np.ones((self.n_opponent_actions, 1))), axis=1)
                B_ub = np.zeros(self.n_opponent_actions)
                A_eq = np.array([[1] * self.n_actions + [0]])
                B_eq = np.array([1])
                bounds = []
                for a in range(self.n_actions):
                    bounds.append((0, 1))
                bounds.append((None, None))
                res = optimize.linprog(c, A_ub, B_ub, A_eq, B_eq, bounds=tuple(bounds))
                policy = res['x']
                policys.append(policy[:-1])
            policys = th.tensor(policys, dtype=th.float32)
        return policys

                    



