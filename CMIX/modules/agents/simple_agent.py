import torch.nn as nn
import torch.nn.functional as F


class SimpleAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=64):
        super(SimpleAgent, self).__init__()

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)


    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        q = self.fc2(x)
        return q
