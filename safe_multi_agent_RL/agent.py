from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, env, hidden_size=16,):
        super(Policy, self).__init__()
        state_size = env.state_space
        action_size = env.action_space
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class ReinforceAgent:
    def __init__(self, env, lr, gamma, hidden_size=16):
        self.model = Policy(env, hidden_size)
        self.model = self.model.to(self.model.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        self.gamma = gamma

        self.log_probs = []
        self.rewards = []
        self.policy_losses = []

    def act(self, state):
        return self.model.act(state)

    def append(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def step(self):
        discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, self.rewards)])
        policy_loss = []
        for log_prob in self.log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.policy_losses.append(policy_loss)
        self.rewards = []
        self.log_probs = []

    def update(self):
        loss = torch.mean((torch.stack(self.policy_losses)))
        self.policy_losses = []
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
