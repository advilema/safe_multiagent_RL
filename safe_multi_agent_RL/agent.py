from torch.distributions import Categorical
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable


pi = Variable(torch.FloatTensor([math.pi])).cpu()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class DiscretePolicy(nn.Module):
    def __init__(self, env, hidden_size=16):
        super(DiscretePolicy, self).__init__()
        state_size = env.state_space
        action_size = env.action_space
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class ContinuousPolicy(nn.Module):
    def __init__(self, env, hidden_size=16,):
        super(ContinuousPolicy, self).__init__()
        state_size = env.state_space
        action_size = env.action_space

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.fc2_ = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        sigma_sq = self.fc2_(x)

        return mu, sigma_sq

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mu, sigma_sq = self.forward(state)
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt() * Variable(eps).to(device)).data
        prob = normal(action, mu, sigma_sq)

        log_prob = prob.log().sum().unsqueeze(0)
        return action.cpu().detach().numpy(), log_prob


class Critic(nn.Module):
    def __init__(self, env, hidden_size1=128, hidden_size2=256):
        state_size = env.state_space
        action_size = env.action_space
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, state):
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        value = self.fc3(output)
        return value

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = self.forward(state)
        return value.cpu().detach().numpy()


class AbstractAgent:
    def __init__(self, env, params, continuous=False, hidden_size=16):
        if continuous:
            self.actor = ContinuousPolicy(env, hidden_size)
        else:
            self.actor = DiscretePolicy(env, hidden_size)
        self.actor = self.actor.to(device)
        self.optimizerActor = optim.Adam(self.actor.parameters(), lr=params.actor_lr)
        self.actor.train()
        self.gamma = params.gamma

        self.log_probs = []
        self.rewards = []
        self.actor_losses = []

    def act(self, state):
        """
        Return an action for the given state according to the policy
        """
        return self.actor.act(state)

    def append(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_returns(self):
        discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
        returns = [a * b for a, b in zip(discounts, self.rewards)]
        return returns

    def step(self):
        """
        compute the policy losses at the end of an episode
        """
        R = sum(self.compute_returns())
        actor_loss = []
        for log_prob in self.log_probs:
            actor_loss.append(-log_prob * R)
        actor_loss = torch.cat(actor_loss).sum()

        self.actor_losses.append(actor_loss)
        self.rewards = []
        self.log_probs = []

    def update(self):
        """
        Update the policy depending on the mean of the losses of the episodes in the batch
        """
        loss = torch.mean((torch.stack(self.actor_losses)))
        self.actor_losses = []
        self.optimizerActor.zero_grad()
        loss.backward()
        self.optimizerActor.step()


class ReinforceAgent(AbstractAgent):
    def __init__(self, env, params, continuous=False, hidden_size=16):
        super().__init__(env, continuous, params, hidden_size)

    def step(self):
        """
        compute the policy losses at the end of an episode
        """
        R = sum(super().compute_returns())
        actor_loss = []
        for log_prob in self.log_probs:
            actor_loss.append(-log_prob * R)
        actor_loss = torch.cat(actor_loss).sum()

        self.actor_losses.append(actor_loss)
        self.rewards = []
        self.log_probs = []


class ACAgent(AbstractAgent):
    def __init__(self, env, params, continuous=False, hidden_size_actor=16, hidden_size_critic1=128, hidden_size_critic2=256):
        super().__init__(env, params, continuous=continuous, hidden_size=hidden_size_actor)
        self.critic = Critic(env, hidden_size_critic1, hidden_size_critic2)
        self.optimizerCritic = optim.Adam(self.critic.parameters(), lr=params.critic_lr)
        self.values = []
        self.critic_losses = []


    def get_value(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = self.critic(state_tensor)
        return value

    def act(self, state):
        """
        Return an action for the given state according to the policy
        """
        value = self.get_value(state)
        self.values.append(value)
        return super().act(state)

    def compute_returns(self):
        R = self.values[-1]
        returns = []
        rewards = self.rewards[:-1]
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            returns.insert(0, R)
        return returns

    def step(self):
        """
        compute the policy losses at the end of an episode
        """
        returns = self.compute_returns()
        advantage = [r - v for r, v in zip(torch.FloatTensor(returns), torch.cat(self.values[:-1]))]
        advantage = torch.cat(advantage)

        self.log_probs = torch.cat(self.log_probs[:-1])
        actor_loss = -(self.log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()

        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.rewards = []
        self.values = []
        self.log_probs = []

    def update(self):
        """
        Update the policy depending on the mean of the losses of the episodes in the batch
        """

        actor_loss = torch.mean((torch.stack(self.actor_losses)))
        self.actor_losses = []
        self.optimizerActor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizerActor.step()

        critic_loss = torch.mean((torch.stack(self.critic_losses)))
        self.critic_losses = []
        self.optimizerCritic.zero_grad()
        critic_loss.backward()
        self.optimizerCritic.step()


class PPOAgent(ACAgent):
    def __init__(self, env, params, continuous=False, hidden_size_actor=16, hidden_size_critic1=128, hidden_size_critic2=256, eps_clip=0.2):
        super().__init__(env, params, continuous=False, hidden_size_actor=16, hidden_size_critic1=128, hidden_size_critic2=256)
        self.MseLoss = nn.MSELoss()
        self.eps_clip = eps_clip
        self.K_epochs = 10
        self.states = []
        self.actions = []

    def append(self, log_prob, rew, state, action):
        super().append(log_prob, rew)
        self.states.append(state)
        self.actions.append(action)

    def step(self):
        # Monte Carlo estimate of returns
        if len(self.old_states) == 0:
            return

        # Normalizing the returns
        returns = super().compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(device)
        old_log_probs = torch.squeeze(torch.stack(self.old_log_probs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        # Evaluating old actions and values
        for _ in range(self.K_epochs):
            log_probs = []
            state_values = []
            for state in self.old_states:
                _, log_prob = self.act(state)
                log_probs.append(log_prob)
                state_value = super().get_value(state)
                state_values.append(state_value)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantage = [r - v for r, v in zip(torch.FloatTensor(returns), state_values)]
            advantage = torch.cat(advantage)
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns)

            # take gradient step
            self.optimizerActor.zero_grad()
            loss.mean().backward()
            self.optimizerCritic.step()







