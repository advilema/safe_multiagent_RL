from torch.distributions import Categorical, MultivariateNormal
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

    def get_dist(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        dist = Categorical(probs)
        return dist

    def act(self, state):
        dist = self.get_dist(state)
        action = dist.sample()
        return action, dist.log_prob(action)


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
        sigma_sq = F.relu(self.fc2_(x))

        return mu, sigma_sq

    def get_dist(self, state): #state here is a tensor
        mu, sigma_sq = self.forward(state)
        cov_mat = torch.diag_embed(sigma_sq).to(device)
        #print('shape', cov_mat.shape[1])
        #print('state', state)
        #print('mu {}, sigma: {}'.format(mu, sigma_sq))
        #print('cov_mat', cov_mat)
        #print(cov_mat)
        dist = MultivariateNormal(mu, cov_mat)
        return dist


    def act(self, state): #state here is a numpy array
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist = self.get_dist(state)
        action = dist.sample()
        return action, dist.log_prob(action)


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
        self.continuous = continuous
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
        action, log_prob = self.actor.act(state)
        self.log_probs.append(log_prob)
        if self.continuous:
            return action.tolist(), log_prob
        else:
            return action.item(), log_prob

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
        R = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R
            returns.insert(0, R)
        return returns

    def step(self):
        """
        compute the policy losses at the end of an episode
        """
        returns = self.compute_returns()
        advantage = [r - v for r, v in zip(torch.FloatTensor(returns), torch.cat(self.values))]
        advantage = torch.cat(advantage)

        self.log_probs = torch.cat(self.log_probs)
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
        super().__init__(env, params, continuous=continuous, hidden_size_actor=16, hidden_size_critic1=128, hidden_size_critic2=256)
        self.MseLoss = nn.MSELoss()
        self.eps_clip = eps_clip
        self.K_epochs = 10
        self.states = []
        self.actions = []
        self.returns = []

    def act(self, state):
        value = self.get_value(state)
        action, log_prob = self.actor.act(state)
        state = torch.FloatTensor(state).to(device)

        self.values.append(value)
        self.actions.append(action)
        self.states.append(state)
        self.log_probs.append(log_prob)

        if self.continuous:
            return action.tolist(), log_prob
        else:
            return action.item(), log_prob

    def evaluate(self, state, action):
        dist = self.actor.get_dist(state)
        log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return log_probs, state_values, dist_entropy

    def step(self):
        returns = super().compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        self.returns.extend(returns)
        self.rewards = []

    def update(self):
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.states), dim=0).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(device)
        old_log_probs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(device)
        returns = torch.squeeze(torch.stack(self.returns, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        # Evaluating old actions and values
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) - 0.01*dist_entropy

            # take gradient step
            self.optimizerActor.zero_grad()
            self.optimizerCritic.zero_grad()
            loss.mean().backward()
            self.optimizerActor.step()
            self.optimizerCritic.step()

        self.states = []
        self.actions = []
        self.log_probs = []
        self.returns = []







