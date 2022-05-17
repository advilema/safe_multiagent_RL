import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DiscreteActor:
    def __init__(self, env, hidden_size=16):
        state_size = env.state_space
        action_size = env.action_space
        actor = Sequential()
        actor.add(Dense(24, input_dim=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=actor_lr))
        self.actor = actor

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


class ContinuousActor:
    def __init__(self, env, hidden_size=16,):
        super(ContinuousPolicy, self).__init__()
        state_size = env.state_space
        action_size = env.action_space

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.fc2_ = nn.Linear(hidden_size, action_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        sigma_sq = self.fc2_(x)

        return mu, sigma_sq

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mu, sigma_sq = self.forward(state)
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt() * Variable(eps).to(self.device)).data
        prob = normal(action, mu, sigma_sq)

        log_prob = prob.log()
        return action.cpu().detach().numpy(), log_prob



class Agent:
    def __init__(self, env, lr, gamma, continuous=False, hidden_size=16):

        self.actor_1 = self.build_actor()
        self.actor_2 = self.build_actor()
        self.critic = self.build_critic()
        self.critic2 = self.build_critic2()

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        #actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size*2, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        #critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    def get_action_1(self, state):
        policy = self.actor_1.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def get_maxaction_2(self,state):
        policy = self.actor_2.predict(state, batch_size=1).flatten()
        return np.argmax(policy)

