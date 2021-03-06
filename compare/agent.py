import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf



class A2CAgents:
    def __init__(self, params, env, continuous=False, hidden_size=16):
        self.state_size = env.state_space
        self.action_size = env.action_space
        self.value_size = 1
        self.constraint_size = sum(env.constraint_space)
        self.n_agents = params.n_agents
        self.continuous = continuous
        self.actor_lr = params.lr
        self.critic_lr = params.lr
        self.gamma = params.gamma
        self.threshold = np.array(params.thresholds)
        self.batch_size = params.batch_size
        self.max_t = params.max_t
        self.lam = np.zeros(sum(env.constraint_space))
        self.actors = [self.build_actor() for _ in range(self.n_agents)]
        self.critic = self.build_critic()
        self.meta_critic = self.build_meta_critic()

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        if self.continuous:
            actor.add(Dense(self.action_size+1, activation='softmax',kernel_initializer='he_uniform'))
        else:
            actor.add(Dense(self.action_size, activation='softmax',kernel_initializer='he_uniform'))
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=self.actor_lr))
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.compile(loss="mse", optimizer=Adam(learning_rate=self.critic_lr))
        return critic

    def build_meta_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.constraint_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.compile(loss="mse", optimizer=Adam(learning_rate=self.critic_lr))
        return critic

    def act(self, state):
        actions = []
        state = np.array(state).reshape(1, self.state_size)
        for actor in self.actors:
            if self.continuous:
                pred = actor.predict(state, batch_size=1, verbose=0).flatten()
                mu = pred[:self.action_size]
                sigma = pred[self.action_size]
                action = np.random.normal(mu, sigma, size=2)
            else:
                policy = actor.predict(state, batch_size=1, verbose=0).flatten()
                action = np.random.choice(self.action_size, 1, p=policy)[0]
            actions.append(action)
        return actions

    def train_model(self, state, action, reward, next_state, constr, done):

        state = tf.convert_to_tensor(np.array(state).reshape(1, self.state_size))
        next_state = tf.convert_to_tensor(np.array(next_state).reshape(1, self.state_size))

        advantages = np.zeros((self.n_agents, self.action_size))
        target = np.zeros((1, self.value_size))
        meta_target = np.zeros((1, self.constraint_size))

        value = self.critic.predict(state, verbose=0)[0]
        next_value = self.critic.predict(next_state, verbose=0)[0]
        next_meta_value = self.meta_critic.predict(next_state, verbose=0)[0]

        reward = reward + self.lam * (constr - self.threshold/self.max_t) #TODO rendere vettoriale

        if done:
            for agent_idx, a in zip(range(self.n_agents), action):
                advantages[agent_idx][a] = -1 * (reward[0] - value)
            target[0][0] = reward[0]
            meta_target[0] = constr
        else:
            for agent_idx, a in zip(range(self.n_agents), action):
                advantages[agent_idx][a] = -1 * (reward[0] + self.gamma * (next_value) - value)
            target[0][0] = reward[0] + self.gamma * next_value
            meta_target[0] = constr + self.gamma * next_meta_value

        for actor, adv in zip(self.actors, advantages):
            adv = adv.reshape(1, adv.shape[0])
            actor.fit(state, adv, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)
        self.meta_critic.fit(state, meta_target, epochs=1, verbose=0)

        lam = self.lam + 0.001 * (self.meta_critic.predict(state, verbose=0)[0] - self.threshold/self.max_t)
        self.lam = [min(max(0, l), 5) for l in lam]

        return reward
