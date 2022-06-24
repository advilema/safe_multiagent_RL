import numpy as np
from scipy.spatial import distance_matrix
from random import random

#TODO: implement matplotlib rendering
class CollisionAvoidance(object):
    """This class implements a grid MDP."""

    def __init__(self, size, n_agents, n_landmarks=None, shuffle=True, agents_size=0.5):
        self.size = size
        self.n_agents = n_agents
        self.agents_size = agents_size
        self.agents = [Agent(i, size, agents_size) for i in range(n_agents)]
        self.n_landmarks = n_landmarks if n_landmarks is not None else 1
        self.start_landmarks = [(np.random.rand(2) * self.size).tolist()]
        self.landmarks = self.start_landmarks.copy()
        self.shuffle = shuffle
        self.action_space = 2  # (up, down), (left, right)
        self.state_space = 2*n_agents
        if self.shuffle:
            self.state_space += 2*self.n_landmarks #2D position of one agent + 2D position of the landmarks
        self.viewer = None
        self.constraint_space = [1]

    def reset(self):
        for agent in self.agents:
            agent.done = False
        if self.shuffle:
            return self._reset()
        else:
            return self._restart()

    def _reset(self):
        state = []
        for agent in self.agents:
            agent.state = agent.reset()
            state.append(agent.state.copy())
        #self._reset_landmarks()
        [state.append(landmark) for landmark in self.landmarks] #concatenate the state of each agent with the landmarks
        #state = self.normalize_state(state)
        return state

    def _restart(self):
        state = []
        for agent in self.agents:
            agent.state = agent.start
            state.append(agent.state.copy())
        self.landmarks = self.start_landmarks.copy()
        #[pos.extend(landmark) for landmark in self.landmarks for pos in state] #concatenate the state of each agent with the landmarks
        #state = self.normalize_state(state)
        return state

    def _reset_landmarks(self):
        self.landmarks = [(np.random.rand(2) * self.size).tolist() for _ in range(self.n_landmarks)]

    def transition(self, action):
        """Transition p(s'|s,a)."""
        states = []

        for agent, a in zip(self.agents, action):
            if agent.done:
                states.append(agent.state.copy())
                continue
            x, y = agent.state
            dx, dy = np.squeeze(a)
            norm = np.sqrt(dx**2+dy**2)
            max_norm = 1
            if norm > max_norm:
                dx = dx / norm
                dy = dy / norm
            x_ = max(0, min(self.size, x + dx))
            y_ = max(0, min(self.size, y + dy))
            agent.state = [x_, y_]
            states.append(agent.state.copy())
            for land in self.landmarks:
                if np.linalg.norm(np.array(agent.state) - np.array(land)) < self.agents_size:
                    agent.done = True
        return states

    def reward(self):
        """Reward depends on the color of the state"""
        rew = - self._agents_landmarks_distances()
        return [rew for i in range(self.n_agents)]

    def constraint(self):
        return [self._collisions()]

    def check_done(self):
        return [agent.done for agent in self.agents]

    def step(self, action):
        state = self.transition(action)
        if self.shuffle:
            [state.append(landmark) for landmark in self.landmarks.copy()]
        reward = self.reward()
        constraint = self.constraint()
        done = self.check_done()
        #state = self.normalize_state(state)
        return state, reward, constraint, done

    def _collisions(self):
        states = [agent.state for agent in self.agents if not agent.done]
        if len(states) == 0:
            return 0
        agent_distances = distance_matrix(states, states)
        n_collisions = (np.sum(agent_distances < self.agents_size) - len(states))/2
        return n_collisions

    def _agents_landmarks_distances(self):
        states = [agent.state for agent in self.agents]
        agents_landmarks_distances = distance_matrix(states, self.landmarks)
        dist = np.sum(np.amin(agents_landmarks_distances, axis=1))
        return dist

    def normalize_state(self, states):
        return [[s/self.size for s in state] for state in states]

    def _coupleToInt(self, x, y):
        return y + x * self.size

    def _intToCouple(self, n):
        return int(np.floor(n / self.size)), int(n % self.size)


class Agent(object):

    def __init__(self, index, size, agent_size):
        self.index = index
        self.env_size = size
        self.agent_size = agent_size
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False

    def reset(self):
        return (np.random.rand(2)*self.env_size).tolist()
        #return [(2*self.index%self.env_size + 1)*self.agent_size, int(self.index/self.env_size)]

if __name__ == '__main__':
    size = 3
    n_agents = 3
    n_landmarks = 1

    env = CollisionAvoidance(size, n_agents, n_landmarks, shuffle=False)

    state = env.reset()

    #env.render()

    for i in range(100):
        action = [[random() for j in range(env.action_space)] for k in range(n_agents)]
        #action = [int(random()) for agent in range(n_agents)]
        state, reward, constraint, done = env.step(action)
        print(state, reward, constraint, done)

        #env.render()
