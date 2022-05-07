import numpy as np
#from gym.envs.classic_control import rendering
from scipy.spatial import distance_matrix
from random import random

class Space(object):
    """This class implements a grid MDP."""

    def __init__(self, size, n_agents, n_landmarks, shuffle=True, agents_size = 0.5):
        self.size = size
        self.n_agents = n_agents
        self.agents_size = agents_size
        self.agents = [Agent(i, size) for i in range(n_agents)]
        self.n_landmarks = n_landmarks
        self.start_landmarks = [(np.random.rand(2)*size).tolist() for _ in range(n_landmarks)]
        self.landmarks = self.start_landmarks.copy()
        self.shuffle = shuffle
        self.action_space = 5 #up, down, left, right, stay
        self.state_space = 2*n_agents #TODO: creare array con specifica space size per ogni agente
        if self.shuffle:
            self.state_space += 2*self.n_landmarks #2D position of one agent + 2D position of the landmarks
        self.viewer = None

    def reset(self):
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
        [pos.extend(landmark) for landmark in self.landmarks for pos in state] #concatenate the state of each agent with the landmarks
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
        directions = np.array([[1, -1, 0, 0, 0], [0, 0, -1, 1, 0]])
        states = []

        for i, agent in enumerate(self.agents):
            x, y = agent.state
            #dx = np.sum(directions[0]*action[i])
            #dy = np.sum(directions[1]*action[i])
            dx, dy = directions[:, action[i]]
            x_ = max(0, min(self.size - 1, x + dx))
            y_ = max(0, min(self.size - 1, y + dy))
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states


    def reward(self):
        """Reward depends on the color of the state"""
        rew = 0

        rew -= self.size * self._collisions()
        rew -= self._agents_landmarks_distances()
        return [rew for i in range(self.n_agents)]

    def constraint(self, action):
        directions = np.array([[1, -1, 0, 0, 0], [0, 0, -1, 1, 0]])
        con = []
        for i in range(self.n_agents):
            con.append(directions*action[i])
        return con

    def check_done(self):
        return [False for i in range(self.n_agents)]

    def step(self, action):
        state = self.transition(action)
        if self.shuffle:
            [pos.extend(landmark) for landmark in self.landmarks.copy() for pos in state.copy()]
        reward = self.reward()
        constraint = self.constraint(action)
        done = self.check_done()
        #state = self.normalize_state(state)
        return state, reward, constraint, done

    def make_tables(self):
        """Returns tabular version of reward and transition functions r and p.
    """
        r = np.zeros((self.size * self.size, 4))
        p = np.zeros((self.size * self.size, 4, self.size * self.size))
        directions = np.array([[1, -1, 0, 0], [0, 0, -1, 1]])
        for x in range(self.size):
            for y in range(self.size):
                for a in range(4):
                    i = x * self.size + y
                    r[i, a] = self.reward((x, y))
                    for d in range(4):
                        dx, dy = directions[:, d]
                        x_ = max(0, min(self.size - 1, x + dx))
                        y_ = max(0, min(self.size - 1, y + dy))
                        j = x_ * self.size + y_
                        if self.noise is not None:
                            p[i, a, j] += 0.3 * self.noise[x, y, a, d] + 0.7 * int(a == d)
                        else:
                            p[i, a, j] += int(a == d)
        return r, p

    def _collisions(self):
        states = [agent.state for agent in self.agents]
        agent_distances = distance_matrix(states, states)
        n_collisions = np.sum(agent_distances < self.agents_size) - self.n_agents
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

"""
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.size, 0, self.size)

        agents_render = []
        for agent in self.agents:
            agent_render = self.viewer.draw_circle(radius=0.4)
            agent_render.set_color(.8, 0, 0)
            agent_x, agent_y = agent.state
            transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))
            agent_render.add_attr(transform)
            agents_render.append(agent_render)

        landmarks_render = []
        for landmark in self.landmarks:
            landmark_render = self.viewer.draw_circle(radius=0.45)
            landmark_render.set_color(0, 0.8, 0)
            landmark_x, landmark_y = landmark
            transform = rendering.Transform(translation=(landmark_x + 0.5, landmark_y + 0.5))
            landmark_render.add_attr(transform)
            landmarks_render.append(landmark_render)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
"""

class Agent(object):

    def __init__(self, index, size):
        self.index = index
        self.env_size = size
        self.start = self.reset()
        self.state = self.start.copy()

    def reset(self):
        return (np.random.rand(2)*self.env_size).tolist()

if __name__ == '__main__':
    size = 10
    n_agents = 3
    n_landmarks = 3

    env = Space(size, n_agents, n_landmarks, shuffle=False)

    state = env.reset()

    #env.render()

    for i in range(100):
        action = [[random() for j in range(5)] for agent in range(n_agents)]
        #action = [int(random()) for agent in range(n_agents)]
        state, reward, constraint, done = env.step(action)
        print(state)

        #env.render()
