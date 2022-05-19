import numpy as np
#from gym.envs.classic_control import rendering
from scipy.spatial import distance_matrix
from random import random
import time


class ExploreContinuous(object):
    """This class implements a grid MDP."""

    def __init__(self, size, n_agents, shuffle=False, agents_size=0.5, fieldview_size=None, weights=None, coarseness=None):
        self.size = size
        self.n_agents = n_agents
        self.agents_size = agents_size
        if fieldview_size is None:
            self.fieldview_size = size / (np.sqrt(n_agents))
        else:
            self.fieldview_size = fieldview_size
        self.agents = [Agent(i, size, continuous=True) for i in range(n_agents)]
        self.action_space = 2  # (up, down), (left, right)
        self.state_space = 2 * self.n_agents
        self.constraint_space = [1 for i in range(n_agents)]
        self.shuffle = shuffle
        self.viewer = None
        self.weights = weights
        self.coarseness = coarseness

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
        # [pos.extend(landmark) for landmark in self.landmarks for pos in state] #concatenate the state of each agent with the landmarks
        # state = self.normalize_state(state)
        return state

    def _restart(self):
        state = []
        for agent in self.agents:
            agent.state = agent.start
            state.append(agent.state.copy())
        # [pos.extend(landmark) for landmark in self.landmarks for pos in state] #concatenate the state of each agent with the landmarks
        # state = self.normalize_state(state)
        return state

    def transition(self, action):
        """Transition p(s'|s,a)."""
        states = []

        for agent, a in zip(self.agents, action):
            if agent.done:
                states.append(agent.state.copy())
                continue
            x, y = agent.state
            dx, dy = np.squeeze(a)
            if self.coarseness is not None:
                norm = np.sqrt(dx**2+dy**2)
                max_norm = np.sqrt(2)*self.size/self.coarseness
                if np.sqrt(norm) > max_norm:
                    dx = (dx / norm) * max_norm
                    dy = (dy / norm) * max_norm
            x_ = max(0, min(self.size, x + dx))
            y_ = max(0, min(self.size, y + dy))
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states

    def reward(self):
        states = [agent.state for agent in self.agents]
        agents_distances = distance_matrix(states, states)
        rew = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.fieldview_size - agents_distances[i, j] > 0:
                    rew -= (self.fieldview_size - agents_distances[i, j]) ** 2
        reward = [rew for i in range(self.n_agents)]

        if self.weights is not None:
            reward = [rew * w for rew, w in zip(reward, self.weights)]

        return reward

    def constraint(self, action):
        con = []
        for a in action:
            con.append(np.linalg.norm(np.squeeze(a)))
        return con

    def check_done(self):
        return [False for i in range(self.n_agents)]

    def step(self, action):
        state = self.transition(action)
        reward = self.reward()
        constraint = self.constraint(action)
        done = self.check_done()
        # state = self.normalize_state(state)
        return state, reward, constraint, done

    def _agents_landmarks_distances(self):
        states = [agent.state for agent in self.agents]
        agents_landmarks_distances = distance_matrix(states, self.landmarks)
        dist = -1 * np.sum(np.where(agents_landmarks_distances == 0))
        # dist += -1*np.sum(np.where(agents_landmarks_distances>0))
        dist += np.sum(np.amin(agents_landmarks_distances, axis=1))
        return dist

    def normalize_state(self, states):
        return [[s / self.size for s in state] for state in states]

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
            self.viewer = rendering.Viewer(650, 650)
            self.viewer.set_bounds(0, self.size, 0, self.size)

        fieldviews_render = []
        agents_render = []

        colours = self.make_colours()
        for i, agent in enumerate(self.agents):
            agent_x, agent_y = agent.state
            transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))

            fieldview_render = self.viewer.draw_circle(radius=self.fieldview_size, filled=False, res=100)
            fieldview_render.set_color(*colours[i])
            fieldview_render.add_attr(transform)
            fieldviews_render.append(fieldview_render)

        for i, agent in enumerate(self.agents):
            agent_x, agent_y = agent.state
            transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))

            agent_render = self.viewer.draw_circle(radius=0.4)
            agent_render.set_color(*colours[i])
            agent_render.add_attr(transform)
            agents_render.append(agent_render)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def make_colours(self):
        return np.random.rand(self.n_agents, 3).tolist()
"""


class ExploreDiscrete(ExploreContinuous):
    def __init__(self, size, n_agents, shuffle=False, agents_size=0.5, fieldview_size=None, weights=None):
        super().__init__(size, n_agents, agents_size=agents_size, fieldview_size=fieldview_size,
                         weights=weights)
        self.agents = [Agent(i, size, continuous=False) for i in range(n_agents)]
        self.action_space = 5 # up, down, left, right, stay
        self.shuffle = shuffle

    def transition(self, action):
        """Transition p(s'|s,a)."""
        directions = np.array([[1, -1, 0, 0, 0], [0, 0, -1, 1, 0]])
        states = []

        for i, agent in enumerate(self.agents):
            if agent.done:
                states.append(agent.state.copy())
                continue
            x, y = agent.state
            dx, dy = directions[:, action[i]]
            x_ = max(0, min(self.size, x + dx))
            y_ = max(0, min(self.size, y + dy))
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states

    def constraint(self, action):
        travelled_distance = [1, 1, 1, 1, 0]
        con = []
        for a in action:
            con.append(travelled_distance[a])
        return con

"""
    def render(self, mode='human', close=False):
        super().render(mode='human', close=False)
        # Draw the grid
        for i in range(self.size):
            self.viewer.draw_line((0, i), (self.size, i))
            self.viewer.draw_line((i, 0), (i, self.size))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
"""


class ExploreDiscretized(ExploreContinuous):
    def __init__(self, size, n_agents, coarseness=20, shuffle=False, agents_size=0.5, fieldview_size=None, weights=None):
        super().__init__(size, n_agents, shuffle=shuffle, agents_size=agents_size, fieldview_size=fieldview_size,
                         weights=weights)
        self.coarseness = coarseness
        self.zoom_fac = self.coarseness/self.size
        self.agents = [Agent(i, size, zoom_fac=self.zoom_fac) for i in range(n_agents)]
        self.action_space = 9 # up, down, left, right, up-left, up-right, down-left, down-right, stay

    def transition(self, action):
        """Transition p(s'|s,a)."""
        directions = np.array([[1, -1, 0, 0, 1, 1, -1, -1, 0], [0, 0, -1, 1, 1, -1, 1, -1, 0]])
        states = []

        for i, agent in enumerate(self.agents):
            if agent.done:
                states.append(agent.state.copy())
                continue
            x, y = np.array(agent.state)*self.zoom_fac
            dx, dy = directions[:, action[i]]
            x_ = max(0, min(self.size*self.zoom_fac, x + dx))/self.zoom_fac
            y_ = max(0, min(self.size*self.zoom_fac, y + dy))/self.zoom_fac
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states

    def constraint(self, action):
        travelled_distance = np.array([1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2),  0])*(self.size/self.coarseness)
        con = []
        for a in action:
            con.append(travelled_distance[a])
        return con

"""
    def render(self, mode='human', close=False):
        super().render(mode='human', close=False)
        # Draw the grid
        for i in range(self.size):
            self.viewer.draw_line((0, i), (self.size, i))
            self.viewer.draw_line((i, 0), (i, self.size))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
"""


class Agent(object):

    def __init__(self, index, size, zoom_fac=None, continuous=True):
        self.index = index
        self.env_size = size
        self.continuous = continuous
        self.zoom_fac = zoom_fac
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False

    def reset(self):
        state = (np.random.rand(2) * self.env_size).tolist()
        if not self.continuous:
            state = np.floor(state).tolist()
        if self.zoom_fac is not None:
            state = np.floor(np.array(state)*self.zoom_fac)/self.zoom_fac
            state = state.tolist()
        return state


if __name__ == '__main__':
    size = 10
    n_agents = 10
    np.random.seed(9)

    env = ExploreContinuous(size, n_agents, shuffle=False)

    state = env.reset()

    #env.render()

    for i in range(100):
        action = [[random() for j in range(env.action_space)] for k in range(n_agents)]
        state, reward, constraint, done = env.step(action)

        #env.render()
        #time.sleep(100)
