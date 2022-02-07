import numpy as np
from gym.envs.classic_control import rendering
from scipy.spatial import distance_matrix
from random import random

class Explore(object):
    """This class implements a grid MDP."""

    def __init__(self, size, n_agents, shuffle=False, agents_size=0.5, fieldview_size=None, weights=None):
        self.size = size
        self.n_agents = n_agents
        self.agents_size = agents_size
        if fieldview_size is None:
            self.fieldview_size = size/n_agents
        else:
            self.fieldview_size = fieldview_size
        self.agents = [Agent(i, size) for i in range(n_agents)]
        self.action_space = 5 #up, down, left, right, stay
        self.state_space = 2*self.n_agents
        self.constraint_space = [1 for i in range(n_agents)]
        self.shuffle = shuffle
        self.viewer = None
        self.weights = weights

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
        #[pos.extend(landmark) for landmark in self.landmarks for pos in state] #concatenate the state of each agent with the landmarks
        #state = self.normalize_state(state)
        return state

    def _restart(self):
        state = []
        for agent in self.agents:
            agent.state = agent.start
            state.append(agent.state.copy())
        #[pos.extend(landmark) for landmark in self.landmarks for pos in state] #concatenate the state of each agent with the landmarks
        #state = self.normalize_state(state)
        return state

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
            x_ = max(0, min(self.size - 1, x + dx))
            y_ = max(0, min(self.size - 1, y + dy))
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states


    def reward(self):
        states = [agent.state for agent in self.agents]
        agents_distances = distance_matrix(states, states)
        rew = 0
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if self.fieldview_size - agents_distances[i,j] > 0:
                    rew -= (self.fieldview_size - agents_distances[i,j])**2
        reward = [rew for i in range(self.n_agents)]

        if self.weights is not None:
            reward = [rew*w for rew, w in zip(reward, self.weights)]

        return reward

    def constraint(self, action):
        travelled_distance = [1,1,1,1,0]
        con = []
        for a in action:
            con.append(travelled_distance[a])
        return con

    def check_done(self):
        return [False for i in range(self.n_agents)]

    def step(self, action):
        state = self.transition(action)
        reward = self.reward()
        constraint = self.constraint(action)
        done = self.check_done()
        #state = self.normalize_state(state)
        return state, reward, constraint, done

    def _agents_landmarks_distances(self):
        states = [agent.state for agent in self.agents]
        agents_landmarks_distances = distance_matrix(states, self.landmarks)
        dist = -1*np.sum(np.where(agents_landmarks_distances==0))
        #dist += -1*np.sum(np.where(agents_landmarks_distances>0))
        dist += np.sum(np.amin(agents_landmarks_distances, axis=1))
        return dist

    def normalize_state(self, states):
        return [[s/self.size for s in state] for state in states]

    def _coupleToInt(self, x, y):
        return y + x * self.size

    def _intToCouple(self, n):
        return int(np.floor(n / self.size)), int(n % self.size)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.size, 0, self.size)

        fieldviews_render = []
        agents_render = []

        for agent in self.agents:
            agent_x, agent_y = agent.state
            transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))

            fieldview_render = self.viewer.draw_circle(radius=self.fieldview_size)
            fieldview_render.set_color(.5, 0, 0)
            fieldview_render.add_attr(transform)
            fieldviews_render.append(fieldview_render)

        for agent in self.agents:
            agent_x, agent_y = agent.state
            transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))

            agent_render = self.viewer.draw_circle(radius=0.4)
            agent_render.set_color(.8, 0, 0)
            agent_render.add_attr(transform)
            agents_render.append(agent_render)

        # Draw the grid
        for i in range(self.size):
            self.viewer.draw_line((0, i), (self.size, i))
            self.viewer.draw_line((i, 0), (i, self.size))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class Agent(object):

    def __init__(self, index, size):
        self.index = index
        self.env_size = size
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False

    def reset(self):
        return np.floor(np.random.rand(2)*self.env_size).tolist()

if __name__ == '__main__':
    size = 10
    n_agents = 3
    n_landmarks = 3

    env = Explore(size, n_agents, shuffle=False, fieldview_size=4)

    state = env.reset()

    env.render()

    for i in range(100):
        action = [int(random()*env.action_space) for i in range(n_agents)]
        state, reward, constraint, done = env.step(action)

        env.render()
