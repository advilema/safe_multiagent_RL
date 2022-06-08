import numpy as np
#from gym.envs.classic_control import rendering
from scipy.spatial import distance_matrix
from random import random

class PotentialGrid(object):
    """This class implements a potential grid MDP.
    In the potential grid, all the agents start in the bottom left corner of the grid. Each square in the grid,
    has a score which is minus the L1 distance to the upper right square * number of agents in the given square."""

    def __init__(self, size, n_agents):
        self.size = size
        self.n_agents = n_agents
        self.agents = [Agent(i, size) for i in range(n_agents)]
        self.landmark = [size, size] #upper left square
        self.action_space = 5 #up, down, left, right, stay
        self.state_space = 2*n_agents
        self.constraint_space = [1 for i in range(n_agents)]
        self.viewer = None

    def reset(self):
        for agent in self.agents:
            agent.done = False
        return self._restart()

    def _restart(self):
        state = []
        for agent in self.agents:
            agent.state = agent.start
            state.append(agent.state.copy())
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
            x_ = max(0, min(self.size, x + dx))
            y_ = max(0, min(self.size, y + dy))
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states


    def reward(self):
        """distance from goal * n of agents in the same square"""
        col = self._collisions()
        rew = self._agents_landmark_distances()
        rew = [-r*(c+1) for r,c in zip(rew,col)]
        return rew

    def constraint(self, action):
        travelled_distance = [1, 1, 1, 1, 0]
        con = []
        for a, agent in zip(action, self.agents):
            if agent.done:
                con.append(0)
            else:
                con.append(travelled_distance[a])
        return con

    def check_done(self):
        done = []
        for agent in self.agents:
            if agent.done:
                done.append(True)
                continue
            if np.all(np.array(agent.state) - np.array(self.landmark) == 0):
                agent.done = True
                done.append(True)
            if not agent.done:
                done.append(False)
        return done

    def step(self, action):
        state = self.transition(action)
        reward = self.reward()
        constraint = self.constraint(action)
        done = self.check_done()
        return state, reward, constraint, done

    def _collisions(self):
        states = [agent.state for agent in self.agents]
        agent_distances = distance_matrix(states, states)
        n_collisions = []
        #in the goal and the starting position there is no congestion
        for agent_dist, agent in zip(agent_distances, self.agents):
            if agent.done:
                n_collisions.append(0)
            elif agent.state == agent.start:
                n_collisions.append(0)
            else:
                n_collisions.append(np.sum(agent_dist == 0) - 1)
        return n_collisions

    def _agents_landmark_distances(self):
        states = [agent.state for agent in self.agents]
        dist = []
        for s in states:
            dist.append(np.linalg.norm(np.array(s)-self.landmark, ord=1))
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
        # Draw the grid
        epsilon = 0.5
        unit = 1 - 2 * epsilon / size
        for i in range(self.size+1):
            value = unit*i + epsilon
            self.viewer.draw_line((epsilon, value), (self.size-epsilon, value))
            self.viewer.draw_line((value, epsilon), (value, self.size-epsilon))
        agents_render = []
        collisions = self._collisions()
        for agent, c in zip(self.agents, collisions):
            agent_render = self.viewer.draw_circle(radius=0.05)
            congestion = (c+2)/(self.n_agents+1)
            agent_render.set_color(congestion, 1-congestion, 0)
            agent_x, agent_y = agent.state
            agent_x_rendered = agent_x * unit + epsilon
            agent_y_rendered = agent_y * unit + epsilon
            #transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))
            transform = rendering.Transform(translation=(agent_x_rendered, agent_y_rendered))
            agent_render.add_attr(transform)
            agents_render.append(agent_render)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
"""

class Agent(object):

    def __init__(self, index, size):
        self.index = index
        self.env_size = size
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False

    def reset(self):
        """
        Start from the bottom left square
        """
        return [0,0]

if __name__ == '__main__':
    size = 3
    n_agents = 5

    env = PotentialGrid(size, n_agents)

    state = env.reset()

    #env.render()

    for i in range(100):
        action = [int(random()*5) for agent in range(n_agents)]
        state, reward, constraint, done = env.step(action)
        print(state)

        #env.render()