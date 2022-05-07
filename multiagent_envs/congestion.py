import numpy as np
from gym.envs.classic_control import rendering
from scipy.spatial import distance_matrix
from random import random


HOURLY_COMPENSATION = 30.0 #30 euro per hour
AVERAGE_RIDE_COMPENSATION = 7.5
AVERAGE_RIDE_COST = 4.0
CONGESTION_COST = 2.0


class Congestion(object):
    # TODO add the noise
    """This class implements a potential grid MDP.
    In the potential grid, all the agents start in the bottom left corner of the grid. Each square in the grid,
    has a score which is minus the L1 distance to the upper right square * number of agents in the given square."""

    def __init__(self, size, n_agents, noise=0.1, shuffle=False):
        self.size = size
        self.n_agents = n_agents
        self.agents = [Agent(i, size) for i in range(n_agents)]
        self.landmark = [size, size]  # upper left square
        self.action_space = 5  # up, down, left, right, stay
        self.state_space = 2 * n_agents
        self.constraint_space = [1]
        #self.demand_rate = np.random.rand(size+1, size+1)*8 + 2 #range between 2 and 10 people per area per hour
        self.demand_rate = np.array([[2,2,4,4],[3,6,10,5],[3,8,3,4],[4,6,7,8]])
        self.viewer = None
        assert 0 <= noise <= 1
        self.noise = noise
        self.shuffle = shuffle

    def reset(self):
        for agent in self.agents:
            agent.done = False
        return self._restart()

    def _restart(self):
        state = []
        for agent in self.agents:
            if self.shuffle:
                agent.state = agent.reset()
            else:
                agent.state = agent.start
            state.append(agent.state.copy())
        return state

    def transition(self, action):
        """Transition p(s'|s,a).
        action = 0,1,2,3 => go right, down, left, up with 90% of probability and in one of the other directions at random with
        10 % probability
        action = 5 => wait for a new rider, you can ends up in the same state or one of the neightbouring vertices
        with equal probability"""
        directions = np.array([[1, -1, 0, 0, 0], [0, 0, -1, 1, 0]])
        states = []

        for i, agent in enumerate(self.agents):
            if agent.done:
                states.append(agent.state.copy())
                continue
            x, y = agent.state

            if random() < 1 - self.noise:
                move = action[i]
            else:
                move = int(random() * 5)

            dx, dy = directions[:, move]
            x_ = max(0, min(self.size, x + dx))
            y_ = max(0, min(self.size, y + dy))
            agent.edge = [x,y,x_,y_]
            agent.state = [x_, y_]
            states.append(agent.state.copy())
        return states

    def reward(self, action):
        """distance from goal * n of agents in the same square"""
        congestions = self._congestions(action)
        reward = []

        for act, con, agent in zip(action, congestions, self.agents):
            if act < 4:
                rew = - AVERAGE_RIDE_COST - con*CONGESTION_COST
            else:
                rew = - HOURLY_COMPENSATION*(con+1)/self.demand_rate[int(agent.state[0]), int(agent.state[1])] + \
                    AVERAGE_RIDE_COMPENSATION - AVERAGE_RIDE_COST

            reward.append(rew)
        return reward

    #requires one third of the agents to be in node (0,0)
    def constraint(self, action):
        min_agents_in_node = self.n_agents//3
        agents_in_node = 0
        for agent in self.agents:
            if agent.state == [0,0]:
                agents_in_node += 1
        constr = [max(0, min_agents_in_node - agents_in_node)]
        return constr
        #return [0 for agent in range(self.n_agents)]

    def check_done(self):
        return [False for _ in range(self.n_agents)]

    def step(self, action):
        state = self.transition(action)
        reward = self.reward(action)
        constraint = self.constraint(action)
        done = self.check_done()
        return state, reward, constraint, done

    def _congestions(self, action):
        congestions = [0 for _ in range(self.n_agents)]

        for i, agent in enumerate(self.agents):
            if congestions[i]: #if already checked congestions for that edge continue
                continue
            if action[i] < 4:
                edge = agent.edge
                agents_sharing_edge = [i]
                for j, agent2 in enumerate(self.agents[i+1:]):
                    if action[i+j+1] < 5:
                        if agent2.edge == edge:
                            agents_sharing_edge.append(j+i+1)
                for a in agents_sharing_edge:
                    congestions[a] = len(agents_sharing_edge) - 1
            else:
                state = agent.edge[:2]
                agents_sharing_state = [i]
                for j, agent2 in enumerate(self.agents[i+1:]):
                    if action[i+j+1] == 5:
                        if agent2.edge[:2] == state:
                            agents_sharing_state.append(j+i+1)
                for a in agents_sharing_state:
                    congestions[a] = len(agents_sharing_state) - 1
        return congestions

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

    def normalize_state(self, states):
        return [[s / self.size for s in state] for state in states]

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

        # Draw the grid
        epsilon = 0.5
        unit = 1 - 2 * epsilon / self.size
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


class Agent(object):

    def __init__(self, index, size):
        self.index = index
        self.env_size = size
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False
        self.edge = [0,0,0,0] #indicates the edge that the agent is traversing in the current time step (the coordinates of the 2 vertices)

    def reset(self):
        """
        Start from the bottom left square
        """
        if self.index == 0:
            return np.array([0,0])
        return np.floor(np.random.rand(2)*self.env_size).tolist()


if __name__ == '__main__':
    size = 2
    n_agents = 4

    env = Congestion(size, n_agents)

    state = env.reset()

    env.render()

    for i in range(100):
        action = [int(random() * 5) for agent in range(n_agents)]
        state, reward, constraint, done = env.step(action)
        print(state, constraint, reward)

        env.render()
