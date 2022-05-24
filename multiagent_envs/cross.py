import numpy as np
#from gym.envs.classic_control import rendering
from scipy.spatial import distance_matrix
from random import random

class Cross(object):
    """This class implements a grid MDP."""

    def __init__(self, agents_size=0.5):
        self.cross = [[0,0,1,1,0,0],
                      [0,0,1,1,0,0],
                      [1,1,1,1,1,1],
                      [1,1,1,1,1,1],
                      [0,0,1,1,0,0],
                      [0,0,1,1,0,0]]
        self.size = 6
        self.n_agents = 4
        self.agents_size = agents_size
        self.agents = [Agent(i, self.size, agents_size) for i in range(self.n_agents)]
        self.n_landmarks = 2
        self.start_landmarks = [[3, 6], [0, 3]]
        self.landmarks = self.start_landmarks.copy()
        self.action_space = 2  # (up, down), (left, right)
        self.state_space = 2*self.n_agents
        self.viewer = None
        self.constraint_space = [1]

    def reset(self):
        for agent in self.agents:
            agent.done = False
        state = self._restart()
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
        return [agent.state for agent in self.agents]

    def step(self, action):
        state = self.transition(action)
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
        n_collisions = (np.sum(agent_distances < 2*self.agents_size) - len(states))/2
        return n_collisions

    def _agents_landmarks_distances(self):
        states = [agent.state for agent in self.agents]
        agents_landmarks_distances0 = distance_matrix(states[:2], [self.landmarks[0]])
        dist0 = np.sum(np.amin(agents_landmarks_distances0, axis=1))
        agents_landmarks_distances1 = distance_matrix(states[2:4], [self.landmarks[1]])
        dist1 = np.sum(np.amin(agents_landmarks_distances0, axis=1))
        return dist0+dist1

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
            agent_render = self.viewer.draw_circle(radius=self.agents_size)
            agent_render.set_color(.8, 0, 0)
            agent_x, agent_y = agent.state
            transform = rendering.Transform(translation=(agent_x, agent_y))
            agent_render.add_attr(transform)
            agents_render.append(agent_render)

        landmarks_render = []
        for landmark in self.landmarks:
            landmark_render = self.viewer.draw_circle(radius=0.45)
            landmark_render.set_color(0, 0.8, 0)
            landmark_x, landmark_y = landmark
            transform = rendering.Transform(translation=(landmark_x, landmark_y))
            landmark_render.add_attr(transform)
            landmarks_render.append(landmark_render)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
"""


class Agent(object):

    def __init__(self, index, size, agent_size):
        self.index = index
        self.env_size = size
        self.agent_size = agent_size
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False

    def reset(self):
        #return (np.random.rand(2)*self.env_size).tolist()
        if self.index == 0:
            return [2.5, 0]
        elif self.index == 1:
            return [3.5, 0]
        elif self.index == 2:
            return [6, 2.5]
        elif self.index == 3:
            return [6, 3.5]

if __name__ == '__main__':
    size = 3
    n_agents = 3
    n_landmarks = 1

    env = Cross(size, n_agents, n_landmarks, shuffle=False)

    state = env.reset()

    #env.render()

    for i in range(100):
        action = [[random() for j in range(env.action_space)] for k in range(n_agents)]
        #action = [int(random()) for agent in range(n_agents)]
        state, reward, constraint, done = env.step(action)
        print(state, reward, constraint, done)

        #env.render()
