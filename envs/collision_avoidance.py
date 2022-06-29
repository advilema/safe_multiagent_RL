import numpy as np
from scipy.spatial import distance_matrix
from random import random

#TODO: implement matplotlib rendering
class CollisionAvoidance:
    """
    Implement a continuous action and continuous space MDP environment with agents and landmarks.

    The environment is a square of dimension size*size.
    - The state: is represented by the concatenation of the 2D coordinates of the agents, therefore the state dimension is
     2 * n_agents. if the position of the landmarks changes over episodes, their 2D coordinates are appended to the state
     (therefore the state dimensions would be 2*(n_agents + n_landmarks).
    - The action: for each agent is a 2D continuous vector. The first dimension represent an horizontal shift, and the second
    second dimension a vertical shift. The new position for an agent is the sum between its previous position and its
    action vector.
    - The reward: is given by the sum of the distances of the agents from their closest landmark.
    - The constraint: is given by the sum of collisions happening between agents in that time step. The agents are
     circles of ray agents_size and a collision happens if their distance is less than 2*agents_size

    Attributes
    ----------
    size : int
        size of the edge of the square that represent the state space
    n_agents : int
        number of agents
    agents_size : float
        agents are circles of ray agents_size
    agents : Agent
        the class Agent is used to track the position of the agents
    n_landmarks : int
        number of landmarks
    landmarks : list
        concatenation of the 2D coordinates of the landmarks. The position of the landmarks are generated at random
    shuffle : bool
        If False, the starting position of the agents and the position of the landmarks will be the same in every episode,
        state_space will be 2*n_agents
        if True, at the beginning of each episode, the starting position of the agents and the position of the landmarks
        will change (randomly generated). In that case the state space will be 2*(n_agents + n_landmarks)


    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    # TODO: what to put as return type for init? None or CollisionAvoidance? Also putting nothing would work
    def __init__(self, size: int, n_agents: int, n_landmarks=1, shuffle=True, agents_size=0.25, normalize_state=False):
        # check that the data type is the right one
        assert type(size) == int
        assert type(n_agents) == int
        assert type(n_landmarks) == int
        assert type(shuffle) == bool
        assert type(agents_size) == float or type(agents_size) == int

        self.size = size
        self.n_agents = n_agents
        self.agents_size = agents_size
        self.agents = [Agent(i, size, agents_size) for i in range(n_agents)]
        self.n_landmarks = n_landmarks
        self.landmarks = [(np.random.rand(2) * self.size).tolist()]
        self.shuffle = shuffle
        self.action_space = 2  # continuous action space with two dimensions: (up, down), (left, right)
        self.state_space = 2*n_agents # 2D coordinates of each agent
        if self.shuffle:
            # if the position of the landmarks change (shuffle=True) add 2D coordinates of landmarks
            self.state_space += 2*self.n_landmarks
        self.constraint_space = [1] # The only constraint is on the number of collisions in an episode
        self.normalize_state = normalize_state

    def reset(self) -> list:
        for agent in self.agents:
            agent.done = False
        if self.shuffle:
            return self._reset()
        else:
            return self._restart()

    def _reset(self) -> list:
        state = []
        for agent in self.agents:
            agent.state = agent.reset()
            state.append(agent.state.copy())
        self._reset_landmarks()
        [state.append(landmark) for landmark in self.landmarks] #concatenate the state of each agent with the landmarks
        if self.normalize_state:
            state = self._normalize_state(state)
        return state

    def _restart(self) -> list:
        state = []
        for agent in self.agents:
            agent.state = agent.start
            state.append(agent.state.copy())
        if self.normalize_state:
            state = self._normalize_state(state)
        return state

    def _reset_landmarks(self) -> list:
        self.landmarks = [(np.random.rand(2) * self.size).tolist() for _ in range(self.n_landmarks)]

    def transition(self, action: list) -> list:
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

    def reward(self) -> list:
        """Reward depends on the color of the state"""
        rew = - self._agents_landmarks_distances()
        return [rew for i in range(self.n_agents)]

    def constraint(self) -> list:
        return [self._collisions()]

    def check_done(self) -> list:
        return [agent.done for agent in self.agents]

    # TODO: try to write tuple of lists
    def step(self, action) -> tuple:
        state = self.transition(action)
        if self.shuffle:
            [state.append(landmark) for landmark in self.landmarks.copy()]
        reward = self.reward()
        constraint = self.constraint()
        done = self.check_done()
        if self.normalize_state:
            state = self._normalize_state(state)
        return state, reward, constraint, done

    def _collisions(self) -> int:
        states = [agent.state for agent in self.agents if not agent.done]
        if len(states) == 0:
            return 0
        agent_distances = distance_matrix(states, states)
        n_collisions = (np.sum(agent_distances < 2*self.agents_size) - len(states))/2
        return n_collisions

    def _agents_landmarks_distances(self) -> float:
        states = [agent.state for agent in self.agents]
        agents_landmarks_distances = distance_matrix(states, self.landmarks)
        dist = np.sum(np.amin(agents_landmarks_distances, axis=1))
        return dist

    def _normalize_state(self, states: list) -> list:
        return [[s/self.size for s in state] for state in states]


class Agent(object):

    def __init__(self, index: int, size: int, agent_size: float):
        self.index = index
        self.env_size = size
        self.agent_size = agent_size
        self.start = self.reset()
        self.state = self.start.copy()
        self.done = False

    def reset(self) -> list:
        return (np.random.rand(2)*self.env_size).tolist()
        #return [(2*self.index%self.env_size + 1)*self.agent_size, int(self.index/self.env_size)]


# Test the environment here
if __name__ == '__main__':
    size = 3
    n_agents = 3
    n_landmarks = 1

    env = CollisionAvoidance(size, n_agents, n_landmarks, shuffle=False)

    state = env.reset()

    for i in range(100):
        action = [[random() for j in range(env.action_space)] for k in range(n_agents)]
        state, reward, constraint, done = env.step(action)
        print(state, reward, constraint, done)
