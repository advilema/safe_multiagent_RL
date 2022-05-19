import numpy as np
import copy 

class BlockerGameEnv():
    def __init__(self, args):
        self.args = args
        self.grid_shape = (4, 7)
        self.blockers = [[0, 2], [4, 6]]
        self._generate_grid()
        
        # for cql reward
        self.delta = -1
        self.tau = args.tau
        self.avg_cost_ubound = 0.3
        
        self.n_agents = 3
        self.observation_spaces = [self.grid_shape[0] * self.grid_shape[1] + 1] * self.n_agents
        self.state_space = self.grid_shape[0] * self.grid_shape[1] * 2
        self.action_spaces = [5] * self.n_agents
        self.n_opponent_actions = 2
    
    def setup(self):
        # generate agents position (maybe randomly)
        #init_pos = np.random.choice(self.grid_shape[1], self.n_agents, replace=False)
        #self.agents_pos_init = [[0, init_pos[i]] for i in range(self.n_agents)]
        
        self.agents_pos_init = [[0, 1], [0, 3], [0, 6]] # static position, applied in CMIX ECML version
        
        self.global_step = 0
        self.returns = None
        self.total_costs = None
        self.peak_violation_sum = None

        return self.n_agents, self.state_space, self.observation_spaces, self.action_spaces, self.n_opponent_actions

    def init_cql(self):
        self.delta = -1

    def set_logger(self, logdir):
        self.logdir = logdir
        self.cost_file = open("{}/cost.log".format(logdir), "w", 1)
        self.return_file = open("{}/return.log".format(logdir), "w", 1)
        self.peak_violation_file = open("{}/peak_violation.log".format(logdir), "w", 1)

    def set_scheme(self, scheme):
        self.scheme = scheme
    
    def get_rlinfo(self):
        return self.n_agents, self.state_space, self.observation_spaces, self.action_spaces, self.n_opponent_actions
    

    def _generate_grid(self):
        self.grid = np.zeros(self.grid_shape) # trap or other element
        self.traps = [[1, 3], [1, 6]] # traps
        
        self.cost_matrix = np.array([
            [0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
            [1.0, 1.0, 0.1, 1.0, 0.1, 0.1, 1.0],
            [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]])

        print("cost_matrix:", self.cost_matrix)

    def get_grid_obs(self):
        grid_obs = np.ones(self.grid_shape)
        for i in range(self.n_agents):
            grid_obs[self.agents_pos[i][0], self.agents_pos[i][1]] = 0 
        
        return grid_obs

    def reset(self):
        self.win_flag = False
        self.global_step = 0
        
        if self.returns != None:
            avg_cost = self.total_costs / self.returns
            print(self.returns, file=self.return_file)
            print(avg_cost, file=self.cost_file)
            print(self.peak_violation_sum, file=self.peak_violation_file)
        self.returns = 0
        self.total_costs = 0
        self.peak_violation_sum = 0

        self.agents_pos = copy.deepcopy(self.agents_pos_init)
        
        grid_obs = self.get_grid_obs().flatten().tolist()
        cost_obs = self.cost_matrix.flatten().tolist()
        obs_total = grid_obs + cost_obs
        
        obses = []
        for i in range(self.n_agents):
            index = [0] * self.grid_shape[0] * self.grid_shape[1]
            index[self.agents_pos[i][0] * self.grid_shape[1] + self.agents_pos[i][1]] = 1
            obses.append(np.array(index + [self.global_step]))
            
        
        
        state = np.array(obs_total)

        return state, obses
    
    '''
        check whether a cell is vacant(can move in)
    ''' 
    def is_vacant(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_shape[0]:
            return False
        if y < 0 or y >= self.grid_shape[1]:
            return False
        for blocker in self.blockers:
            if x == self.grid_shape[0] - 1 and y >= blocker[0] and y <= blocker[1]:
                return False
        if pos in self.agents_pos:
            return False

        return True

    def is_safe(self, pos):
        x, y = pos
        if pos in self.traps:
            return False
        
        return True
    
    '''
        Blockers moving policy
    '''
    def move_blockers(self):
        # for two blocker 
        if [self.grid_shape[0] - 2, 0] in self.agents_pos:
            self.blockers[0] = [0, 2]
            if [self.grid_shape[0] - 2, self.grid_shape[1] - 1] in self.agents_pos:
                self.blockers[1] = [4, 6]
            else:
                self.blockers[1] = [3, 5]
        else:
            self.blockers = [[1, 3], [4, 6]]
        

    def step(self, actions):
        extra_reward = 0.
        
        costs = [0.] * self.n_agents
        peak_violation = 0
        if self.win_flag:
            done_mask = 0
        else:
            done_mask = 1
            for i in range(self.n_agents):
                next_pos = copy.copy(self.agents_pos[i])
                if actions[i] == 1:
                    next_pos[0] += 1
                elif actions[i] == 2:
                    next_pos[0] -= 1
                elif actions[i] == 3:
                    next_pos[1] += 1
                elif actions[i] == 4:
                    next_pos[1] -= 1
                elif actions[i] == 0:
                    pass
                
                if actions[i] != 0 and self.is_vacant(next_pos):
                    self.agents_pos[i] = next_pos
                    costs[i] = self.cost_matrix[next_pos[0], next_pos[1]]
                    if actions[i] == 1:
                        extra_reward += 1 / (self.n_agents + 1)
                    elif actions[i] == 2:
                        extra_reward -= 1/ (self.n_agents + 1) 
                if actions[i] != 0 and not self.is_safe(next_pos):
                    peak_violation += 1

                if self.agents_pos[i][0] == self.grid_shape[0] - 1:
                    self.win_flag = True
                    print("Win !!!")
            self.move_blockers()
        # cql
        if not self.win_flag:
            #reward = -1
            reward = -1 + extra_reward
        else:
            reward = 0 
        
        avg_cost = sum(costs) / self.n_agents
        if done_mask == 1:
            self.total_costs += avg_cost 
            self.returns += 1
            self.peak_violation_sum += peak_violation
        
        
        if self.scheme == "simple":
            global_reward = reward 
        else:
            global_reward = [reward - self.delta - peak_violation, self.avg_cost_ubound - avg_cost - peak_violation]
            if done_mask == 1:
                self.delta = self.tau * self.delta + (1 - self.tau) * reward
        local_rewards = [global_reward] * self.n_agents
        
        # update state observation
        self.global_step += 1
        
        if self.win_flag:
            self.agents_pos = copy.deepcopy(self.agents_pos_init) # set final state tor start state, for CQL assumption

        grid_obs = self.get_grid_obs().flatten().tolist()
        cost_obs = self.cost_matrix.flatten().tolist()
        obs_total = grid_obs + cost_obs
         
       
        obses = []
        for i in range(self.n_agents):
            index = [0] * self.grid_shape[0] * self.grid_shape[1]
            index[self.agents_pos[i][0] * self.grid_shape[1] + self.agents_pos[i][1]] = 1
            obses.append(np.array(index + [self.global_step]))

        state = np.array(obs_total)
        
        return state, obses, local_rewards, global_reward, done_mask, reward, avg_cost
