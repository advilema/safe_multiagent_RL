import numpy as np
import math
import copy

class VNEnv():
    def __init__(self, args):
        self.args = args
        
        # network parameter setting
        self.n_cell = 3
        self.cell_distr_prob = [1/3, 1/3, 1/3]
        self.cell_link_set = [[0,1], [1,2], [0,2]]
        self.resource_list = [1e6, 2e6, 3e6]
        self.n_cluster = 6 
        self.cluster_intra_ratio = 0.5 
        self.num_range = [5, 10] # vehicle number range
        self.num_relay = 4
        self.relay_method = "SINR"
        self.SINR_range = [0, 30] # dB
        self.V2I_latency_para = [10, 3.5, 0.5] 
        self.V2V_latency_para = [10, 3.0, 0.5]
        self.latency_ubound = 100 
        self.latency_avebound = 60 
        

        self.discount_factor = self.args.gamma 
        self.delta = 1 # CQL-mixer based algorithm, altually this is (1-\gamma)\delta in cql-paper
        self.utility_scale = None
        self.scheme = "basic"
        

        self._generate_network()
        

        self.n_agents = 0
        for i in range(self.n_cluster):
            self.n_agents += self.cluster_list[i].n_vehicle
        self.observation_spaces = []
        for i in range(self.n_cluster):
            self.observation_spaces += self.cluster_list[i].get_observation_spaces()
        self.state_space = sum(self.observation_spaces)
        self.action_spaces = []
        for i in range(self.n_cluster):
            self.action_spaces += self.cluster_list[i].get_action_spaces()
        self.n_opponent_actions = 2
        
        self.tau = args.tau
        self.global_step = 0
    
    def setup(self):
        for cluster in self.cluster_list:
            cluster.setup()
        
        actions = [-2] * self.n_vehicle_net
        # TO DO: remove the step() call here, use a better implementation
        temp = self.scheme
        self.scheme = "basic"
        ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = self.step(actions)
        self.utility_scale = utility_sum
        self.scheme = temp

        return self.n_agents, self.state_space, self.observation_spaces, self.action_spaces, self.n_opponent_actions
        
    def init_cql(self):
        self.delta = 1

    def set_logger(self, logdir):
        self.logdir = logdir
        self.utility_sum_file = open("%s/utility_sum.log" % (logdir), "w", 1)
        self.ave_latency_file = open("%s/ave_latency.log" % (logdir), "w", 1)
        self.peak_violation_file = open("%s/peak_violation.log" % (logdir), "w", 1)
   
    def set_scheme(self, scheme):
        self.scheme = scheme
    
    def get_rlinfo(self):
        return self.n_agents, self.state_space, self.observation_spaces, self.action_spaces, self.n_opponent_actions
    
    def reset(self):
        self.global_step = 0
        obses = []
        for cluster in self.cluster_list:
            obses_tmp = cluster.reset()
            obses += obses_tmp
        state = []
        for obs in obses:
            state += obs
        state = np.array(state)
        obses = [np.array(obs) for obs in obses]
        return state, obses
    
    def step(self, actions):
        utility_sum = 0.0
        utility_list = []
        latency_sum = 0
        peak_violation_sum = 0
        cell_shannon_rate_list = [0.0] * self.n_cell
        cell_serve_num_list = [0] * self.n_cell
        cell_serve_num_cluster_list = [] 
        next_state_list = []
        candidate_relay_cellid_list = []
        violation_indicator_list = []
        
        action_list = []
        n_vehicle_tmp = 0
        for i in range(self.n_cluster):
            action_list.append(actions[n_vehicle_tmp : n_vehicle_tmp + self.cluster_list[i].n_vehicle])
            n_vehicle_tmp += self.cluster_list[i].n_vehicle

        for i in range(self.n_cluster):
            cluster = self.cluster_list[i]
            n_vehicle = cluster.n_vehicle
            next_state, candidate_relay_cellid, utility, ave_latency, peak_violation, cell_serve_num, cell_shannon_rate, violation_indicator = cluster.step(action_list[i])
            next_state_list += next_state # state observations has shape [[...], [...], ...] rather than gather the state of each cluster in a unique list 
            candidate_relay_cellid_list += candidate_relay_cellid
            utility_sum += utility
            utility_list.append(utility)
            latency_sum += ave_latency * n_vehicle
            peak_violation_sum += peak_violation
            violation_indicator_list.append(violation_indicator)

            cell_serve_num_cluster_list.append(cell_serve_num)
            if cluster.cluster_type_id == 0:
                cell_shannon_rate_list[cluster.cell_id] += cell_shannon_rate[0]
                cell_serve_num_list[cluster.cell_id] += n_vehicle
            elif cluster.cluster_type_id == 1:
                # left cell
                cell_id = cluster.cell_id[0]
                cell_shannon_rate_list[cell_id] += cell_shannon_rate[0]
                cell_serve_num_list[cell_id] += cell_serve_num[0]
                # right cell
                cell_id = cluster.cell_id[1]
                cell_shannon_rate_list[cell_id] += cell_shannon_rate[1]
                cell_serve_num_list[cell_id] += cell_serve_num[1]
        
        ave_latency_whole = latency_sum/self.n_vehicle_net
        ave_rate, resource_allocation_list, cell_tail_utility_list = self.compute_total_rate(cell_shannon_rate_list, cell_serve_num_list)
        utility_sum += sum(cell_tail_utility_list) # derived by utility_sum = \sum_b(n_b,v * log W_b,v + \sum_v \in N_b log shannon-rate_n,v) 
        self.cell_serve_num_list = cell_serve_num_list
        self.action_list = action_list
        self.resource_allocation_list = resource_allocation_list
        for i in range(self.n_vehicle_net):
            candidate_resource = [resource_allocation_list[j] for j in candidate_relay_cellid_list[i]]
            # state normalization
            max_candidate_resource = max(candidate_resource)
            for j in range(len(candidate_resource)):
                candidate_resource[j] /= max_candidate_resource
            
            next_state_list[i] += candidate_resource
        
        self.global_step += 1
        
        if self.scheme == "basic":
            # for non-RL based algorithms
            return ave_rate, utility_sum, ave_latency_whole, peak_violation_sum
        elif self.scheme == "cql":
            # for CQL-(IQL|VDN|QMIX) algorithms
            penalty_sum = peak_violation_sum
            
            obses = []
            state = []
            for obs in next_state_list:
                obses.append(np.array(obs))
                state += obs
                
            state = np.array(state)

            global_reward = [utility_sum / self.utility_scale, (self.latency_avebound - ave_latency_whole) / self.latency_avebound] #normalized reward 

            # update average avg global rewards
            self.delta = self.tau * self.delta + (1 - self.tau) * global_reward[0] 
            global_reward[0] -= self.delta
            
            for i in range(self.n_opponent_actions):
                global_reward[i] -= penalty_sum
            
            local_rewards = [global_reward] * self.n_agents
        elif self.scheme == "simple":
            # for simple qmix algorithm
            obses = []
            state = []
            for obs in next_state_list:
                obses.append(np.array(obs))
                state += obs
            state = np.array(state)
            global_reward = utility_sum / self.utility_scale 
            local_rewards = self.get_per_rate()
        
        done_mask = 1 
        
        print(utility_sum, file=self.utility_sum_file)
        print(ave_latency_whole, file=self.ave_latency_file)
        print(peak_violation_sum, file=self.peak_violation_file)
        return state, obses, local_rewards, global_reward, done_mask

    def get_per_rate(self):
        res = []
        for i in range(self.n_cluster):
            cluster = self.cluster_list[i]
            per_v_shannon_rate, per_v_cell_id_list = cluster.get_per_rate(self.action_list[i])
            for j in range(cluster.n_vehicle):
                if i < self.n_intra_cluster:
                    cell_id = cluster.cell_id
                else:
                    cell_id = cluster.cell_id[per_v_cell_id_list[j]]
                res.append(per_v_shannon_rate[j]*self.resource_allocation_list[cell_id])
        return res

    def _generate_network(self):
        self.n_intra_cluster = int(self.n_cluster * self.cluster_intra_ratio)
        self.n_inter_cluster = self.n_cluster - self.n_intra_cluster
        self.cluster_list = []
        self.intra_cell_id_list = []
        self.inter_cell_link_list = [] #the linked cells id to the cluster
        self.n_vehicle_net = 0
        self.n_intra_vehicle = 0
        self.n_inter_vehicle = 0
        
        for _ in range(self.n_intra_cluster):
            cell_id = self.n_cell - 1
            tmp = 0.0
            randid = np.random.rand()
            for i in range(self.n_cell):
                tmp += self.cell_distr_prob[i]
                if randid <= tmp:
                    cell_id = i
                    break

            self.intra_cell_id_list.append(cell_id)
            cluster = Cluster(self.num_range, self.SINR_range, self.V2I_latency_para, self.V2V_latency_para, self.num_relay, self.relay_method, self.latency_ubound, self.latency_avebound, cell_id, cluster_type = "intra")
            self.n_vehicle_net += cluster.n_vehicle
            self.n_intra_vehicle += cluster.n_vehicle
            self.cluster_list.append(cluster)
        for _ in range(self.n_inter_cluster):
            link_id = np.random.randint(0, len(self.cell_link_set))
            self.inter_cell_link_list.append(link_id)
            cluster = Cluster(self.num_range, self.SINR_range, self.V2I_latency_para, self.V2V_latency_para, self.num_relay, self.relay_method, self.latency_ubound, self.latency_avebound, self.cell_link_set[link_id], cluster_type = "inter")
            self.n_vehicle_net += cluster.n_vehicle
            self.n_inter_vehicle += cluster.n_vehicle
            self.cluster_list.append(cluster)
    
    '''
    compute the cell serve number except the cluster_id's, only for DCRA algorithm
    '''
    def get_mcb(self, cluster_id, cell_position):
        if cluster_id < self.n_intra_cluster:
            raise Exception("Expect an inter cluster id")
        if cell_position == "left":
            cell_id = self.cluster_list[cluster_id].cell_id[0]
            return self.cell_serve_num_list[cell_id] - self.action_list[cluster_id].count(-3)
        elif cell_position == "right":
            cell_id = self.cluster_list[cluster_id].cell_id[1]
            return self.cell_serve_num_list[cell_id] - self.action_list[cluster_id].count(-4)
        else:
            raise Exception("cell position error")

    '''
    assume the vehicles select the best relay in left/right, return W_l * shannon rate
    '''
    def get_rcb(self, cluster_id, cell_position):
        if cluster_id < self.n_intra_cluster:
            raise Exception("Expect an inter cluster id")
        if cell_position == "left":
            cell_id = self.cluster_list[cluster_id].cell_id[0]
            return self.resource_list[cell_id]*self.cluster_list[cluster_id].max_relay_rate_left
        elif cell_position == "right":
            cell_id = self.cluster_list[cluster_id].cell_id[1]
            return self.resource_list[cell_id]*self.cluster_list[cluster_id].max_relay_rate_right
        else:
            raise Exception("cell position error")
    
    
    
    
    def compute_total_rate(self, cell_shannon_rate_list, cell_serve_num_list):
        ave_rate = 0.0
        resource_allocation_list = []
        cell_tail_utility_list = []
        for i in range(self.n_cell):
            if cell_serve_num_list[i] == 0:
                W_v = 0
                ave_rate += W_v*cell_shannon_rate_list[i]
                resource_allocation_list.append(W_v)
                cell_tail_utility_list.append(0.0)
            else:
                W_v = self.resource_list[i] / cell_serve_num_list[i]
                ave_rate += W_v * cell_shannon_rate_list[i]
                resource_allocation_list.append(W_v)
                cell_tail_utility_list.append(math.log(W_v)*cell_serve_num_list[i])
        ave_rate /= self.n_vehicle_net
        return ave_rate, resource_allocation_list, cell_tail_utility_list
    
    def print_info(self):
        print("agent_number:", self.n_agents)


class Cluster:
    def __init__(self, num_range, SINR_range, V2I_latency_para, V2V_latency_para, num_relay, relay_method, latency_ubound, latency_avebound, cell_id, cluster_type = "intra"):
        # 1) generate cluster
        self.num_range = num_range
        self.SINR_range = SINR_range
        self.V2I_latency_para = V2I_latency_para
        self.V2V_latency_para = V2V_latency_para
        self.num_relay = num_relay
        self.relay_method = relay_method
        self.cluster_type = cluster_type
        
        self.latency_ubound = latency_ubound
        self.latency_avebound = latency_avebound
        self.cell_id = cell_id # for the virtual network: single int for the intra cluster, list for the inter cluster
        
        self.action_offset = 1000 # for baseline to select relay directly

        self._generate_cluster()

        # 2) initialize state space
        # observation space:
        # choosen relay, SINR and latency(V2I + V2V) for candidate relay & V2I directly, allocated resource for candidate relay(calculated in VNENV step)
        self.observation_spaces = [self.num_relay + 1 + (self.num_relay + 1) * 3] * self.n_vehicle 
        self.n_action_list = [self.num_relay + 1] * self.n_vehicle # currently self.n_vehicle > self.num_relay always satisfies

        self.global_step = 0        
    
    
    def setup(self):
        self.SINR = np.random.uniform(self.SINR_range[0], self.SINR_range[1], size = self.n_vehicle)
        self.shannon_rates = [math.log(1+math.pow(10,val/10),2) for val in self.SINR]
        self.utility_list = [round(math.log(val), 3) for val in self.shannon_rates] # log shannon_rates(throughput) as utilities
        
        self.V2I_latency = self.V2I_latency_para[0] + np.random.lognormal(self.V2I_latency_para[1], self.V2I_latency_para[2], size = self.n_vehicle)
        self.V2V_latency = self.V2V_latency_para[0] + np.random.lognormal(self.V2V_latency_para[1], self.V2V_latency_para[2], size = (self.n_vehicle,self.n_vehicle))
        for i in range(self.n_vehicle):
            self.V2V_latency[i][i] = 0

        # for dynamics on latencys
        self.V2I_latency_copy = copy.deepcopy(self.V2I_latency)
        self.V2V_latency_copy = copy.deepcopy(self.V2V_latency)
        self.SINR_copy = copy.deepcopy(self.SINR)

        self._update_state()
        self.select_relays(self.num_relay, self.relay_method)


    def get_observation_spaces(self):
        return self.observation_spaces

    def get_action_spaces(self):
        return self.n_action_list

    def get_max_rate(self):
        return self.utility_list[self.max_SINR_relay]
        
    def get_per_rate(self, action_list):
        per_v_shannon_rate = []
        per_v_cell_id_list = []
        for i in range(self.n_vehicle):
            if action_list[i] >= 0:       
                relay = self.candidate_relay[i][action_list[i]] # 0 for self-relay
            else:
                if action_list[i] == -1: # action == -1 means no relay
                    relay = i
                elif action_list[i] == -2: # action == -2 means max_SINR
                    relay = self.max_SINR_relay
                elif action_list[i] == -3: # action == -2 means max_SINR_left
                    relay = self.max_SINR_relay_left
                elif action_list[i] == -4: # action == -2 means max_SINR_right
                    relay = self.max_SINR_relay_right
                else:
                    relay = action_list[i] + self.action_offset
                
            per_v_shannon_rate.append(self.shannon_rates[relay])
            per_v_cell_id_list.append(self.cell_id_list[relay])
        return per_v_shannon_rate, per_v_cell_id_list
    
    def step(self, action_list):
        utility = 0.0
        cell_shannon_rate = [0.0] * self.n_cell
        latency_list = []
        cell_serve_num = [0] * self.n_cell
        next_state_list = []
        chosen_relay_list = []
        
        for i in range(self.n_vehicle):
            if action_list[i] >= 0:
                relay = self.candidate_relay[i][action_list[i]]
            else:
                if action_list[i] == -1: # action == -1 means no relay
                    relay = i
                elif action_list[i] == -2: # action == -2 means max_SINR
                    relay = self.max_SINR_relay
                elif action_list[i] == -3: # action == -3 means max_SINR_left
                    relay = self.max_SINR_relay_left
                elif action_list[i] == -4: # action == -4 means max_SINR_right
                    relay = self.max_SINR_relay_right
                else:
                    relay = action_list[i] + self.action_offset # for baselines to select the relay directly

            chosen_relay_list.append(relay)

            utility += self.utility_list[relay]
            cell_shannon_rate[self.cell_id_list[relay]] += self.shannon_rates[relay]
            cell_serve_num[self.cell_id_list[relay]] += 1

            latency_list.append(self.V2I_latency[relay] + self.V2V_latency[i][relay])
        
        ave_latency = sum(latency_list) / self.n_vehicle
        peak_violation = 0
        violation_indicator = []
        for val in latency_list:
            if val > self.latency_ubound:
                peak_violation += 1
                violation_indicator.append(1)
            else:
                violation_indicator.append(0)
        # for greedy algorithm
        self.latency_list = latency_list
        self.peak_violation = peak_violation
        self.ave_latency = ave_latency
        
        # For CMIX version, we didn't add noise/dynamics toward the environment
        #self.add_noise()
        #self._update_state()
        #self.select_relays(self.num_relay, self.relay_method)
        
        # add extra state for candidiate relay info 
        candidate_relay_cellid_list = [] # for resource allocation
        for i in range(self.n_vehicle):
            # calculate state info
            state = []
            for j in self.candidate_relay[i]:
                if chosen_relay_list[i] == j:
                    state.append(1)
                else:
                    state.append(0)
            candidate_latency_list = [self.V2I_latency[j] + self.V2V_latency[i][j] for j in self.candidate_relay[i]]
            candidate_shannon_rate_list = [self.shannon_rates[j] for j in self.candidate_relay[i]]
            
            # state normalization
            max_candidate_latency = max(candidate_latency_list)
            max_candidate_shannon_rate = max(candidate_shannon_rate_list)
            for j in range(self.num_relay + 1):
                candidate_latency_list[j] /= max_candidate_latency
                candidate_shannon_rate_list[j] /= max_candidate_shannon_rate
            
            state += candidate_latency_list
            state += candidate_shannon_rate_list    
            next_state_list.append(state)

            if self.cluster_type == "intra":
                candidate_relay_cellid = [self.cell_id for j in self.candidate_relay[i]]
            else:
                candidate_relay_cellid = [self.cell_id[self.cell_id_list[j]] for j in self.candidate_relay[i]]
            candidate_relay_cellid_list.append(candidate_relay_cellid)
        self.global_step += 1 

        return next_state_list, candidate_relay_cellid_list,  utility, ave_latency, peak_violation, cell_serve_num, cell_shannon_rate, violation_indicator 

    def reset(self):
        state_list = [[0] * self.observation_spaces[i] for i in range(self.n_vehicle)]
        self.global_step = 0
        return state_list

    def _generate_cluster(self):
        self.n_vehicle = np.random.randint(self.num_range[0], self.num_range[1])
        
        # generate vehicles 
        if self.cluster_type == "intra":
            self.n_cell = 1
            self.cluster_type_id = 0
            self.cell_id_list = [0] * self.n_vehicle
        elif self.cluster_type == "inter":
            self.n_cell = 2
            self.cluster_type_id = 1
            while True:
                rand_cell = np.random.rand(self.n_vehicle)
                rand_cell[rand_cell >= 0.5] = 1
                rand_cell[rand_cell < 0.5] = 0
                self.cell_id_list = rand_cell.astype(np.int32)
                if sum(self.cell_id_list) < self.n_vehicle:
                    break
        else:
            raise Exception("Cluster type error")
    
    def add_noise2(self):
        self.SINR = np.random.uniform(self.SINR_range[0], self.SINR_range[1], size = self.n_vehicle)
        self.shannon_rates = [math.log(1+math.pow(10,val/10),2) for val in self.SINR]
        self.utility_list = [round(math.log(val), 3) for val in self.shannon_rates] # log shannon_rates(throughput) as utilities
        
        self.V2I_latency = self.V2I_latency_para[0] + np.random.lognormal(self.V2I_latency_para[1], self.V2I_latency_para[2], size = self.n_vehicle)
        self.V2V_latency = self.V2V_latency_para[0] + np.random.lognormal(self.V2V_latency_para[1], self.V2V_latency_para[2], size = (self.n_vehicle,self.n_vehicle))
        for i in range(self.n_vehicle):
            self.V2V_latency[i][i] = 0

    
    def add_noise(self):
        self.V2I_latency_noise = (np.random.rand(self.n_vehicle) - 0.5) / 2 # 5
        self.V2I_latency = (1 + self.V2I_latency_noise) * self.V2I_latency_copy
        self.V2V_latency_noise = (np.random.rand(self.n_vehicle,self.n_vehicle)-0.5) / 2 # 5
        self.V2V_latency = (1 + self.V2V_latency_noise) * self.V2V_latency_copy
        for i in range(self.n_vehicle):
            self.V2V_latency[i][i] = 0

        self.SINR_noise = (np.random.rand(self.n_vehicle) - 0.5) / 2 # 5
        self.SINR = (1 + self.SINR_noise) * self.SINR_copy

        self.shannon_rates = [math.log(1+math.pow(10,val/10),2) for val in self.SINR]
        self.utility_list = [round(math.log(val), 3) for val in self.shannon_rates] # log shannon_rates(throughput) as utilities
        
    
    def _update_state(self):
        self.SINR_sort_index = np.argsort(self.SINR)
        self.max_SINR_relay = self.SINR_sort_index[-1]
        self.V2I_latency_sort_index = list(np.argsort(self.V2I_latency))
        self.V2I_latency_sort_index.reverse()
        
        if self.cluster_type == "inter":
            self.max_SINR_relay_left = -1
            self.max_SINR_relay_right = -1
            for i in range(self.n_vehicle):
                relay_tmp = self.SINR_sort_index[self.n_vehicle - 1 - i]
                relay_tmp_cell_id = self.cell_id_list[relay_tmp]
                if relay_tmp_cell_id == 0 and self.max_SINR_relay_left == -1:
                    self.max_SINR_relay_left = relay_tmp
                if relay_tmp_cell_id == 1 and self.max_SINR_relay_right == -1:
                    self.max_SINR_relay_right = relay_tmp
            self.max_relay_rate_left = self.utility_list[self.max_SINR_relay_left]
            self.max_relay_rate_right = self.utility_list[self.max_SINR_relay_right]


    def select_relays(self, num_relay, method):
        if num_relay > self.n_vehicle - 1:
            self.num_relay = self.n_vehicle - 1

        if method == "SINR":
            sort_index = self.SINR_sort_index
        elif method == "V2I":
            sort_index = self.V2I_latency_sort_index
        else:
            raise Exception("Relay selection method error")
        
        self.candidate_relay = []
        for i in range(self.n_vehicle):
            tmp_relay = [i]
            for j in reversed(sort_index):
                if j != i:
                    tmp_relay.append(j)
                    if len(tmp_relay) == self.num_relay + 1:
                        break
            self.candidate_relay.append(tmp_relay)
        
        # For CMIX version, we assume each agent has a action space of num_relay + 1 and self.n_vehicle > self.num_relay always satisfies
        self.n_action_list = [] # action space of each vehicle
        for i in range(self.n_vehicle):
            self.n_action_list.append(len(self.candidate_relay[i]))
        

    