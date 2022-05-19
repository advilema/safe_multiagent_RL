# -*- coding: utf-8 -*-
import random
import numpy as np
import math, time
from scipy import optimize

def no_relay(network):
    action_list = []
    for cluster in network.cluster_list:
        action_list.append([-1]*cluster.n_vehicle)
    ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))
    per_v_rate = network.get_per_rate()
    Jains_index = com_Jains_index(per_v_rate)
    geo_mean = math.exp(utility_sum / network.n_vehicle_net)
    q_ave_val = (network.latency_avebound - ave_latency_whole) / (1 - network.discount_factor)
    return ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, network.n_vehicle_net
    
def max_SINR(network):
    action_list =[]
    for cluster in network.cluster_list:
        action_list.append([-2]*cluster.n_vehicle)
    ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))
    per_v_rate = network.get_per_rate()
    Jains_index = com_Jains_index(per_v_rate)
    geo_mean = math.exp(utility_sum / network.n_vehicle_net)
    q_ave_val = (network.latency_avebound - ave_latency_whole) / (1 - network.discount_factor)

    return ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, network.n_vehicle_net

def DCRA(network):
    action_list = []
    for cluster in network.cluster_list:
        action_list.append([-2]*cluster.n_vehicle)
    ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))

    n_vehicle_net = network.n_vehicle_net
    max_DCRA_step = 300 
    num_cluster = network.n_cluster
    for step in range(max_DCRA_step):
        cluster_id = np.random.randint(network.n_intra_cluster, num_cluster)
        cluster = network.cluster_list[cluster_id]
        n_vehicle_c = cluster.n_vehicle
        tmp = []
        for ncb_left in range(n_vehicle_c + 1):
            mcb_left = network.get_mcb(cluster_id, "left")
            if ncb_left + mcb_left == 0:
                utility_left = 0.0
            else:
                utility_left = network.get_rcb(cluster_id, "left") / (ncb_left + mcb_left)
            ncb_right = n_vehicle_c - ncb_left
            mcb_right = network.get_mcb(cluster_id, "right")
            if ncb_right + mcb_right == 0:
                utility_right = 0.0
            else:
                utility_right = network.get_rcb(cluster_id, "right")/(ncb_right + mcb_right)
            tmp.append(min([utility_left, utility_right]))
        ncb_left_best = np.argmax(tmp)
        action = [-3]*ncb_left_best + [-4]*(n_vehicle_c - ncb_left_best) 
        action_list[cluster_id] = action

        ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))

    per_v_rate = network.get_per_rate()
    Jains_index = com_Jains_index(per_v_rate)
    geo_mean = math.exp(utility_sum / network.n_vehicle_net)
    q_ave_val = (network.latency_avebound - ave_latency_whole) / (1 - network.discount_factor)

    return ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, network.n_vehicle_net


def min_latency(network):
    action_list = []
    for cluster in network.cluster_list:
        V2I_latency = cluster.V2I_latency
        V2V_latency = cluster.V2V_latency
        action = []
        for v in range(cluster.n_vehicle):
            tmp = []
            for u in range(cluster.n_vehicle):
                tmp.append(V2V_latency[v][u] + V2I_latency[u])
            relay = np.argmin(tmp)
            action.append(relay - cluster.action_offset) # we define the negative offset in VNEnv
        action_list.append(action)
    ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))

    per_v_rate = network.get_per_rate()
    Jains_index = com_Jains_index(per_v_rate)
    geo_mean = math.exp(utility_sum / network.n_vehicle_net)
    q_ave_val = (network.latency_avebound - ave_latency_whole) / (1 - network.discount_factor)

    return ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, network.n_vehicle_net

def greedy(network):
    action_list = []
    for cluster in network.cluster_list:
        action_list.append([cluster.max_SINR_relay - cluster.action_offset] * cluster.n_vehicle)
    ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))

    latency_avebound = network.latency_avebound
    constraint_state = []
    for cluster in network.cluster_list:
        if cluster.peak_violation > 0 or cluster.ave_latency > latency_avebound:
            constraint_state.append(1)
        else:
            constraint_state.append(0)

    max_greedy_step = 10000
    num_cluster = network.n_cluster
    cluster_id_pool = [i for i in range(num_cluster)]
    for step in range(max_greedy_step):
        cluster_id = np.random.choice(cluster_id_pool)
        cluster = network.cluster_list[cluster_id]
        n_vehicle_c = cluster.n_vehicle

        if step % 500 == 0 or step + 10 >= max_greedy_step:
            per_v_rate = network.get_per_rate()
            Jains_index = com_Jains_index(per_v_rate)
        
        if constraint_state[cluster_id] == 0:
            cluster_id_pool.remove(cluster_id)
            if len(cluster_id_pool) == 0:
                break
            continue
        
        v_id_max_latency = np.argmax(cluster.latency_list)
        SINR_list = cluster.SINR
        curr_relay = action_list[cluster_id][v_id_max_latency] + cluster.action_offset
        next_relay = curr_relay
        next_SINR = -1
        
        if v_id_max_latency not in cluster.SINR_sort_index[- cluster.num_relay:]:
            candidate_relay = [v_id_max_latency]
            for v in cluster.SINR_sort_index[- cluster.num_relay:]:
                candidate_relay.append(v)
        else:
            candidate_relay = cluster.SINR_sort_index[- cluster.num_relay:]
        
        # find the relay with biggest SINR which is no more than current relay
        for v in candidate_relay:
            if v == curr_relay:
                continue
            if SINR_list[v] < SINR_list[curr_relay] and SINR_list[v] > next_SINR:
                next_relay = v
                next_SINR = SINR_list[v]
        action_list[cluster_id][v_id_max_latency] = next_relay - cluster.action_offset
        ave_rate, utility_sum, ave_latency_whole, peak_violation_sum = network.step(action_list_flat(action_list))
        
        cluster = network.cluster_list[cluster_id]
        if cluster.peak_violation == 0 and cluster.ave_latency <= latency_avebound:
            constraint_state[cluster_id] = 0
            cluster_id_pool.remove(cluster_id)
            if len(cluster_id_pool) == 0:
                break
        

    per_v_rate = network.get_per_rate()
    Jains_index = com_Jains_index(per_v_rate)
    geo_mean = math.exp(utility_sum / network.n_vehicle_net)
    q_ave_val = (network.latency_avebound - ave_latency_whole) / (1 - network.discount_factor)

    return ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, network.n_vehicle_net


def action_list_flat(action_list):
    actions = []
    for action in action_list:
        actions += action 
    return actions

'''
A metrics for fairness, not used in CMIX
'''
def com_Jains_index(val_list):
    n = len(val_list)
    val_sum = sum(val_list)
    sqr_val_list = [item*item for item in val_list]
    return val_sum*val_sum/(n*sum(sqr_val_list))


