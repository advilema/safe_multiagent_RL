'''
argv[1]: log_dir path
argv[2]: the VNEnv pickle instance path
'''
from baselines.algorithm import *
from envs.vn_env import VNEnv
from utils.utils import cleanup_dir

import sys
import pickle
import time

scheme_list = ["no_relay", "max_SINR", "DCRA", "min_latency", "greedy"]

log_dir = sys.argv[1]    
cleanup_dir(log_dir)

load_env_file = open(sys.argv[2], "rb")
env = pickle.load(load_env_file)
load_env_file.close()
env.reset()
env.set_scheme("basic")

store_env_file = open("%s/env.pickle" % (log_dir), "wb")
pickle.dump(env, store_env_file)
store_env_file.close()

file_out = open("%s/result.log" % (log_dir), "w")

# Average q value not applied in this work
print("Average data rate, Global utility, Jains index, Average latency, Peak violation, Average constraint q value, Geo mean, Number of vehicles, Running time", file = file_out)
for scheme in scheme_list:
    if scheme == "no_relay":
        t0 = time.time()
        ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, n_vehicle_net = no_relay(env)
        t1 = time.time()
        print("no_relay:", round(ave_rate, 2), round(utility_sum, 2), round(Jains_index, 4), round(ave_latency_whole, 2), peak_violation_sum, round(q_ave_val, 2), round(geo_mean, 3), n_vehicle_net, t1-t0, file = file_out)
    elif scheme == "max_SINR":
        t0 = time.time()
        ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, n_vehicle_net = max_SINR(env)
        t1 = time.time()
        print("max_SINR:", round(ave_rate, 2), round(utility_sum, 2), round(Jains_index, 4), round(ave_latency_whole, 2), peak_violation_sum, round(q_ave_val, 2), round(geo_mean, 3), n_vehicle_net, t1-t0, file = file_out)
    elif scheme == "DCRA":
        t0 = time.time()
        ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, n_vehicle_net = DCRA(env)
        t1 = time.time()
        print("DCRA:", round(ave_rate, 2), round(utility_sum, 2), round(Jains_index, 4), round(ave_latency_whole, 2), peak_violation_sum, round(q_ave_val, 2), round(geo_mean, 3), n_vehicle_net, t1-t0, file = file_out)
    elif scheme == "greedy":
        t0 = time.time()
        ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, n_vehicle_net = greedy(env)
        t1 = time.time()
        print("greedy:", round(ave_rate, 2), round(utility_sum, 2), round(Jains_index, 4), round(ave_latency_whole, 2), peak_violation_sum, round(q_ave_val, 2), round(geo_mean, 3), n_vehicle_net, t1-t0, file = file_out)
    elif scheme == "min_latency":
        t0 = time.time()
        ave_rate, utility_sum, Jains_index, ave_latency_whole, peak_violation_sum, q_ave_val, geo_mean, n_vehicle_net = min_latency(env)
        t1 = time.time()
        print("min_latency:", round(ave_rate, 2), round(utility_sum, 2), round(Jains_index, 4), round(ave_latency_whole, 2), peak_violation_sum, round(q_ave_val, 2), round(geo_mean, 3), n_vehicle_net, t1-t0, file = file_out)
    


