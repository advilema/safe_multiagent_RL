from safe_multi_agent_RL.buffer import Buffer
import numpy as np
import json

if __name__ == '__main__':
    dir = "results/Explore_s5_n3_0-0_0"
    constraints = np.load(dir+"constr.npy")
    scores = np.load(dir+"scores.npy")
    modified_scores = np.load(dir+"modified_scores.npy")
    lambdas = np.load(dir+"lambdas.npy")
    with open(dir+"params.json") as file:
        params = json.load(file)

    buffer = Buffer(params, save_path=dir)
    buffer.constraints = constraints
    buffer.lambdas = lambdas
    buffer.scores = scores
    buffer.modified_scores = modified_scores

    buffer.save_results()

