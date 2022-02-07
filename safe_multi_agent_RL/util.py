from multiagent_envs.explore import Explore
from multiagent_envs.grid import Grid


def make_env(params):
    if params.env == "Explore":
        return Explore(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights)
    elif params.env == "Grid":
        return Grid(params.size, params.n_agents, params.n_landmarks, shuffle=params.shuffle)
    else:
        print("Error: params.env need to be either Explore or Grid")
        exit(1)


def print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle):
    scores, modified_score, constraints = buffer.mean_score()
    print('Meta-Agent lc {}\t Agents lc {}\t Score: {:.2f}, '
          'Modified Score: {:.2f}, Constraints: {}'.format(meta_agent_learning_cycle,
                                                           agents_learning_cycle, scores,
                                                           modified_score, constraints))