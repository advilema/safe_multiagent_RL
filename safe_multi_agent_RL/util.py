from multiagent_envs.explore import ExploreContinuous, ExploreDiscrete
from multiagent_envs.grid import Grid
from multiagent_envs.potential_grid import PotentialGrid
from multiagent_envs.congestion import Congestion
from multiagent_envs.space import Space

def make_env(params):
    continuous = False
    if params.environment == "ExploreDiscrete":
        return ExploreDiscrete(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights), continuous
    elif params.environment == "ExploreContinuous":
        continuous = True
        return ExploreContinuous(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights), continuous
    elif params.environment == "Grid":
        return Grid(params.size, params.n_agents, params.n_landmarks, shuffle=params.shuffle), continuous
    elif params.environment == "Space":
        continuous = True
        return Space(params.size, params.n_agents, n_landmarks=params.n_landmarks, shuffle=params.shuffle), continuous
    elif params.environment == "PotentialGrid":
        return PotentialGrid(params.size, params.n_agents), continuous
    elif params.environment == "Congestion":
        return Congestion(params.size, params.n_agents, noise=params.noise, shuffle=params.shuffle), continuous
    else:
        print("Error: params.env need to be either ExploreDiscrete, ExploreContinuous or Grid")
        exit(1)


def print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle):
    scores, modified_score, constraints = buffer.mean_score()
    print('Meta-Agent lc {}\t Agents lc {}\t Score: {:.2f}, '
          'Modified Score: {:.2f}, Constraints: {}'.format(meta_agent_learning_cycle,
                                                           agents_learning_cycle, scores,
                                                           modified_score, constraints))