from multiagent_envs.explore import ExploreContinuous, ExploreDiscrete, ExploreDiscretized
from multiagent_envs.grid import Grid
from multiagent_envs.potential_grid import PotentialGrid
from multiagent_envs.congestion import Congestion
from multiagent_envs.space import Space
from multiagent_envs.cross import Cross
from safe_multi_agent_RL.agent import ReinforceAgent, ACAgent, PPOAgent

def make_env(params):
    continuous = False
    if params.environment == "ExploreDiscrete":
        return ExploreDiscrete(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights), continuous
    elif params.environment == "ExploreDiscretized":
        return ExploreDiscretized(params.size, params.n_agents, coarseness=params.coarseness, shuffle=params.shuffle, weights=params.weights), continuous
    elif params.environment == "ExploreContinuous":
        continuous = True
        return ExploreContinuous(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights, coarseness=params.coarseness), continuous
    elif params.environment == "Cross":
        continuous = True
        return Cross(), continuous
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


def make_agent(env, params, continuous):
    if params.algo == 'reinforce':
        return ReinforceAgent(env, params, continuous=continuous)
    elif params.algo == 'ac':
        return ACAgent(env, params, continuous=continuous)
    elif params.algo == 'ppo':
        return PPOAgent(env, params, continuous=continuous)
    else:
        print("Error: params.env need to be either Reinforce or A2C")
        exit(1)


def print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle):
    scores, modified_scores, constraints = buffer.mean_score()
    print('Meta-Agent lc {}\t Agents lc {}\t Scores: {}, '
          'Modified Scores: {}, Constraints: {}'.format(meta_agent_learning_cycle,
                                                           agents_learning_cycle, scores,
                                                           modified_scores, constraints))