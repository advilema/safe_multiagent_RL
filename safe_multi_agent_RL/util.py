from envs.coverage import CoverageContinuous, CoverageDiscrete, CoverageDiscretized
from envs.congestion import Congestion
from envs.collision_avoidance import CollisionAvoidance
from safe_multi_agent_RL.agent import ReinforceAgent, ACAgent, PPOAgent

def make_env(params):
    continuous = False
    if params.environment == "CoverageDiscrete":
        return CoverageDiscrete(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights), continuous
    elif params.environment == "CoverageDiscretized":
        return CoverageDiscretized(params.size, params.n_agents, coarseness=params.coarseness, shuffle=params.shuffle, weights=params.weights), continuous
    elif params.environment == "CoverageContinuous":
        continuous = True
        return CoverageContinuous(params.size, params.n_agents, shuffle=params.shuffle, weights=params.weights, coarseness=params.coarseness), continuous
    elif params.environment == "Collision":
        continuous = True
        return CollisionAvoidance(params.size, params.n_agents, n_landmarks=params.n_landmarks, shuffle=params.shuffle), continuous
    elif params.environment == "Congestion":
        return Congestion(params.size, params.n_agents, noise=params.noise, shuffle=params.shuffle), continuous
    else:
        print("Error: params.env need to be either CoverageDiscrete, CoverageDiscretized, CoverageContinuous, Collision or Congestion")
        exit(1)


def make_agent(env, params, continuous):
    if params.algo == 'reinforce':
        return ReinforceAgent(env, params, continuous=continuous)
    elif params.algo == 'ac':
        return ACAgent(env, params, continuous=continuous)
    elif params.algo == 'ppo':
        return PPOAgent(env, params, continuous=continuous)
    else:
        print("Error: params.algo need to be either reinforce, ac or ppo")
        exit(1)


def print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle):
    scores, modified_scores, constraints = buffer.mean_score()
    print('Meta-Agent lc {}\t Agents lc {}\t Scores: {}, '
          'Modified Scores: {}, Constraints: {}'.format(meta_agent_learning_cycle,
                                                           agents_learning_cycle, scores,
                                                           modified_scores, constraints))