REGISTRY = {}
from .q_learner import QLearner 
from .cql_learner import CQLLearner
REGISTRY["simple"] = QLearner
REGISTRY["cql"] = CQLLearner