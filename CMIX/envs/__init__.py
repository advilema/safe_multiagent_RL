REGISTRY = {}

from .vn_env import VNEnv
from .blockergame_env import BlockerGameEnv
from envs.explore import ExploreDiscretized

REGISTRY["vn"] = VNEnv
REGISTRY["blocker"] = BlockerGameEnv
REGISTRY["explore"] = ExploreDiscretized

