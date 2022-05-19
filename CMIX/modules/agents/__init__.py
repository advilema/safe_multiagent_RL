REGISTRY = {}

from .simple_agent import SimpleAgent
from .cql_agent import CQLAgent
REGISTRY['simple'] = SimpleAgent
REGISTRY['cql'] = CQLAgent