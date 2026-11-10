"""Two-tier decision system for long-horizon game agents.

Both tiers run the same model under different role prompts. They exchange only
the objects declared in ``contract``: the planner publishes a ``PlanDirective``,
the executor reports an ``ExecutorStep`` upward and receives a ``WorldView``. The
contract is the seam that makes the exchange auditable and the trajectories
distillable.
"""

from .adapter import GameAdapter
from .client import ModelCall, ModelClient
from .contract import (
    SCHEMA_VERSION,
    ExecutorReply,
    ExecutorStep,
    LegalAction,
    PlanDirective,
    PlannerReply,
    WorldView,
    parse_executor_reply,
    parse_planner_reply,
)
from .outer import OuterEscalation, PlannerPolicy
from .policy import ExecutorPolicy, JevPolicy
from .runner import run_episode
from .types import Action, Decision, EpisodeResult, Observation

__all__ = [
    "Action",
    "Decision",
    "EpisodeResult",
    "ExecutorPolicy",
    "ExecutorReply",
    "ExecutorStep",
    "GameAdapter",
    "JevPolicy",
    "LegalAction",
    "ModelCall",
    "ModelClient",
    "Observation",
    "OuterEscalation",
    "PlanDirective",
    "PlannerPolicy",
    "PlannerReply",
    "SCHEMA_VERSION",
    "WorldView",
    "parse_executor_reply",
    "parse_planner_reply",
    "run_episode",
]
