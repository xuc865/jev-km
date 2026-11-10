"""General JEV agent harness.

Adapters expose observations/legal actions; JevPolicy scores them; the runner
loops. No game-specific fallback lives in this package.
"""

from .adapter import GameAdapter
from .outer import OuterEscalation
from .policy import JevPolicy
from .runner import run_episode
from .types import Action, Decision, EpisodeResult, Observation

__all__ = [
    "Action",
    "Decision",
    "EpisodeResult",
    "GameAdapter",
    "JevPolicy",
    "Observation",
    "OuterEscalation",
    "run_episode",
]
