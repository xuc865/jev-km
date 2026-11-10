"""Game adapter protocol.

An adapter is the only game-specific code. It translates the game's native
observation/action space into Observation/Action and executes one action.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Action, Observation


@runtime_checkable
class GameAdapter(Protocol):
    name: str

    def reset(self) -> None:
        """Reset/reload the game if the adapter can; otherwise no-op."""

    def observe(self) -> Observation:
        """Return the current screenshot/state and the legal actions."""

    def act(self, action: Action) -> None:
        """Execute one legal action."""

    def close(self) -> None:
        """Release any client/session resources."""
