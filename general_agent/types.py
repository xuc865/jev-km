"""Core types shared by the general JEV agent harness.

The harness is game-agnostic: adapters expose observations and legal actions;
policies choose one; the runner executes. Nothing here knows about mazes,
cards, shooters, or any specific game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """One legal action the game exposed this frame."""

    id: str
    label: str = ""
    kind: str = "hold"  # hold | click | key | axis | custom
    keys: tuple[str, ...] = ()
    seconds: float = 0.15
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        bits = [self.label or self.id]
        if self.seconds:
            bits.append(f"hold {self.seconds:g}s")
        if self.keys:
            bits.append("keys " + "+".join(self.keys))
        return " | ".join(bits)


@dataclass
class Observation:
    """Everything the policy may use for one decision."""

    actions: list[Action]
    image: str = ""  # data URL, optional; JEV is multimodal when present
    state: dict[str, Any] = field(default_factory=dict)
    level: int = 1
    done: bool = False
    success: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """One policy decision; no game-specific rule is hidden in here."""

    action_id: str
    by: str = "jev"  # jev | escalate | wait | stop
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str = ""
    image_count: int = 0
    ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    note: str = ""


@dataclass
class EpisodeResult:
    """Outcome of one general-harness episode."""

    success: bool
    steps: int
    seconds: float
    decisions: list[Decision] = field(default_factory=list)
    stopped: bool = False
    error: str = ""
