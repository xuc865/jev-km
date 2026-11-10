"""Agent harness: Thread/Turn + context compress, skills, master, trace.

Self-contained in ``jevgamer/``. Mechanisms live under this package; naming
docs for Thread/Turn/Item sit in ``third_party/openai-codex/``.

Invariant: a model call is NEVER a single-shot user message.
"""

from .condense_store import CondenseSegment, CondenseStore, SegmentKind
from .context import ContextConfig, ContextEngine, estimate_messages_tokens, llm_input_budget
from .master import MasterInvokeResult, MasterRuntime
from .skills import SkillCard, SkillSession, render_roster
from .thread import Item, Thread, Turn
from .trace import Span, TraceStore

__all__ = [
    "CondenseSegment",
    "CondenseStore",
    "SegmentKind",
    "ContextConfig",
    "ContextEngine",
    "estimate_messages_tokens",
    "llm_input_budget",
    "MasterInvokeResult",
    "MasterRuntime",
    "SkillCard",
    "SkillSession",
    "render_roster",
    "Item",
    "Thread",
    "Turn",
    "Span",
    "TraceStore",
]
