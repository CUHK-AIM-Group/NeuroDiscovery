"""Independent subagent execution module.

Provides SubagentManager for spawning and managing independent agent sessions,
and PersonaAgent / MultiAgentDiscussion for persona-driven multi-agent discussions.
"""

from .manager import SubagentManager, SubagentHandle, SubagentResult
from .persona_agent import PersonaAgent, MultiAgentDiscussion, DiscussionResult

__all__ = [
    "SubagentManager",
    "SubagentHandle",
    "SubagentResult",
    "PersonaAgent",
    "MultiAgentDiscussion",
    "DiscussionResult",
]
