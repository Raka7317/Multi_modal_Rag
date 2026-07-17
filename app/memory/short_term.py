"""
Short-term / working memory = conversation state (message history, last
retrieved chunks, scratch variables) scoped to a single session. LangGraph's
checkpointer persists the graph's state between invocations keyed by
thread_id, so a multi-turn conversation "remembers" what was just said
without us hand-rolling any of that bookkeeping.

InMemorySaver is fine for a single dev instance / a single ECS task with
sticky sessions. For multi-instance production behind an ALB without
session affinity, swap this for a shared backend (Redis / Postgres
checkpointer) so any task can pick up any thread_id.
"""
from langgraph.checkpoint.memory import InMemorySaver

_checkpointer: InMemorySaver | None = None


def get_short_term_checkpointer() -> InMemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
    return _checkpointer
