"""
The query-time graph:

  load_memory -> retrieve (hybrid) -> rerank (cross-encoder) -> generate -> save_memory

`thread_id` (short-term, per-conversation) is handled automatically by the
LangGraph checkpointer. `user_id` (long-term, cross-session) facts are
fetched/merged into the prompt explicitly at the load_memory node.
"""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank
from app.memory.short_term import get_short_term_checkpointer
from app.memory.long_term import get_facts, upsert_fact


class RagState(TypedDict):
    user_id: str
    question: str
    messages: Annotated[list[dict], operator.add]
    long_term_facts: dict[str, str]
    retrieved: list[dict]
    reranked: list[dict]
    answer: str


def load_memory_node(state: RagState) -> dict:
    facts = get_facts(state["user_id"])
    return {"long_term_facts": facts}


def retrieve_node(state: RagState) -> dict:
    hits = hybrid_search(state["question"], k=20)
    return {"retrieved": hits}


def rerank_node(state: RagState) -> dict:
    top = rerank(state["question"], state["retrieved"])
    return {"reranked": top}


def generate_node(state: RagState) -> dict:
    context = "\n\n".join(f"[{c['modality']}] {c['content']}" for c in state["reranked"])
    facts_str = "\n".join(f"- {k}: {v}" for k, v in state["long_term_facts"].items())

    prompt = f"""You are a helpful assistant. Use the retrieved context to answer.
Known facts about this user:
{facts_str or "(none)"}

Context:
{context}

Question: {state['question']}
Answer:"""

    # Plug in your LLM call here, e.g. the Anthropic API.
    # answer = anthropic_client.messages.create(...).content[0].text
    answer = f"[LLM ANSWER PLACEHOLDER for prompt of {len(prompt)} chars]"

    return {
        "answer": answer,
        "messages": [{"role": "user", "content": state["question"]},
                     {"role": "assistant", "content": answer}],
    }


def save_memory_node(state: RagState) -> dict:
    # Example: naive long-term fact extraction hook. In production, run this
    # through a small extraction prompt/classifier rather than storing raw Q&A.
    upsert_fact(state["user_id"], "last_question", state["question"])
    return {}


def build_graph():
    graph = StateGraph(RagState)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=get_short_term_checkpointer())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str, user_id: str, thread_id: str) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"user_id": user_id, "question": question, "messages": []},
        config=config,
    )
    return result
