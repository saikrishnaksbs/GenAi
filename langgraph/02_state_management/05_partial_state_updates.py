"""
Multiple State Channels and Partial State Updates
=====================================================
A node never has to return the entire state - only the keys ("channels")
it wants to change. LangGraph merges the returned partial dict into the
existing state using each key's reducer (or overwrite, if no reducer is
defined). This lets you design a state schema with many independent
channels, where each node only touches the ones relevant to it, keeping
nodes decoupled and easy to test in isolation.
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END


class PipelineState(TypedDict):
    raw_text: str
    cleaned_text: str
    word_count: int
    warnings: Annotated[list[str], operator.add]
    sentiment: str


def clean_node(state: PipelineState) -> dict:
    # Only touches `cleaned_text` -- other channels are left untouched.
    cleaned = state["raw_text"].strip().lower()
    return {"cleaned_text": cleaned}


def count_node(state: PipelineState) -> dict:
    # Only touches `word_count` and conditionally `warnings`.
    words = state["cleaned_text"].split()
    update = {"word_count": len(words)}
    if len(words) == 0:
        update["warnings"] = ["empty input after cleaning"]
    return update


def sentiment_node(state: PipelineState) -> dict:
    # Only touches `sentiment`.
    positive_words = {"good", "great", "excellent"}
    has_positive = any(w in positive_words for w in state["cleaned_text"].split())
    return {"sentiment": "positive" if has_positive else "neutral"}


builder = StateGraph(PipelineState)
builder.add_node("clean", clean_node)
builder.add_node("count", count_node)
builder.add_node("sentiment", sentiment_node)
builder.add_edge(START, "clean")
builder.add_edge("clean", "count")
builder.add_edge("count", "sentiment")
builder.add_edge("sentiment", END)
graph = builder.compile()

result = graph.invoke({
    "raw_text": "  This is Great stuff  ",
    "cleaned_text": "",
    "word_count": 0,
    "warnings": [],
    "sentiment": "",
})
print(result)
# -> {
#      "raw_text": "  This is Great stuff  ",
#      "cleaned_text": "this is great stuff",
#      "word_count": 4,
#      "warnings": [],
#      "sentiment": "positive",
#    }
# Each node contributed a partial update; unmentioned keys simply passed through.
