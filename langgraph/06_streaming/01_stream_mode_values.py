"""
stream_mode="values": Full State After Each Step
====================================================
`graph.stream(input, config, stream_mode="values")` yields the ENTIRE
state snapshot after each super-step completes, not just what changed.
This is the simplest streaming mode to reason about - each chunk is a
complete, self-consistent state dict - at the cost of re-sending
unchanged fields on every step. Good for UIs that just want to render
"current state" on each update.
"""

from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    steps: Annotated[list[str], operator.add]
    counter: int


def step_one(state: State) -> dict:
    return {"steps": ["one"], "counter": state["counter"] + 1}


def step_two(state: State) -> dict:
    return {"steps": ["two"], "counter": state["counter"] + 1}


def step_three(state: State) -> dict:
    return {"steps": ["three"], "counter": state["counter"] + 1}


builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_node("step_three", step_three)
builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", "step_three")
builder.add_edge("step_three", END)
graph = builder.compile()

for chunk in graph.stream({"steps": [], "counter": 0}, stream_mode="values"):
    # Each `chunk` is the FULL state as of that point -- always safe to
    # render directly without merging against previous chunks.
    print(chunk)
# -> {"steps": [], "counter": 0}                          (initial state, before any node)
# -> {"steps": ["one"], "counter": 1}                      (after step_one)
# -> {"steps": ["one", "two"], "counter": 2}               (after step_two)
# -> {"steps": ["one", "two", "three"], "counter": 3}      (after step_three)
