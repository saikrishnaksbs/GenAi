"""
stream_mode="updates": Only the Diff from Each Node
=======================================================
Instead of the full state, `stream_mode="updates"` yields a small dict
mapping `{node_name: partial_state_returned_by_that_node}` for each step.
This is more bandwidth-efficient than "values" and makes it easy to show
"node X just did Y" progress messages, but the consumer must merge diffs
itself if it wants the full running state.
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


builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)
graph = builder.compile()

for chunk in graph.stream({"steps": [], "counter": 0}, stream_mode="updates"):
    # Each chunk is {node_name: <what that node returned>} -- just the diff.
    print(chunk)
# -> {"step_one": {"steps": ["one"], "counter": 1}}
# -> {"step_two": {"steps": ["two"], "counter": 2}}

# To reconstruct full state yourself, you'd apply each diff through the
# same reducers the graph itself uses (here operator.add for "steps").
running_state = {"steps": [], "counter": 0}
for chunk in graph.stream({"steps": [], "counter": 0}, stream_mode="updates"):
    for node_name, partial in chunk.items():
        running_state["steps"] = running_state["steps"] + partial.get("steps", [])
        running_state["counter"] = partial.get("counter", running_state["counter"])
print(running_state)
# -> {"steps": ["one", "two"], "counter": 2}
