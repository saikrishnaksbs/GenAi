"""
Reducers: Annotated[list, operator.add] and Custom Reducer Functions
========================================================================
By default, when two nodes update the same state key, the second write
wins (or, in parallel branches, LangGraph raises an error for conflicting
concurrent writes to a plain key). A reducer changes this: it's a
function of (current_value, new_value) -> merged_value that LangGraph
calls automatically whenever a node returns a value for that key. This
lets keys behave like accumulators, sets, or anything else you define,
rather than simple overwrite slots. `operator.add` is the built-in
reducer for concatenating lists (or summing numbers).
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    # operator.add on a list means "append/concatenate", not overwrite.
    log: Annotated[list[str], operator.add]
    # operator.add on an int means "sum".
    total_cost: Annotated[int, operator.add]


def step_one(state: State) -> dict:
    return {"log": ["step_one ran"], "total_cost": 10}


def step_two(state: State) -> dict:
    return {"log": ["step_two ran"], "total_cost": 25}


print("===============================================================================")
print("            LANGGRAPH STATE MANAGEMENT: BUILT-IN & CUSTOM REDUCERS             ")
print("===============================================================================\n")

builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)
graph = builder.compile()

print("➕ [Built-in Reducer] operator.add (List Concatenation & Integer Sum):")
result = graph.invoke({"log": [], "total_cost": 0})
print(f"   • Accumulated Log : {result['log']}")
print(f"   • Total Cost      : {result['total_cost']}\n")


# --- Custom reducer: merge dictionaries, keeping the max value per key ---
def merge_max_scores(current: dict, new: dict) -> dict:
    merged = dict(current)
    for key, value in new.items():
        merged[key] = max(merged.get(key, value), value)
    return merged


class ScoreState(TypedDict):
    scores: Annotated[dict, merge_max_scores]


def scorer_a(state: ScoreState) -> dict:
    return {"scores": {"relevance": 0.6, "clarity": 0.9}}


def scorer_b(state: ScoreState) -> dict:
    return {"scores": {"relevance": 0.8, "clarity": 0.5}}


score_builder = StateGraph(ScoreState)
score_builder.add_node("scorer_a", scorer_a)
score_builder.add_node("scorer_b", scorer_b)
score_builder.add_edge(START, "scorer_a")
score_builder.add_edge("scorer_a", "scorer_b")
score_builder.add_edge("scorer_b", END)
score_graph = score_builder.compile()

print("🎯 [Custom Reducer] merge_max_scores (Retains highest score per key):")
score_res = score_graph.invoke({"scores": {}})
print(f"   • Merged Scores   : {score_res['scores']}\n")

# -> {"scores": {"relevance": 0.8, "clarity": 0.9}}  (max of each key kept)
