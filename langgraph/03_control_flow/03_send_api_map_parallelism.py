"""
Dynamic Edges with the Send() Primitive
===========================================
Fan-out with static edges requires knowing the branch count ahead of
time. `Send` lets a routing function spawn a *variable* number of
parallel node invocations at runtime - a "map" over a dynamically sized
list. Each `Send(node_name, state_dict)` schedules one execution of
`node_name` with its own independent state. This is the map-style
parallelism pattern: e.g. run the same "grade_essay" node once per essay
in a list, however many essays there are.
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class OverallState(TypedDict):
    essays: list[str]
    grades: Annotated[list[int], operator.add]


class GradeState(TypedDict):
    essay: str


def grade_essay(state: GradeState) -> dict:
    # Each invocation only sees a single essay, not the whole list.
    grade = min(10, len(state["essay"].split()))  # Toy scoring by word count.
    return {"grades": [grade]}


def dispatch_essays(state: OverallState) -> list[Send]:
    # Returning a list of Send objects fans out to N parallel `grade_essay`
    # runs, one per essay, each with its own scoped state.
    return [Send("grade_essay", {"essay": essay}) for essay in state["essays"]]


def collect(state: OverallState) -> dict:
    return {}  # Grades already accumulated via the operator.add reducer.


builder = StateGraph(OverallState)
builder.add_node("grade_essay", grade_essay)
builder.add_node("collect", collect)

# The routing function attached to START replaces a static edge: it decides
# at runtime how many parallel branches to create.
builder.add_conditional_edges(START, dispatch_essays, ["grade_essay"])
builder.add_edge("grade_essay", "collect")
builder.add_edge("collect", END)

graph = builder.compile()

result = graph.invoke({
    "essays": ["short one", "a somewhat longer essay text here", "medium length essay"],
    "grades": [],
})
print(result["grades"])
# -> [2, 6, 3]  (one grade per essay, computed in parallel)
