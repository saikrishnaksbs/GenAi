"""
Forking Execution by Updating State at a Past Checkpoint
=============================================================
Time travel becomes "forking" when, instead of just replaying a past
checkpoint verbatim, you first call `update_state()` on it with new
values. This creates a NEW checkpoint branching off the historical one -
the original timeline is preserved untouched, and a fresh sequence of
checkpoints grows from the fork point. This is how you explore
"what if node X had produced different output" without losing the
original run.
"""

from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    log: Annotated[list[str], operator.add]
    multiplier: int
    total: int


def step_a(state: State) -> dict:
    return {"log": ["a"], "multiplier": 2}


def step_b(state: State) -> dict:
    return {"log": ["b"], "total": 10 * state["multiplier"]}


builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "fork-demo"}}
original = graph.invoke({"log": [], "multiplier": 1, "total": 0}, config=config)
print(original)
# -> {"log": ["a", "b"], "multiplier": 2, "total": 20}  (original timeline)

# Find the checkpoint right after step_a, before step_b used `multiplier`.
history = list(graph.get_state_history(config))
checkpoint_after_a = next(s for s in history if s.values["log"] == ["a"])

# update_state at a HISTORICAL checkpoint_id forks: it creates a new
# checkpoint branching off that point, rather than mutating the original.
fork_config = {
    "configurable": {
        "thread_id": "fork-demo",
        "checkpoint_id": checkpoint_after_a.config["configurable"]["checkpoint_id"],
    }
}
forked_config = graph.update_state(fork_config, {"multiplier": 5})

# Resume from the fork -- step_b now runs with multiplier=5 instead of 2.
forked_result = graph.invoke(None, config=forked_config)
print(forked_result)
# -> {"log": ["a", "b"], "multiplier": 5, "total": 50}  (new, forked timeline)

# The original checkpoint chain (multiplier=2, total=20) still exists in
# history and was never overwritten -- get_state_history now shows both
# branches reachable from the shared "a" checkpoint.
