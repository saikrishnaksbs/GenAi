"""
Replaying from a Specific Checkpoint
========================================
Any past checkpoint can be resumed exactly as it was, by passing that
checkpoint's config (thread_id + checkpoint_id) to `.invoke()`. LangGraph
loads state as of that checkpoint and continues execution from the node
that was "next" at that point - effectively replaying the remainder of
the run. The original, later checkpoints are left untouched unless the
replay diverges (see 03_forking_execution.py for that case).
"""

from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    log: Annotated[list[str], operator.add]


def step_a(state: State) -> dict:
    return {"log": ["a"]}


def step_b(state: State) -> dict:
    return {"log": ["b"]}


def step_c(state: State) -> dict:
    return {"log": ["c"]}


builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_node("step_c", step_c)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", "step_c")
builder.add_edge("step_c", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "replay-demo"}}
graph.invoke({"log": []}, config=config)

# Find the checkpoint captured right after step_a (before step_b ran).
history = list(graph.get_state_history(config))
checkpoint_after_a = next(s for s in history if s.values["log"] == ["a"])

# Build a config pointing at that EXACT checkpoint_id.
replay_config = {
    "configurable": {
        "thread_id": "replay-demo",
        "checkpoint_id": checkpoint_after_a.config["configurable"]["checkpoint_id"],
    }
}

# Invoking with input=None + a historical checkpoint_id replays forward
# from that point -- step_b and step_c run again from that saved state.
replayed = graph.invoke(None, config=replay_config)
print(replayed["log"])
# -> ["a", "b", "c"]  (deterministic re-run of step_b, step_c from the "a" checkpoint)

# Because step_b/step_c here are pure, the replay matches the original run
# exactly. In real graphs with side effects (API calls, randomness), replay
# lets you deterministically re-execute the SAME state transitions for
# debugging without re-triggering the earlier steps' side effects.
