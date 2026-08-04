"""
get_state_history() to List Past Checkpoints
================================================
With a checkpointer attached, LangGraph writes a new checkpoint after
every super-step. `graph.get_state_history(config)` returns an iterator
of `StateSnapshot` objects for a given thread_id, newest first, each
capturing the full state, which node(s) run next, and metadata about how
that checkpoint came to be. This is the foundation for debugging past
runs and for time-travel/replay (see the next files).
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

config = {"configurable": {"thread_id": "history-demo"}}
graph.invoke({"log": []}, config=config)

# Iterate the full checkpoint history, newest first.
for snapshot in graph.get_state_history(config):
    print(snapshot.values["log"], "| next:", snapshot.next, "| step:", snapshot.metadata["step"])
# -> ['a', 'b', 'c'] | next: ()          | step: 3   (final)
# -> ['a', 'b']      | next: ('step_c',) | step: 2
# -> ['a']           | next: ('step_b',) | step: 1
# -> []              | next: ('step_a',) | step: 0   (initial, before any node ran)

# Each snapshot's `.config` carries a unique checkpoint_id, which can be
# used to jump back to that exact point (see 02_replay_from_checkpoint.py).
history = list(graph.get_state_history(config))
earliest_after_a = [s for s in history if s.values["log"] == ["a"]][0]
print(earliest_after_a.config["configurable"]["checkpoint_id"])
# -> "1ef...."  a specific checkpoint_id we can replay from
