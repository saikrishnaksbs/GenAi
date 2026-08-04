"""
Checkpoint Structure: Inspecting get_state()
================================================
Every checkpoint captures more than just the state values: it also
records which node(s) are `next` to run, the config (including
thread_id and a unique checkpoint_id), and metadata (step number, source
of the write, any pending writes). `graph.get_state(config)` returns a
`StateSnapshot` with all of this, which is what powers debugging, human
review, and time-travel workflows.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator


class State(TypedDict):
    steps_done: Annotated[list[str], operator.add]


def step_a(state: State) -> dict:
    return {"steps_done": ["a"]}


def step_b(state: State) -> dict:
    return {"steps_done": ["b"]}


builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "inspect-demo"}}
graph.invoke({"steps_done": []}, config=config)

snapshot = graph.get_state(config)

# `.values` -- the actual state dict at this checkpoint.
print(snapshot.values)
# -> {"steps_done": ["a", "b"]}

# `.next` -- which node(s) would run next; empty tuple means the graph finished.
print(snapshot.next)
# -> ()

# `.config` -- includes thread_id AND a specific checkpoint_id, so this
# snapshot can be referenced again later (e.g. for time travel).
print(snapshot.config["configurable"]["checkpoint_id"])
# -> "1ef...."  (a ULID-like unique id for this exact checkpoint)

# `.metadata` -- bookkeeping about how this checkpoint was produced.
print(snapshot.metadata)
# -> {"source": "loop", "step": 2, "writes": {"step_b": {"steps_done": ["b"]}}, ...}

# get_state_history() (see 08_time_travel_replay) yields every checkpoint
# ever written for a thread, oldest last, each as its own StateSnapshot.
history = list(graph.get_state_history(config))
print(len(history))
# -> 3 (initial + after step_a + after step_b, roughly one per super-step)
