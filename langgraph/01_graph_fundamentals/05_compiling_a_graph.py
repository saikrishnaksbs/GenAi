"""
Compiling a Graph: compile(), Config Schema, and Recursion Limit
====================================================================
A `StateGraph` builder is just a blueprint until you call `.compile()`,
which validates the graph (no dangling nodes, all edges resolve) and
returns a runnable `CompiledStateGraph`. compile() also accepts a
checkpointer for persistence (covered in 04_persistence_checkpointing),
interrupt points for human-in-the-loop, and a `ConfigSchema` describing
per-invocation configuration (e.g. which model or user to use). Every
invocation also respects a `recursion_limit`, which caps how many
super-steps a graph can take — a safety net against infinite loops.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    count: int


class ConfigSchema(TypedDict):
    # Config values are passed per-invocation via `config={"configurable": {...}}`
    # and are accessible to nodes through the second `config` argument.
    multiplier: int


def increment_node(state: State, config: dict) -> dict:
    # Read a per-invocation setting out of the RunnableConfig.
    multiplier = config.get("configurable", {}).get("multiplier", 1)
    return {"count": state["count"] + 1 * multiplier}


builder = StateGraph(State, config_schema=ConfigSchema)
builder.add_node("increment", increment_node)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)


print("===============================================================================")
print("          LANGGRAPH FUNDAMENTALS: COMPILING & RECURSION LIMITS                ")
print("===============================================================================\n")

# compile() turns the builder into an executable graph. Nothing runs until
# this call - it just wires and validates the structure.
graph = builder.compile()

print("⚙️ [Config Execution] Invoking graph with per-invocation config (multiplier=3):")
result = graph.invoke(
    {"count": 0},
    config={"configurable": {"multiplier": 3}},
)
print(f"   • Final Count: {result['count']}\n")

# recursion_limit guards against runaway cycles (e.g. a loop that never
# satisfies its exit condition). Exceeding it raises GraphRecursionError.
print("🚨 [Recursion Limit Safety Test] Triggering infinite loop with limit=5:")
try:
    unbounded_builder = StateGraph(State)
    unbounded_builder.add_node("increment", lambda s: {"count": s["count"] + 1})
    unbounded_builder.add_edge(START, "increment")
    unbounded_builder.add_edge("increment", "increment")  # Never terminates.
    unbounded_graph = unbounded_builder.compile()
    unbounded_graph.invoke({"count": 0}, config={"recursion_limit": 5})
except Exception as exc:
    print(f"   ⚠️ Caught expected exception: {type(exc).__name__} ({exc})\n")

