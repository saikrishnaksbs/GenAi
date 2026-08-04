# LangGraph Fundamentals

This directory covers the foundational building blocks of **LangGraph**—an orchestration library for building stateful, multi-actor applications with loops and branches. It details how to declare nodes and edges, configure state schemas, implement field reducers, set up conditional routing, form cycle loops, and compile/limit graphs.

---

## Table of Contents
1. [StateGraph: Nodes and Edges](#1-stategraph-nodes-and-edges)
2. [State Schema Design & Reducers](#2-state-schema-design--reducers)
3. [Conditional Edges & Routing Functions](#3-conditional-edges--routing-functions)
4. [Cycles, Loops, and Agentic Retries](#4-cycles-loops-and-agentic-retries)
5. [Compiling the Graph & Safety Limits](#5-compiling-the-graph--safety-limits)

---

## 1. StateGraph: Nodes and Edges
A LangGraph application is structured as a state machine. It is initialized using the **`StateGraph`** class, which acts as a blueprint.
- **State**: A shared memory structure accessed by all nodes.
- **Nodes**: Plain Python functions or Runnables. A node accepts the current state as input and returns a dictionary representing a *partial update* to that state. Register nodes using `builder.add_node(name, func)`.
- **Edges**: Paths defining transitions. Wired using `builder.add_edge(source, destination)`. Sentinel constants **`START`** and **`END`** specify where graph execution enters and exits.

Demonstrated in [01_stategraph_nodes_edges.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/01_graph_fundamentals/01_stategraph_nodes_edges.py):
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("research", research_node)
builder.add_edge(START, "research")
builder.add_edge("research", END)
graph = builder.compile()
```

---

## 2. State Schema Design & Reducers
The graph state is a schema defined using Python typing (covered in [02_state_schema_reducers.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/01_graph_fundamentals/02_state_schema_reducers.py)):
- **`TypedDict`**: Lightweight schemas with no runtime type checking.
- **`Pydantic BaseModel`**: Provides runtime field validation and type coercion.

### Reducers
By default, if a node returns a dictionary containing a key that already exists in the state, the new value **overwrites** the old one. You can change this behavior by wrapping fields in **`Annotated`** and specifying a **reducer function**:
- **`operator.add`**: Appends lists of objects together rather than overwriting.
- **`add_messages`**: A specialized LangGraph reducer that merges message lists, automatically replacing existing message objects that share the same `id` (crucial for streaming updates).

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    # Appends new items to the list instead of overwriting it
    logs: Annotated[list[str], operator.add]
```

---

## 3. Conditional Edges & Routing Functions
While basic edges always route execution from node A to node B, [Conditional Edges](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/01_graph_fundamentals/03_conditional_edges.py#L4-L12) decide where to route dynamically at runtime.
- **Routing Function**: A helper function that reads the state and returns the name of the next node to run.
- Wires routes using `builder.add_conditional_edges(source_node, routing_func, path_map)`. The `path_map` dictionary maps returned strings to actual node names.

```python
def route(state: State) -> str:
    return "tools" if len(state["messages"]) > 0 else "end"

builder.add_conditional_edges("agent", route, {"tools": "tool_node", "end": END})
```

---

## 4. Cycles, Loops, and Agentic Retries
Linear LCEL chains are Directed Acyclic Graphs (DAGs) and can only execute forward. LangGraph supports loops by allowing edges to point **backward** to previously executed nodes (covered in [04_cycles_and_loops.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/01_graph_fundamentals/04_cycles_and_loops.py)).
- Combining backward edges with conditional routing allows you to implement agent loops (e.g. executing tool-call retry loops until a target task succeeds).

---

## 5. Compiling the Graph & Safety Limits
Calling [`.compile()`](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/01_graph_fundamentals/05_compiling_a_graph.py#L4-L12) on the StateGraph builder validates its structure (checks for dangling references or unresolved edges) and returns a runnable `CompiledStateGraph`.
- **`recursion_limit`**: Configured on execution (typically via `config={"recursion_limit": 25}`). It defines the maximum number of super-steps the graph can run, acting as a safeguard to abort infinite loop cycles.
- **Checkpointers**: Registered at compile time to persist state (details in `04_persistence_checkpointing`).
