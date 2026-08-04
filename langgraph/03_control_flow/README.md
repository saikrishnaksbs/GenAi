# LangGraph Control Flow & Branching

This directory details control flow patterns in LangGraph. It covers conditional routing, parallel execution branches (fan-out/fan-in), dynamic execution loops with the `Send()` mapping primitive, and multi-entry point graph configurations.

---

## Table of Contents
1. [Conditional Edges & Dynamic Branching](#1-conditional-edges--dynamic-branching)
2. [Parallel Execution (Fan-Out / Fan-In)](#2-parallel-execution-fan-out--fan-in)
3. [Dynamic Parallelism: The Send() Primitive](#3-dynamic-parallelism-the-send-primitive)
4. [Entry & Exit Sentinels (Multiple Entry Points)](#4-entry--exit-sentinels-multiple-entry-points)

---

## 1. Conditional Edges & Dynamic Branching
Dynamic control flow is managed using conditional routing (covered in [01_conditional_edges_routing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/03_control_flow/01_conditional_edges_routing.py)).
- A **Routing Function** reads the current state and returns a node name.
- Registered via `builder.add_conditional_edges(source_node, routing_fn)`.
- It is commonly used to inspect model outputs (e.g. deciding whether to execute a tool, return a response to the user, or escalate a request).

---

## 2. Parallel Execution (Fan-Out / Fan-In)
When a single node has multiple outgoing edges to downstream nodes, LangGraph executes the target nodes concurrently (covered in [02_parallel_fan_out_fan_in.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/03_control_flow/02_parallel_fan_out_fan_in.py)).
- **Fan-Out**: Parallel execution of downstream branches (e.g., querying three retrieval sources concurrently).
- **Fan-In**: The execution waits until all parallel branches complete before executing the next downstream node.
- **State Merging**: Since multiple parallel nodes write to the state concurrently, any shared state keys they update **must** have a reducer configured to prevent conflict errors.

---

## 3. Dynamic Parallelism: The Send() Primitive
Static fan-out requires knowing the number of parallel branches during graph design. For dynamic parallel workloads (e.g., running a grading node once for each essay in a dynamically sized list), LangGraph provides the **`Send`** primitive (covered in [03_send_api_map_parallelism.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/03_control_flow/03_send_api_map_parallelism.py)).
- Spawns parallel executions at runtime.
- The routing function returns a list of `Send(node_name, state_dict)` objects.
- Each `Send` invocation receives its own isolated, local copy of the state.
- The outputs of all runs are merged back into the parent state using the registered state key reducers.

```python
from langgraph.constants import Send

def continue_routing(state: State):
    # Dynamically spawn 1 target node per element in list
    return [Send("evaluate_chunk", {"chunk": item}) for item in state["chunks"]]
```

---

## 4. Entry & Exit Sentinels (Multiple Entry Points)
Graph execution begins at **`START`** and ends at **`END`** (covered in [04_start_end_multiple_entry_points.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/03_control_flow/04_start_end_multiple_entry_points.py)).
- Standard graphs connect `START` directly to a single entry node.
- You can design **multiple entry points** by attaching a conditional edge directly to `START`.
- This routes the initial step dynamically based on input properties before running the main graph logic.

```python
# Route entry point dynamically
builder.add_conditional_edges(START, route_entry)
```
