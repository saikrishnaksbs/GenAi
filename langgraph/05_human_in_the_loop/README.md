# LangGraph Human-in-the-Loop (HITL)

This directory details Human-in-the-Loop (HITL) workflows in LangGraph. It covers pausing executions, resuming paused threads, editing states mid-run, implementing approval gates, and executing dynamic inline prompts using the node-level `interrupt()` function.

---

## Table of Contents
1. [Compile-Time Interrupts (Static Pauses)](#1-compile-time-interrupts-static-pauses)
2. [Modifying State Mid-Run (State Updates)](#2-modifying-state-mid-run-state-updates)
3. [Gateways & Tool Approval Workflows](#3-gateways--tool-approval-workflows)
4. [Dynamic Pauses: The interrupt() Function](#4-dynamic-pauses-the-interrupt-function)

---

## 1. Compile-Time Interrupts (Static Pauses)
LangGraph allows you to define static pause points at compile time (covered in [01_interrupt_before_after.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/05_human_in_the_loop/01_interrupt_before_after.py)).
- **`interrupt_before`**: Pauses execution right *before* a specific node runs.
- **`interrupt_after`**: Pauses execution right *after* a specific node finishes.
These require a checkpointer to save state at the pause point. Control is returned to the caller, and execution can be resumed by calling `.invoke(None, config)` with the same `thread_id`. Passing `None` as the input payload tells LangGraph to continue from the saved checkpoint.

```python
# Pause execution before executing the "action" node
graph = builder.compile(checkpointer=memory, interrupt_before=["action"])
```

---

## 2. Modifying State Mid-Run (State Updates)
Pausing execution allows humans to inspect and modify the graph state before resuming (covered in [02_editing_state_before_resume.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/05_human_in_the_loop/02_editing_state_before_resume.py)).
- **`graph.update_state(config, values, as_node=None)`**: Writes new values into the thread's checkpoint.
- The update values flow through the state key's defined reducer (e.g. appending to lists).
- Specifying `as_node` writes the state as if the specified node had returned the update, updating the graph's execution history.
- The next `.invoke(None, config)` call resumes using the updated state.

---

## 3. Gateways & Tool Approval Workflows
A common HITL pattern is a **gatekeeper node** that prompts a user for approval before running sensitive operations (e.g., executing database writes or triggering API calls) (covered in [03_approval_workflows.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/05_human_in_the_loop/03_approval_workflows.py)).
- The graph pauses before the tool node using `interrupt_before`.
- The user reviews the planned action (inspected via `graph.get_state(config)`).
- The user updates the state with an approval/rejection flag.
- The graph resumes, and a conditional edge routes execution to either the tool node or an early exit path based on the user's decision.

---

## 4. Dynamic Pauses: The interrupt() Function
Compile-time interrupts pause execution at fixed node boundaries. For dynamic pauses triggered by conditions at runtime (e.g., prompting a user for input when the agent needs clarification), LangGraph provides the **`interrupt`** function (covered in [04_interrupt_function.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/05_human_in_the_loop/04_interrupt_function.py)):
- Calling `interrupt(payload)` inside a node immediately pauses the graph and returns the payload to the caller.
- To resume execution, the caller passes a **`Command(resume=...)`** back to the graph.
- The value passed in `Command(resume=...)` becomes the return value of the `interrupt` function inside the node, allowing execution to continue seamlessly.

```python
from langgraph.types import interrupt, Command

def ask_human_node(state: State):
    # Pause and return a question to the caller
    human_input = interrupt("What is your email?")
    # Resumes here when the caller passes Command(resume="sai@example.com")
    return {"email": human_input}
```
