# LangGraph Persistence & Checkpointing

This directory details checkpointers in LangGraph—the mechanism that persists the state of a graph at every step. Checkpointing enables stateful, multi-turn conversations, process-safe interrupts, error recovery, and time travel.

---

## Table of Contents
1. [What is a Checkpointer?](#1-what-is-a-checkpointer)
2. [Ephemeral vs. Durable Checkpointers](#2-ephemeral-vs-durable-checkpointers)
3. [Multi-Turn Conversations via thread_id](#3-multi-turn-conversations-via-thread_id)
4. [Inspecting State Snapshots](#4-inspecting-state-snapshots)

---

## 1. What is a Checkpointer?
A checkpointer is a persistence layer injected when compiling a graph. After every super-step, LangGraph automatically saves a serialized snapshot of the state schema.

Passing a checkpointer to [`.compile()`](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/04_persistence_checkpointing/01_memory_saver.py#L9) unlocks stateful features:
- In-context memory across separate invocations.
- Thread control loops.
- Human-in-the-loop interrupts and execution edits.
- Time-travel debugger rollbacks.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

---

## 2. Ephemeral vs. Durable Checkpointers
LangGraph provides plug-and-play checkpointer engines:
- **`MemorySaver`**: Stores states in volatile RAM. Ephemeral (disappears when the Python process exits). Perfect for testing, debugging, and local experiments (details in [01_memory_saver.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/04_persistence_checkpointing/01_memory_saver.py)).
- **`SqliteSaver`**: Writes state changes to a local SQLite `.db` file. Survives process restarts, providing basic local persistence.
- **`PostgresSaver`**: Writes state changes to a PostgreSQL server database. Ideal for production applications requiring high availability, concurrent thread access, and long-term durability.

Durable checkpointers are demonstrated in [02_sqlite_postgres_saver.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/04_persistence_checkpointing/02_sqlite_postgres_saver.py). They support connection string setups and setup helper scripts to bootstrap required table structures automatically.

---

## 3. Multi-Turn Conversations via thread_id
To tie separate invocations into a single continuous session, pass a **`thread_id`** in the configuration dictionary (covered in [03_thread_id_multiturn.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/04_persistence_checkpointing/03_thread_id_multiturn.py)):

```python
config = {"configurable": {"thread_id": "user_session_42"}}

# First turn
graph.invoke({"messages": [HumanMessage(content="My name is Sai")]}, config=config)

# Second turn - resumes from latest checkpoint, remembers "Sai"
graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=config)
```
The checkpointer automatically retrieves the latest checkpoint matching the `thread_id` to initialize the starting state, eliminating the need to pass the full history manually on every invocation.

---

## 4. Inspecting State Snapshots
You can query the current checkpoint state of any thread using **`graph.get_state(config)`** (detailed in [04_inspecting_checkpoints.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/04_persistence_checkpointing/04_inspecting_checkpoints.py)).
This returns a **`StateSnapshot`** object containing:
- **`values`**: The current values of all state keys.
- **`next`**: A tuple of node names that are scheduled to execute next.
- **`config`**: The configuration dictionary (including the current `thread_id` and unique `checkpoint_id`).
- **`metadata`**: Metadata detailing the step index, source, and write actions.
