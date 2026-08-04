# LangGraph Time Travel & Replay

This directory details LangGraph's time-travel capabilities. It covers listing a thread's history of checkpoints, replaying historical execution paths verbatim, and branching/forking new execution timelines off past states.

---

## Table of Contents
1. [Time Travel Overview](#time-travel-overview)
2. [Listing Checkpoint History (get_state_history)](#2-listing-checkpoint-history-get_state_history)
3. [Replaying from a Checkpoint](#3-replaying-from-a-checkpoint)
4. [Timeline Forking (Git-Style Branching)](#4-timeline-forking-git-style-branching)

---

## 1. Time Travel Overview
Because LangGraph persists a StateSnapshot after every super-step, the execution history is recorded as a chain of immutable checkpoints. This structure allows you to query past states, reproduce bugs by re-running steps from a specific moment, or fork execution down a new path.

---

## 2. Listing Checkpoint History (get_state_history)
You can inspect all historical states of a thread using [`.get_state_history()`](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/08_time_travel_replay/01_get_state_history.py#L4-L10).
- It returns an iterator of `StateSnapshot` objects, sorted from newest to oldest.
- Each snapshot exposes the state values, the queued `next` node to execute, and the unique `checkpoint_id` corresponding to that step.

```python
config = {"configurable": {"thread_id": "session_123"}}
for snapshot in graph.get_state_history(config):
    print(f"Checkpoint: {snapshot.config['configurable']['checkpoint_id']}")
    print(f"State Values: {snapshot.values}")
```

---

## 3. Replaying from a Checkpoint
To resume execution from a past point in time without modifying the history, pass the specific checkpoint's configuration (containing both `thread_id` and **`checkpoint_id`**) to `.invoke()` (covered in [02_replay_from_checkpoint.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/08_time_travel_replay/02_replay_from_checkpoint.py)):

```python
replay_config = {
    "configurable": {
        "thread_id": "session_123",
        "checkpoint_id": "1ef5e-abc-123",  # Past checkpoint ID
    }
}
# Resume execution from that checkpoint
graph.invoke(None, config=replay_config)
```
LangGraph loads the state of that checkpoint and executes the node that was queued next at that moment.

---

## 4. Timeline Forking (Git-Style Branching)
You can fork a new execution timeline from a past checkpoint (covered in [03_forking_execution.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/08_time_travel_replay/03_forking_execution.py)).
- To fork:
  1. Retrieve a historical `checkpoint_id`.
  2. Call `graph.update_state()` passing the historical config and the modified state values.
  3. This creates a **new fork checkpoint** branching off the historical one.
  4. Invoke the graph using the new config.

The original timeline remains intact, while a new branch of checkpoints grows from the fork point, similar to branching in Git. This is useful for exploring alternative scenarios or correcting agent errors mid-execution.
