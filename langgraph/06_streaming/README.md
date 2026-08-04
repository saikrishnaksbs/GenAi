# LangGraph Streaming Modes

This directory details LangGraph's streaming capabilities. It covers yielding full state updates, node-level updates (diffs), real-time LLM token streams from inside nodes, and fine-grained event telemetry with the asynchronous `astream_events` protocol.

---

## Table of Contents
1. [Values Streaming (stream_mode="values")](#1-values-streaming-stream_modevalues)
2. [Updates Streaming (stream_mode="updates")](#2-updates-streaming-stream_modeupdates)
3. [Messages/Token Streaming (stream_mode="messages")](#3-messagestoken-streaming-stream_modemessages)
4. [Fine-Grained Telemetry (astream_events)](#4-fine-grained-telemetry-astream_events)

---

## 1. Values Streaming (stream_mode="values")
[Values Streaming](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/06_streaming/01_stream_mode_values.py#L4-L10) yields the **entire state dictionary** after each super-step completes.
- Each chunk contains a complete, self-consistent snapshot of the state schema.
- **Best Use Case**: Simplest to implement. Ideal for web frontends that want to re-render the absolute latest state without having to manually merge partial updates.

---

## 2. Updates Streaming (stream_mode="updates")
[Updates Streaming](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/06_streaming/02_stream_mode_updates.py#L4-L10) yields only the dictionary returned by the executing node.
- The output structure is mapped by node name: `{node_name: partial_state_update}`.
- **Best Use Case**: Bandwidth-efficient. Ideal for displaying progress logs (e.g., showing messages like `"Node [search] completed with keys: [docs]"`) to track which nodes have completed work.

---

## 3. Messages/Token Streaming (stream_mode="messages")
To stream tokens in real-time while a node is executing (such as a chat model generating text), set `stream_mode="messages"` (covered in [03_stream_mode_messages.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/06_streaming/03_stream_mode_messages.py)).
- Surfaces token chunks (**`AIMessageChunk`**) emitted by any chat model running inside any node of the graph.
- Each chunk is annotated with metadata identifying the source node.
- **Best Use Case**: Animating typing effects in chat user interfaces.

```python
# Stream messages
async for msg, metadata in graph.astream(input, config, stream_mode="messages"):
    # Yields token chunks in real-time
    print(msg.content, end="")
```

---

## 4. Fine-Grained Telemetry (astream_events)
For advanced telemetry, the **`astream_events`** protocol yields a detailed event log for every component executing inside the graph (covered in [04_astream_events.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/06_streaming/04_astream_events.py)).
- Exposes events such as:
  - `on_chain_start` / `on_chain_end`
  - `on_chat_model_stream`
  - `on_tool_start` / `on_tool_end`
- **Best Use Case**: Building complex UIs that need to display tool execution arguments, execution logs, and token streams concurrently, with clear lineage tracing back to each specific node.
