# LangGraph Subgraphs & Reusability

This directory details graph composition patterns in LangGraph. It covers nesting compiled graphs as nodes, managing state mapping (shared vs. separate schemas), implementing adapter nodes to encapsulate subgraphs, and building reusable graph factory functions.

---

## Table of Contents
1. [Nesting Graphs: Compiled Graph as a Node](#1-nesting-graphs-compiled-graph-as-a-node)
2. [Automatic Mapping via Shared State Schemas](#2-automatic-mapping-via-shared-state-schemas)
3. [Encapsulating State via Adapter Nodes](#3-encapsulating-state-via-adapter-nodes)
4. [Reusable Graph Factories](#4-reusable-graph-factories)

---

## 1. Nesting Graphs: Compiled Graph as a Node
A compiled `StateGraph` is a Runnable. As a result, you can add it directly as a node in a **parent graph** (covered in [01_compiled_graph_as_node.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/07_subgraphs_and_reuse/01_compiled_graph_as_node.py)):

```python
subgraph = sub_builder.compile()
parent_builder.add_node("subgraph_step", subgraph)
```
When execution reaches the `"subgraph_step"` node, LangGraph runs the subgraph, passing the parent state in and merging the returned values back once it completes.

---

## 2. Automatic Mapping via Shared State Schemas
If the parent graph and the subgraph share the same state schema (or share matching key names), LangGraph wires their channels together automatically.
- Matching keys are extracted from the parent state.
- Passed as input to start the subgraph execution.
- Any keys modified by the subgraph are updated in the parent state using the parent's registered reducers.

---

## 3. Encapsulating State via Adapter Nodes
If a subgraph requires private, internal keys (e.g., intermediate steps or custom counters) that the parent graph should not see, you must isolate their state schemas (covered in [02_shared_vs_separate_state_schemas.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/07_subgraphs_and_reuse/02_shared_vs_separate_state_schemas.py)).
- An **Adapter Node** (a plain Python function) is registered in the parent graph.
- The adapter:
  1. Extracts parent values and constructs the subgraph's input dictionary.
  2. Invokes the subgraph synchronously or asynchronously.
  3. Maps the subgraph output back to the parent state schema, filtering out private subgraph fields.

This pattern isolates the subgraph, preventing namespace pollution in the parent state.

---

## 4. Reusable Graph Factories
To reuse a graph pattern across different projects or configuration scopes (e.g. running the same validation loop with different prompts or validation tools), define a **Graph Factory Function** (covered in [03_reusable_graph_factories.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/07_subgraphs_and_reuse/03_reusable_graph_factories.py)).
- Wrapping graph definition in a function prevents sharing mutable builder states.
- Allows parameterizing the graph definition:
```python
def make_editor_graph(model, validator_tool) -> CompiledStateGraph:
    builder = StateGraph(EditorState)
    # Define nodes using configured model and tool arguments
    ...
    return builder.compile()
```
Each invocation returns a freshly compiled, isolated graph instance.
