# Modern LangChain Agents (LangGraph)

This directory details how to build and execute autonomous agents in LangChain. It explains the transition from the legacy `AgentExecutor` runtime to **LangGraph**, which is the current standard for constructing complex stateful, multi-actor agent loops.

---

## Table of Contents
1. [LangChain Agent Constructors](#1-langchain-agent-constructors)
2. [LangGraph Prebuilt Agents](#2-langgraph-prebuilt-agents)
3. [Custom StateGraphs (Custom Agent Loop Mechanics)](#3-custom-stategraphs-custom-agent-loop-mechanics)
4. [Recursion Limits & Graceful Early Stopping](#4-recursion-limits--graceful-early-stopping)
5. [Persistence and Thread Memory](#5-persistence-and-thread-memory)
6. [Human-in-the-Loop Breakpoints](#6-human-in-the-loop-breakpoints)

---

## 1. LangChain Agent Constructors
While LangGraph is preferred for execution, LangChain's built-in prompt compiling and model-tool binding utilities are still widely used to define agent logic.
- **File**: [01_create_react_and_tool_calling_agents.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents/01_create_react_and_tool_calling_agents.py)
- **`create_tool_calling_agent`**: Builds a runnable agent using the LLM's native tool/function calling APIs.
- **`create_react_agent`**: Builds a runnable agent that parses text-based Thought/Action/Observation loops.
- **`AgentExecutor`**: The legacy execution runtime wrapper. (Included for compatibility and transition demonstration).

---

## 2. LangGraph Prebuilt Agents
In modern systems, `AgentExecutor` is replaced by state graphs compiled via LangGraph. 
- **File**: [02_langgraph_agent.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents/02_langgraph_agent.py)
- **`create_react_agent` from `langgraph.prebuilt`**: A pre-compiled LangGraph workflow that executes a model-tool loop. It matches `AgentExecutor`'s functionality out-of-the-box but runs as a compiled graph, natively supporting message state, streaming updates, and checkpointing.

---

## 3. Custom StateGraphs (Custom Agent Loop Mechanics)
For ultimate control over routing and memory, you can design the agent loop manually using a custom `StateGraph`. This removes the need for manual `AgentAction` or `AgentFinish` parsing and raw scratchpads.
- **File**: [03_agent_custom_graph.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents/03_agent_custom_graph.py)
- **State**: A dictionary structure (e.g. `messages` tracking) that carries data across graph nodes.
- **Nodes**: Functions that perform actions (e.g., calling the model or executing tools) and return updates to the state.
- **Edges & Conditional Edges**: Determine control-flow routing (e.g., checking if the model output contains tool calls, looping back to the agent or terminating the run via `END`).

---

## 4. Recursion Limits & Graceful Early Stopping
Guarding agents from entering infinite execution loops is critical when working with autonomous systems.
- **File**: [04_langgraph_early_stopping_and_max_loops.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents/04_langgraph_early_stopping_and_max_loops.py)
- **Recursion Limit**: Specify `recursion_limit` in the run config (e.g. `{"recursion_limit": 5}`) to raise a `GraphRecursionError` if execution goes over the allotted step budget.
- **Graceful Early Stopping Node**: Track `loop_count` inside the graph state, checking it via conditional edges. This allows the graph to route to a fallback node and output a clean, custom message (e.g. "iteration cap reached") instead of throwing a hard exception.

---

## 5. Persistence and Thread Memory
State snapshots allow agents to remember context across independent executions.
- **File**: [05_langgraph_persistence_memory.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents/05_langgraph_persistence_memory.py)
- **Checkpointers**: Instantiating graphs with `MemorySaver` registers state snapshot mechanisms.
- **Threads**: Passing a unique `thread_id` in the `config` loads session memory automatically. Different threads run completely isolated from each other.

---

## 6. Human-in-the-Loop Breakpoints
Ensures safety for destructive or sensitive tools by halting loops for operator confirmation.
- **File**: [06_langgraph_human_in_the_loop.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents/06_langgraph_human_in_the_loop.py)
- **Breakpoints**: Compiling with `interrupt_before=["tools"]` triggers a pause before tools run.
- **Inspecting & Resuming**: Use `get_state()` to print pending tool inputs, then call the compiled application with `None` as the input to resume execution.

