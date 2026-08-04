# LangGraph Prebuilt Agents & Components

This directory details LangGraph's prebuilt agent constructors and components. It covers the quick-start `create_react_agent` constructor, the `ToolNode` tool-execution wrapper, the `tools_condition` router helper, and methods to customize schemas and prompts.

---

## Table of Contents
1. [create_react_agent (One-Line Agent Creation)](#1-create_react_agent-one-line-agent-creation)
2. [ToolNode (Parallel Tool Execution Node)](#2-toolnode-parallel-tool-execution-node)
3. [tools_condition (Built-In Routing Logic)](#3-tools_condition-built-in-routing-logic)
4. [Customizing System Prompts and State Schemas](#4-customizing-system-prompts-and-state-schemas)

---

## 1. create_react_agent (One-Line Agent Creation)
For standard tool-calling agents, hand-wiring nodes and loops adds unnecessary boilerplate. LangGraph provides **`create_react_agent`** (covered in [01_create_react_agent.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/10_prebuilt_agents/01_create_react_agent.py)):
- Compiles an LLM node, a tool node, and a conditional edge loop.
- The returned graph is a compiled `CompiledStateGraph` that supports all standard capabilities (streaming, checkpointing, and time-travel).

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools=[get_weather], checkpointer=memory)
```

---

## 2. ToolNode (Parallel Tool Execution Node)
[ToolNode](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/10_prebuilt_agents/02_toolnode_component.py#L4-L12) is the execution engine behind `create_react_agent`.
- It accepts a list of tools.
- When invoked, it inspects the latest `AIMessage` in the state for `tool_calls`.
- It executes all requested tool calls **in parallel** using a thread pool.
- Automatically catches execution exceptions, converting them into `ToolMessage` payloads with the error details so the model can observe and fix its mistakes.

---

## 3. tools_condition (Built-In Routing Logic)
To wire up a manual loop between an agent node and a tool execution node, you can use the prebuilt **`tools_condition`** routing function instead of writing custom checks (covered in [03_tools_condition_helper.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/10_prebuilt_agents/03_tools_condition_helper.py)):
- Inspects the latest message in the state.
- Returns `"tools"` if the message contains a `tool_calls` list.
- Returns `"__end__"` if the message does not contain any tool calls.
- Snaps together with `ToolNode` (which registers under the name `"tools"` by default).

```python
from langgraph.prebuilt import tools_condition

# Add the routing edge to the graph
builder.add_conditional_edges("agent_node", tools_condition)
```

---

## 4. Customizing System Prompts and State Schemas
While prebuilt agents are fast to instantiate, they still allow customization (covered in [04_customizing_prebuilt_agent.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/10_prebuilt_agents/04_customizing_prebuilt_agent.py)):
- **Prompts**: Configure system instructions by passing a string (e.g. `state_modifier="You are a SQL expert"`), a `SystemMessage` object, or a callable that constructs message histories dynamically based on the state.
- **State Schema**: Extend the default state (which only tracks `{messages}`) by passing a custom schema to `state_schema`. This allows you to track custom keys (e.g., `user_id` or `context`) across node executions.

```python
# Create an agent with system instructions and a custom state schema
agent = create_react_agent(
    model,
    tools=[get_weather],
    state_modifier="You are a polite assistant.",
    state_schema=CustomStateSchema
)
```
