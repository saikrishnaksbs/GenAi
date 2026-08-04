# Legacy LangChain Agents

This directory details LangChain's legacy **Agent** system, built around the `AgentExecutor` runtime. While modern architectures use **LangGraph** to build loops and multi-agent systems, understanding these legacy APIs is vital for maintaining existing codebases and learning the fundamental mechanics of agent loop executions (ReAct reasoning, scratchpads, and execution boundaries).

---

## Table of Contents
1. [Agent Factory Methods: ReAct vs. Native Tool Calling](#1-agent-factory-methods-react-vs-native-tool-calling)
2. [The AgentExecutor Runtime Loop](#2-the-agentexecutor-runtime-loop)
3. [Agent Primitives: AgentAction vs. AgentFinish](#3-agent-primitives-agentaction-vs-agentfinish)
4. [Scratchpads & Intermediate Steps](#4-scratchpads--intermediate-steps)
5. [Loop Guards: Timeout and Iteration Limits](#5-loop-guards-timeout-and-iteration-limits)

---

## 1. Agent Factory Methods: ReAct vs. Native Tool Calling
LangChain provides factory functions to compile a prompt, a model, and a list of tools into an agent (details in [01_create_react_and_tool_calling_agents.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents_legacy/01_create_react_and_tool_calling_agents.py)):
- **`create_react_agent`**: Implements the classic ReAct prompt template loop (Thought -> Action -> Observation -> Thought). Works with any base text model.
- **`create_tool_calling_agent`**: Utilizes the model provider's native tool-calling interface (e.g. Anthropic, OpenAI, Gemini). This is the recommended style for legacy agents.
- **`create_openai_functions_agent`**: Predecessor to the tool calling agent, built specifically for OpenAI's legacy function-calling API format.

---

## 2. The AgentExecutor Runtime Loop
[AgentExecutor](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents_legacy/02_agent_executor.py#L4-L10) is the execution engine that coordinates the agent.
- It queries the agent for the next action.
- Runs the corresponding tool.
- Feeds the tool output back into the conversation memory.
- Loops until the agent decides to finish.

In modern LangGraph-based applications, this orchestrator is replaced by a compiled state graph containing nodes and circular edges.

---

## 3. Agent Primitives: AgentAction vs. AgentFinish
At each step, the agent outputs one of two primitives (covered in [03_agent_action_and_finish.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents_legacy/03_agent_action_and_finish.py)):
- **`AgentAction`**: Signals that a tool needs to be executed. Contains the tool name (`tool`), the target arguments (`tool_input`), and the generated reasoning text (`log`).
- **`AgentFinish`**: Signals that the agent has finished its work. Contains the final answer payload (`return_values`) and the raw completion text (`log`).

```python
# The internal AgentExecutor loop:
response = agent.invoke(state)
if isinstance(response, AgentAction):
    # Execute tool, append to history, and repeat
elif isinstance(response, AgentFinish):
    # Stop and return answer
```

---

## 4. Scratchpads & Intermediate Steps
To maintain reasoning context across loops, the execution history is stored as a list of `(AgentAction, tool_output)` tuples called **intermediate steps** (detailed in [04_intermediate_steps_and_scratchpad.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents_legacy/04_intermediate_steps_and_scratchpad.py)).
- Before each model call, these tuples are formatted into a single string template variable named **`agent_scratchpad`**.
- This lets the model see its previous thoughts and tool outputs, preventing it from repeating the same actions.

---

## 5. Loop Guards: Timeout and Iteration Limits
Autonomous loops can get stuck in infinite execution cycles if the model hallucinates or fails to parse tool outputs. `AgentExecutor` provides safety parameters to guard against this (covered in [05_max_iterations_and_early_stopping.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/09_agents_legacy/05_max_iterations_and_early_stopping.py)):
- **`max_iterations`**: Caps the maximum number of reasoning loops (e.g. 5 steps).
- **`max_execution_time`**: A wall-clock timeout limit in seconds.
- **`early_stopping_method`**: Determines how to handle timeouts. Setting this to `"force"` returns the last generated response immediately, while `"generate"` makes a final call to the model asking for a summary of the work completed so far.
