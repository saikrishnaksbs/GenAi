"""
LANGCHAIN AGENT CREATION (create_react_agent & create_tool_calling_agent)
==========================================================================
This file demonstrates how to build agents using standard LangChain factory
functions. These function calls construct a Runnable agent definition, which is
then executed by AgentExecutor.

- create_tool_calling_agent: Uses the LLM's native function/tool-calling APIs.
- create_react_agent: Uses the classic Thought-Action-Observation text prompt loop
  (compatible with models that don't have native tool calling).

NOTE: While these agent constructors are still standard, `AgentExecutor` is now
considered legacy in favor of LangGraph. For production systems, you should build
agents as state graphs (see `02_langgraph_agent.py` or `03_agent_custom_graph.py`).
"""

from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_ollama import ChatOllama

# ==========================================
# 0. TOOL DEFINITION
# ==========================================
# Agents require "tools" that they can invoke to query external data or perform actions.
# We define a tool using the `@tool` decorator. The function's docstring is crucial
# because it acts as the tool description, which the LLM reads to decide when to call it.
@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)

# Define the model to use and bind the tools to a list
model = ChatOllama(model="qwen2.5:1.5b")
tools = [get_word_length]


# ==========================================
# 1. MODERN TOOL-CALLING AGENT
# ==========================================
# Tool-calling agents use the native API of models that have been fine-tuned for tool use
# (e.g., OpenAI, Claude, Gemini, Qwen, Ollama models supporting tool calls). 
# The model outputs a structured payload (JSON-like tool calls) instead of raw text.

print("=== Running Tool-Calling Agent ===")

# Create a prompt template. The prompt must contain a placeholder for the "agent_scratchpad".
# The `agent_scratchpad` is filled by LangChain with intermediate messages representing
# the history of tool calls and tool responses so the LLM knows what it has already done.
tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools. Use tools to verify your answers."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # Essential: holds the sequence of tool calls and results
])

# create_tool_calling_agent builds a Runnable that:
#   1. Binds the tools to the model (informs the model about the schema/definitions of available tools).
#   2. Passes the input and scratchpad history to the model.
#   3. Parses the model's output into ToolAgentAction or AgentFinish structures.
tool_agent = create_tool_calling_agent(model, tools, tool_calling_prompt)

# AgentExecutor is the runtime that executes the agent. It manages:
#   1. Feeding the user input and scratchpad history to the agent.
#   2. Checking if the agent output is a tool call or a final answer.
#   3. If it is a tool call: running the tool, appending the tool result to the scratchpad, and looping back.
#   4. If it is a final answer: returning it to the user.
tool_executor = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)

# Run the agent using `invoke`
res1 = tool_executor.invoke({"input": "How many letters are in 'LangChain'?"})
print("Result:", res1["output"])


# ==========================================
# 2. REACT-STYLE TEXT AGENT
# ==========================================
# ReAct (Reason + Act) is a classic prompting paradigm designed for models that do not
# support native tool-calling APIs. It forces the model to reason in a text loop:
# Thought -> Action -> Action Input -> Observation -> Thought...
#
# The model must output plain text matching a strict structure (e.g., "Thought: I need to call X... Action: X...")
# which AgentExecutor parses to execute tools.

print("\n=== Running ReAct Agent ===")

# Pull the standard ReAct template from the official LangChain Hub: "hwchase17/react"
# Under the hood, this prompt requires specific input variables:
#   - {tools}: Descriptions of the tools.
#   - {tool_names}: A list of names of the tools.
#   - {agent_scratchpad}: A text log of previous Thoughts, Actions, and Observations.
#   - {input}: The user's query.
react_prompt = hub.pull("hwchase17/react")

# create_react_agent builds a Runnable that:
#   1. Formats the tool descriptions and inserts them into the ReAct prompt template.
#   2. Feeds the formatted text (including the textual scratchpad) to the model.
#   3. Parses the plain text response to detect actions (e.g. "Action: get_word_length") or final answer.
react_agent = create_react_agent(model, tools, react_prompt)

# AgentExecutor runs the loop. When a tool is called, it appends the tool's return value
# to the prompt as "Observation: <output>" so the model can read it in the next step.
react_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

# Run the agent using `invoke`
res2 = react_executor.invoke({"input": "How many letters are in 'LangGraph'?"})
print("Result:", res2["output"])

