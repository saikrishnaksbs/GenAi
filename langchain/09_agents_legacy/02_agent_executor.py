"""
AgentExecutor
===============
AgentExecutor is the runtime loop that drives an agent: it repeatedly calls
the agent to decide the next action, executes any requested tool, feeds the
result back in, and stops when the agent returns a final answer. It also
handles parsing errors, timeouts, and iteration limits.

(In modern LangGraph-based code, AgentExecutor's job is replaced by a graph
with a loop between an "agent" node and a "tools" node — see LangGraph docs.)
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


model = ChatOllama(model="qwen2.5:1.5b")
tools = [add, multiply]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math assistant. Use tools for all calculations."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(model, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,               # hard cap on tool-call loops
    max_execution_time=30,           # seconds, wall-clock timeout
    handle_parsing_errors=True,      # feed parsing errors back to the model instead of crashing
    return_intermediate_steps=True,  # include the (action, observation) trace in the output
)

result = executor.invoke({"input": "What is (3 + 4) multiplied by 5?"})
print(result["output"])
print(result["intermediate_steps"])  # list of (AgentAction, observation) tuples
