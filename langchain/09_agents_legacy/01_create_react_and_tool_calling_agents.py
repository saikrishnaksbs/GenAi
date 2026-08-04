"""
create_react_agent, create_tool_calling_agent, create_openai_functions_agent
================================================================================
Factory functions that wire together a model + a prompt + a list of tools into
an "agent" — a Runnable that decides, at each step, whether to call a tool or
give a final answer. They differ in the underlying reasoning strategy:

- create_react_agent: uses the classic ReAct pattern (Thought/Action/
  Observation text loop) — works with any text-completion-capable model,
  even ones without native tool-calling.
- create_tool_calling_agent: uses the model provider's native tool-calling
  API (works with Anthropic, OpenAI, etc. via bind_tools). Preferred today.
- create_openai_functions_agent: predecessor to tool_calling_agent, specific
  to OpenAI's original "functions" API. Mostly superseded now.
"""

from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_community.chat_models import ChatOllama


@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)


model = ChatOllama(model="qwen2.5:1.5b")
tools = [get_word_length]

# --- Modern approach: tool-calling agent ---
tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # holds intermediate tool call/results
])
agent = create_tool_calling_agent(model, tools, tool_calling_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor.invoke({"input": "How many letters are in 'LangChain'?"}))

# --- ReAct-style agent: relies on parsing "Thought/Action/Observation" text ---
react_prompt = hub.pull("hwchase17/react")  # standard ReAct prompt template
react_agent = create_react_agent(model, tools, react_prompt)
react_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
print(react_executor.invoke({"input": "How many letters are in 'LangGraph'?"}))
