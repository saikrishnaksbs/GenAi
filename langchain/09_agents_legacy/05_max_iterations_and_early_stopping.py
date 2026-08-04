"""
Max Iterations and Early Stopping
====================================
Agents can loop indefinitely if the model keeps calling tools without ever
producing a final answer. AgentExecutor guards against this with:

- max_iterations: hard cap on the number of agent decision steps.
- max_execution_time: wall-clock timeout in seconds.
- early_stopping_method: what to do when a limit is hit — "force" makes the
  executor immediately return whatever it has as a canned "stopped early"
  response instead of raising an error.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama


@tool
def search_web(query: str) -> str:
    """Pretend web search that always returns an unhelpful result (to force looping)."""
    return "No definitive answer found, try again."


model = ChatOllama(model="qwen2.5:1.5b")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You must keep searching until you find a definitive answer."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(model, [search_web], prompt)

executor = AgentExecutor(
    agent=agent,
    tools=[search_web],
    max_iterations=5,                    # stop after 5 tool-call loops no matter what
    max_execution_time=15,               # or after 15 seconds, whichever comes first
    early_stopping_method="force",       # return a canned response instead of raising
    verbose=True,
)

result = executor.invoke({"input": "What is the exact population of Mars colonies today?"})
print(result["output"])
# -> Since the tool never gives a definitive answer, the executor hits max_iterations
#    and returns something like: "Agent stopped due to iteration limit or time limit."
