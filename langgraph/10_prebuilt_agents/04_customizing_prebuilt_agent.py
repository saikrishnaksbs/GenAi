"""
Customizing a Prebuilt Agent's Prompt and State Schema
===========================================================
`create_react_agent` accepts a `prompt` (a system message, string, or a
callable that builds messages from state) to control the agent's
behavior without hand-building the graph, and a `state_schema` to extend
the default `{messages}` state with extra fields your prompt function or
tools need (e.g. a `user_id` for personalization). This covers most
customization needs while still keeping the one-line construction.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent


@tool
def check_account_balance(account_id: str) -> str:
    """Check the balance for an account id."""
    return f"Account {account_id} has a balance of $1,240.50"


# Extend the default {messages} schema with a custom field the prompt
# function can read from.
class CustomAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_tier: str


def build_prompt(state: CustomAgentState) -> list:
    # A callable prompt receives the full custom state and returns the
    # message list to send to the model -- lets you inject dynamic,
    # per-request system instructions instead of a single static string.
    tier = state.get("user_tier", "standard")
    tone = "extra courteous and detailed" if tier == "premium" else "concise"
    system = SystemMessage(content=f"You are a banking assistant. Be {tone}.")
    return [system] + state["messages"]


llm = ChatOllama(model="qwen2.5:1.5b")

agent = create_react_agent(
    llm,
    tools=[check_account_balance],
    prompt=build_prompt,
    state_schema=CustomAgentState,
)

result = agent.invoke({
    "messages": [HumanMessage(content="What's my balance on account A-100?")],
    "user_tier": "premium",
})
print(result["messages"][-1].content)
# -> "Thank you for reaching out! I'm happy to help -- your account A-100
#     currently has a balance of $1,240.50. Let me know if there's anything
#     else you'd like to review."  (premium tone, per the custom prompt fn)

# A simpler static-string prompt also works when no per-request logic is needed:
# agent = create_react_agent(llm, tools=[...], prompt="You are a terse assistant.")
