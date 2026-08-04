"""
create_react_agent: A Quick Tool-Calling Agent
==================================================
`langgraph.prebuilt.create_react_agent` builds a complete ReAct-style
agent graph (LLM node + tool node + the conditional loop between them)
in a single function call, instead of hand-wiring StateGraph nodes and
edges yourself. It's the fastest way to get a working tool-using agent,
and the result is still a normal compiled graph - you can stream it,
checkpoint it, and inspect its state exactly like a hand-built one.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama


@tool
def get_stock_price(ticker: str) -> str:
    """Look up the current price for a stock ticker symbol."""
    fake_prices = {"ACME": 42.10, "WIDGE": 87.55}
    return f"${fake_prices.get(ticker.upper(), 0.0)}"


@tool
def convert_currency(amount: float, to_currency: str) -> str:
    """Convert a USD amount to another currency (toy fixed rates)."""
    rates = {"EUR": 0.92, "GBP": 0.79}
    return f"{amount * rates.get(to_currency.upper(), 1.0):.2f} {to_currency.upper()}"


llm = ChatOllama(model="qwen2.5:1.5b")

# One call builds the full graph: model node, tool node, and the
# conditional edge that loops back to the model after each tool call.
agent = create_react_agent(llm, tools=[get_stock_price, convert_currency])

result = agent.invoke({
    "messages": [HumanMessage(content="What's ACME's price, converted to EUR?")]
})
for m in result["messages"]:
    print(f"{m.__class__.__name__}: {getattr(m, 'content', '')}")
# -> HumanMessage: What's ACME's price, converted to EUR?
# -> AIMessage: (tool_calls=[get_stock_price(ticker='ACME')])
# -> ToolMessage: $42.1
# -> AIMessage: (tool_calls=[convert_currency(amount=42.1, to_currency='EUR')])
# -> ToolMessage: 38.73 EUR
# -> AIMessage: ACME is trading at $42.10, which is about 38.73 EUR.

# The returned `agent` is a normal compiled graph -- it accepts a
# checkpointer, streams like any other graph, etc:
# agent = create_react_agent(llm, tools=[...], checkpointer=MemorySaver())
