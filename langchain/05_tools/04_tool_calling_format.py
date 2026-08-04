"""
TOOL CALLING / FUNCTION CALLING FORMAT
=========================================
Chat models support "tool calling": you give the model a list of tool
schemas via `.bind_tools([...])`, and if the model decides a tool is
needed, it returns an `AIMessage` whose `.tool_calls` list describes which
tool(s) to invoke and with what arguments — instead of (or alongside) a
normal text response.

The model never actually executes the tool; it only requests the call.
Your code is responsible for running the tool and feeding the result back.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama


@tool
def get_stock_price(ticker: str) -> float:
    """Look up the current stock price for a ticker symbol."""
    prices = {"AAPL": 227.50, "GOOG": 168.30}
    return prices.get(ticker.upper(), 0.0)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert an amount from one currency to another."""
    rate = 0.92 if (from_currency, to_currency) == ("USD", "EUR") else 1.0
    return round(amount * rate, 2)


model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# bind_tools attaches the tool JSON schemas to every request this model
# makes, without changing the model's return type (still an AIMessage).
model_with_tools = model.bind_tools([get_stock_price, convert_currency])

messages = [HumanMessage("What is Apple's stock price, in euros?")]
ai_message = model_with_tools.invoke(messages)

print(ai_message.tool_calls)
# -> [{"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_abc123", "type": "tool_call"}]
print(ai_message.content)
# -> ""  (often empty when the model only wants to call a tool)

messages.append(ai_message)

# You execute each requested tool call yourself and report results back
# as ToolMessage objects, matched to the call by `tool_call_id`.
tool_registry = {"get_stock_price": get_stock_price, "convert_currency": convert_currency}

for call in ai_message.tool_calls:
    selected_tool = tool_registry[call["name"]]
    output = selected_tool.invoke(call["args"])
    messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

# Feed the tool results back so the model can produce a final answer,
# possibly issuing a second round of tool calls (e.g. convert_currency).
final_response = model_with_tools.invoke(messages)
print(final_response.content)
# -> "Apple's stock price is approximately 209.30 EUR."

# `bind_tools` also accepts `tool_choice` to force (or forbid) tool use:
forced = model.bind_tools([get_stock_price], tool_choice="get_stock_price")
forced_response = forced.invoke([HumanMessage("Never mind, just say hello.")])
print(forced_response.tool_calls[0]["name"])
# -> "get_stock_price"  (forced, even though the prompt didn't need it)
