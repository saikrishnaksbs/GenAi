"""
TOOLNODE-STYLE EXECUTION AND ERROR HANDLING
==============================================
When a model returns multiple tool calls at once, you need a small runtime
loop that executes each one and turns exceptions into ToolMessages instead
of crashing the whole chain. LangGraph's `ToolNode` does this out of the
box; here we build the same pattern by hand with plain LangChain so the
mechanics are clear, then show the LangGraph shortcut.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b  # raises ZeroDivisionError if b == 0, deliberately unguarded


@tool
def lookup_capital(country: str) -> str:
    """Return the capital city of a country."""
    capitals = {"france": "Paris", "japan": "Tokyo"}
    if country.lower() not in capitals:
        raise ValueError(f"No capital on file for '{country}'")
    return capitals[country.lower()]


tools_by_name = {"divide": divide, "lookup_capital": lookup_capital}


def execute_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    """Hand-rolled equivalent of LangGraph's ToolNode: run every requested
    tool call and convert failures into a ToolMessage the model can see,
    rather than raising and aborting the whole conversation."""
    results = []
    for call in ai_message.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        try:
            output = tool_fn.invoke(call["args"])
            content = str(output)
        except Exception as exc:
            # Surfacing the error as tool output lets the model retry with
            # corrected arguments instead of the program crashing.
            content = f"Error: {exc}"
        results.append(ToolMessage(content=content, tool_call_id=call["id"]))
    return results


model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
model_with_tools = model.bind_tools([divide, lookup_capital])

messages = [HumanMessage("What's 10 divided by 0, and what's the capital of Mars?")]
ai_message = model_with_tools.invoke(messages)

tool_messages = execute_tool_calls(ai_message)
for tm in tool_messages:
    print(tm.content)
# -> "Error: float division by zero"
# -> "Error: No capital on file for 'Mars'"

messages += [ai_message, *tool_messages]
final = model_with_tools.invoke(messages)
print(final.content)
# -> "I couldn't divide by zero, and Mars doesn't have an ISO-recognized capital city."


# --- The LangGraph shortcut -------------------------------------------------
# LangGraph's prebuilt ToolNode implements the same execute-and-catch loop,
# plus integrates directly into a StateGraph's message-passing state.
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([divide, lookup_capital])
node_result = tool_node.invoke({"messages": [ai_message]})
print(node_result["messages"][0].content)
# -> "Error: float division by zero"  (same error-as-content behavior as above)
