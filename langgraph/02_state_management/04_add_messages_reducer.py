"""
The add_messages Reducer for Chat History
=============================================
`add_messages` is a purpose-built reducer for lists of chat messages
(`HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`, etc). Like
`operator.add`, it appends new messages to the list - but it also
understands message identity: if a new message has the same `id` as an
existing one, it *replaces* that message in place instead of duplicating
it (useful for streaming edits or corrections). This is why almost every
LangGraph chatbot/agent state includes
`messages: Annotated[list[AnyMessage], add_messages]`.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot_node(state: ChatState) -> dict:
    # In real code this would call a chat model, e.g.:
    # response = ChatOllama(model="qwen2.5:1.5b").invoke(state["messages"])
    last_user_msg = state["messages"][-1].content
    response = AIMessage(content=f"You said: {last_user_msg}")
    # add_messages appends this to the existing list rather than replacing it.
    return {"messages": [response]}


print("===============================================================================")
print("            LANGGRAPH STATE MANAGEMENT: ADD_MESSAGES REDUCER                   ")
print("===============================================================================\n")

builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

print("💬 [Turn 1] User message sent:")
state = {"messages": [HumanMessage(content="Hello there")]}
result = graph.invoke(state)
for msg in result["messages"]:
    role = "👤 Human" if isinstance(msg, HumanMessage) else "🤖 AI   "
    print(f"   {role} : {msg.content}")
print()

# Multi-turn: pass the accumulated messages back in for the next turn.
print("💬 [Turn 2] Follow-up message sent:")
next_turn = graph.invoke({"messages": result["messages"] + [HumanMessage(content="And you?")]})
print(f"   🤖 AI Response : '{next_turn['messages'][-1].content}'\n")

# Replacing a message by id instead of appending: give the new message
# the same `.id` as an existing one and add_messages will overwrite it.
print("✏️ [In-Place Edit] Replacing AI response by ID:")
edited = AIMessage(content="Corrected response", id=result["messages"][-1].id)
corrected = add_messages(result["messages"], [edited])
print(f"   • Original Count  : {len(result['messages'])}")
print(f"   • Count After Edit: {len(corrected)} (In-place edit replaced existing message)")
print(f"   • Updated Message : '{corrected[-1].content}'\n")

