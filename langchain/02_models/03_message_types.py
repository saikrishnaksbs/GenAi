"""
MESSAGE TYPES
==============
Chat models communicate through typed message objects rather than plain
strings. Each message has a `role` baked into its class:

    SystemMessage   -> instructions/context that steer the model's behavior
    HumanMessage    -> input from the end user
    AIMessage       -> output produced by the model
    ToolMessage     -> the result of executing a tool the model requested
    FunctionMessage -> legacy predecessor to ToolMessage (single-function era)

Using explicit classes (instead of raw dicts) gives you type safety and
lets LangChain validate conversation structure before sending it to a provider.
"""

from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b")

# A conversation is just a list of messages, oldest first.
conversation = [
    SystemMessage(content="You are a terse assistant. Answer in one sentence."),
    HumanMessage(content="What is the capital of Japan?"),
    AIMessage(content="The capital of Japan is Tokyo."),
    HumanMessage(content="And its population?"),
]

response = model.invoke(conversation)
print(response)
# -> AIMessage(content="Tokyo's population is about 14 million.", ...)

# --- ToolMessage: reporting a tool call's result back to the model -----
# When a model requests a tool call, it emits an AIMessage with `tool_calls`.
# You execute the tool yourself, then send the result back as a ToolMessage,
# linked to the original call via `tool_call_id`.
tool_result = ToolMessage(
    content="72 degrees and sunny",
    tool_call_id="call_abc123",  # must match the id the model generated
)
print(tool_result)
# -> ToolMessage(content='72 degrees and sunny', tool_call_id='call_abc123')

# --- FunctionMessage: legacy, pre-dates parallel/multi tool calls ------
# Superseded by ToolMessage but still seen in older code and some
# provider-specific integrations. Prefer ToolMessage in new code.
legacy_function_result = FunctionMessage(
    content="72 degrees and sunny",
    name="get_weather",
)
print(legacy_function_result)
# -> FunctionMessage(content='72 degrees and sunny', name='get_weather')

# Every message also has a `.type` attribute used internally for role mapping.
for msg in conversation:
    print(msg.type, "->", msg.content)
# -> system -> You are a terse assistant. Answer in one sentence.
# -> human -> What is the capital of Japan?
# -> ai -> The capital of Japan is Tokyo.
# -> human -> And its population?
