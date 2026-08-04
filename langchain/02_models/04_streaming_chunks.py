"""
AIMESSAGECHUNK AND STREAMING AGGREGATION
==========================================
When you call `.stream()` on a chat model, LangChain doesn't yield full
AIMessage objects -- it yields `AIMessageChunk` objects, one per token (or
small group of tokens) as they arrive from the provider.

AIMessageChunk supports the `+` operator, which concatenates chunks into a
progressively larger chunk. This lets you either process tokens as they
arrive, or accumulate them into a single final message once streaming ends.
"""

from langchain_core.messages import AIMessageChunk
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b")

# --- Process each chunk as it arrives -----------------------------------
for chunk in model.stream("List three colors of the rainbow."):
    print(chunk)
    # -> AIMessageChunk(content='Red')
    # -> AIMessageChunk(content=',')
    # -> AIMessageChunk(content=' orange')
    # -> AIMessageChunk(content=',')
    # -> AIMessageChunk(content=' yellow')
    # -> AIMessageChunk(content='.')
    print(chunk.content, end="", flush=True)

# --- Aggregate chunks into one final AIMessageChunk ---------------------
full: AIMessageChunk | None = None
for chunk in model.stream("Name a fruit."):
    full = chunk if full is None else full + chunk  # `+` merges content/metadata

print(full)
# -> AIMessageChunk(content='A great fruit to name is the mango.')

# The merged chunk behaves like a normal AIMessage for downstream code --
# `.content` is the concatenated text, and metadata like `response_metadata`
# or `usage_metadata` gets combined where the provider supports it.
print(full.content)
# -> "A great fruit to name is the mango."

# --- Streaming with tool calls: chunks carry partial tool_call fragments -
# When the model streams a tool call, each chunk contains a slice of the
# arguments JSON. Summing the chunks reconstructs the full tool call once
# streaming finishes -- you generally don't parse partial tool_calls yourself.
accumulated = None
for chunk in model.stream("What's 5 + 7? Use the calculator tool if available."):
    accumulated = chunk if accumulated is None else accumulated + chunk

print(accumulated.tool_calls)
# -> [] (empty here since no tools were bound in this example)
