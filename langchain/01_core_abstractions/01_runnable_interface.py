"""
RUNNABLE INTERFACE
===================
Every LangChain component (models, prompts, parsers, retrievers) implements
the `Runnable` protocol. That means they all share the same methods:

    .invoke(input)          -> single call
    .ainvoke(input)         -> async single call
    .batch([inputs])        -> multiple calls in parallel
    .abatch([inputs])       -> async batch
    .stream(input)          -> generator of chunks
    .astream(input)         -> async generator of chunks

Because everything is a Runnable, you can compose them uniformly with LCEL.
"""

from langchain_core.runnables import Runnable, RunnableLambda


# You can build your own Runnable from a plain function.
def shout(text: str) -> str:
    return text.upper() + "!!!"


shout_runnable: Runnable = RunnableLambda(shout)

print(shout_runnable.invoke("hello world"))
# -> "HELLO WORLD!!!"

print(shout_runnable.batch(["hi", "bye"]))
# -> ["HI!!!", "BYE!!!"]

for chunk in shout_runnable.stream("streaming demo"):
    # RunnableLambda over a non-generator function just yields the full result once
    print(chunk)


# --- Production-Grade Custom Runnable ------------------------------------
# For a production-ready custom Runnable, we want:
# 1. Input/Output validation (e.g. using Pydantic or types via .with_types()).
# 2. Native asynchronous support (avoiding thread-pool overhead).
# 3. Native chunk-by-chunk streaming (via a generator/iterator).
# 4. Telemetry/Callback support (accepting RunnableConfig).

import asyncio
from typing import AsyncIterator, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

# Input & Output Schemas for validation and auto-API documentation (LangServe)
class ShoutInput(BaseModel):
    text: str = Field(description="The text to shout")

class ShoutOutput(BaseModel):
    shouted_text: str = Field(description="The shouted text with exclamation marks")

# 1. Native stream/async generator function
async def shout_async_stream(
    input_data: Dict[str, Any], 
    config: RunnableConfig
) -> AsyncIterator[str]:
    """
    Production-grade async generator function.
    - Yields chunks incrementally for real-time streaming.
    - Accepts RunnableConfig to propagate tracing/callbacks (e.g. LangSmith).
    """
    text = input_data.get("text", "")
    # Simulate processing or streaming word by word / chunk by chunk
    words = text.split()
    for idx, word in enumerate(words):
        chunk = word.upper()
        if idx == len(words) - 1:
            chunk += "!!!"
        else:
            chunk += " "
        yield chunk

# 2. Wrap the async generator function into a RunnableLambda, adding type metadata
production_shout_runnable = RunnableLambda(
    shout_async_stream
).with_types(
    input_type=ShoutInput,
    output_type=ShoutOutput
)

async def run_demo():
    print("\n--- Production-Grade Runnable Demo ---")
    
    # Validation & invoke
    # Runnable automatically parses Pydantic inputs
    res = await production_shout_runnable.ainvoke({"text": "production grade hello world"})
    print("Async Invoke Result:", res)
    
    # Native incremental streaming
    print("Streaming Chunks:")
    async for chunk in production_shout_runnable.astream({"text": "streaming incremental data"}):
        print(f"Chunk: {repr(chunk)}")

# Run the async demo
asyncio.run(run_demo())

