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
    input_data: Any, 
    config: RunnableConfig
) -> AsyncIterator[str]:
    """
    Production-grade async generator function.
    - Yields chunks incrementally for real-time streaming.
    - Accepts RunnableConfig to propagate tracing/callbacks (e.g. LangSmith).
    - Safely handles dict inputs (e.g. {"text": "..."}) and raw stream chunk inputs (str).
    """
    if isinstance(input_data, dict):
        text = input_data.get("text", "")
    else:
        text = str(input_data)

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

    # --- .transform() & .atransform() Stream Transformation Example ---
    # .transform(stream) maps an input generator stream to an output stream.
    print("\n--- Stream Transformation (.transform / .atransform) ---")
    
    def sync_token_stream():
        yield "hello "
        yield "world "
        yield "lcel transform"

    # Synchronous .transform(): 'shout_runnable' is a plain function (not a generator),
    # so LangChain buffers the input stream and yields the single transformed output once.
    print("Sync Transformed Stream (Buffered non-generator function):")
    for chunk in shout_runnable.transform(sync_token_stream()):
        print(f"Result: {chunk}")

    async def async_token_stream():
        yield "async "
        yield "stream "
        yield "transformation"

    # Asynchronous .atransform(): 'production_shout_runnable' is a TRUE async generator (yields chunks),
    # so it transforms and streams each chunk incrementally in real time as it arrives!
    print("\nAsync Transformed Stream (Real-Time Generator Streaming):")
    print("Live Stream Output: ", end="", flush=True)
    async for chunk in production_shout_runnable.atransform(async_token_stream()):
        print(chunk, end="", flush=True)
    print("\n")

# Run the async demo
asyncio.run(run_demo())


