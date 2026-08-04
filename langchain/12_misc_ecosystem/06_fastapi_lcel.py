"""
WRAPPING LCEL CHAINS IN PLAIN FASTAPI (NON-LANGSERVE)
======================================================
While LangServe is excellent for automatically exposing LCEL runnables, 
wrapping chains directly in plain FastAPI is common when integrating into 
existing microservices, implementing custom JWT validation middleware, 
or enforcing proprietary rate limits.

This script implements a fully functional async FastAPI server that wraps an 
LCEL chain, demonstrating:
1. Async invocation with `ainvoke`.
2. Token streaming with `astream` routed via FastAPI's `StreamingResponse`.
3. Safe async concurrent handling under load.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Plain FastAPI LCEL Service",
    description="Production-grade async deployment of LCEL chains without LangServe."
)

# 2. Setup standard LCEL Chain
model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
prompt = ChatPromptTemplate.from_template("Translate the following text to Spanish:\n{text}")
chain = prompt | model | StrOutputParser()


# 3. Define Pydantic request models
class TranslationRequest(BaseModel):
    text: str


# 4. Implement Async Invoke Endpoint
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Processes translation request asynchronously."""
    try:
        # Utilize 'ainvoke' to ensure the event loop is not blocked,
        # allowing FastAPI to handle multiple concurrent requests.
        result = await chain.ainvoke({"text": request.text})
        return {"translated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. Implement Async Streaming Endpoint
@app.post("/translate/stream")
async def translate_text_stream(request: TranslationRequest):
    """Streams translation token-by-token using StreamingResponse."""
    
    async def token_generator():
        try:
            # We iterate over 'astream' which yields chunks asynchronously
            async for token in chain.astream({"text": request.text}):
                # Yield tokens immediately to the client
                yield token
                # Brief sleep to release event loop control
                await asyncio.sleep(0)
        except Exception as e:
            yield f"\n[Streaming Error]: {str(e)}"

    return StreamingResponse(token_generator(), media_type="text/plain")


# --- Setup a mock client runner to verify code syntax ---
if __name__ == "__main__":
    print("FastAPI application initialized successfully.")
    print("Routes registered: /translate (POST), /translate/stream (POST)")
    print("Run using: uvicorn 06_fastapi_lcel:app --reload")
