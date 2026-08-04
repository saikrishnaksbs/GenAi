"""
LangServe (Deploying a Chain as a REST API)
==============================================
LangServe wraps any LCEL Runnable into a FastAPI app with auto-generated
routes: /invoke, /batch, /stream, /stream_events, plus an interactive
playground UI and an auto-generated OpenAPI schema/client. Effectively
"one line to turn a chain into a production API".
"""

# --- server.py ---
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

app = FastAPI(title="My LangChain Server", version="1.0")

chain = (
    ChatPromptTemplate.from_template("Translate to {language}: {text}")
    | ChatOllama(model="qwen2.5:1.5b")
    | StrOutputParser()
)

# Registers POST /translate/invoke, /translate/batch, /translate/stream, etc.
add_routes(app, chain, path="/translate")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run with: python server.py
# Playground UI available at: http://localhost:8000/translate/playground/


# --- client.py (calling the deployed chain remotely) ---
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/translate/")
result = remote_chain.invoke({"language": "French", "text": "Good morning"})
print(result)
# The remote chain still behaves like any local Runnable — .invoke/.batch/.stream all work.
