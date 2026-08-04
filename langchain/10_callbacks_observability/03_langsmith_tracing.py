"""
LANGSMITH TRACING
===================
LangSmith is Anthropic/LangChain's observability platform for inspecting
chain runs, latency, token usage, and errors. Tracing is usually enabled
purely through environment variables -- no code changes to your chain are
required.

Required environment variables (typically set in a .env file):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=my-project-name        # optional, defaults to "default"
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # optional override

Once those are set, every Runnable invocation is automatically traced and
visible in the LangSmith UI as a tree of nested spans.
"""

import os
from langsmith import traceable
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Enable tracing for this process. In real projects this lives in .env
# and is loaded via `python-dotenv`, not set inline like this.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-examples"
# os.environ["LANGCHAIN_API_KEY"] = "ls__..."  # set via your shell / secret manager

prompt = ChatPromptTemplate.from_template("Summarize this in one sentence: {text}")
model = ChatOllama(model="qwen2.5:1.5b")
chain = prompt | model | StrOutputParser()

# No extra code needed here -- because LANGCHAIN_TRACING_V2 is set, this
# invocation is automatically captured as a run in the LangSmith project.
summary = chain.invoke({"text": "LangChain is a framework for building LLM apps..."})
print(summary)


# You can also wrap arbitrary Python functions (not just Runnables) with
# @traceable so custom business logic shows up as a span in the same trace.
@traceable(name="postprocess_summary")
def postprocess(summary: str) -> str:
    return summary.strip().rstrip(".") + "."


final = postprocess(summary)
print(final)

# Runs can be grouped and tagged for easier filtering in the LangSmith UI:
tagged_result = chain.invoke(
    {"text": "Callbacks let you observe chain execution."},
    config={"tags": ["demo", "tracing-example"], "metadata": {"user_id": "u_123"}},
)
# -> visible in LangSmith as a run tagged ["demo", "tracing-example"]
# -> with metadata {"user_id": "u_123"} attached for filtering/search
