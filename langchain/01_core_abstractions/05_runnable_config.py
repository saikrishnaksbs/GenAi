"""
RunnableConfig
================
Every Runnable's `.invoke()`, `.stream()`, `.batch()` methods accept an
optional second argument: a `RunnableConfig` dict. It carries cross-cutting
concerns — callbacks, tags, metadata, run name, recursion limits — through
the whole chain without polluting your actual input data.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama


class PrintTokenHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


model = ChatOllama(model="qwen2.5:1.5b", streaming=True)
chain = ChatPromptTemplate.from_template("Say hi to {name}") | model | StrOutputParser()

config: RunnableConfig = {
    "tags": ["greeting-chain", "demo"],
    "metadata": {"user_id": "u_123"},
    "run_name": "greet_user",
    "callbacks": [PrintTokenHandler()],
    "max_concurrency": 5,   # caps parallelism for .batch()
}

result = chain.invoke({"name": "Sai"}, config=config)

# Tags/metadata propagate to every nested Runnable inside the chain, which is
# how tracing tools like LangSmith group and filter related runs.

# You can also bind config permanently to a chain so callers don't need to pass it:
configured_chain = chain.with_config(tags=["always-tagged"])
configured_chain.invoke({"name": "Team"})
