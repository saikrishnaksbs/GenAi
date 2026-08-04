"""
BASE CALLBACK HANDLER
======================
LangChain fires callback events at every stage of execution: when an LLM
call starts/ends, when a chain starts/ends, when a tool is invoked, when an
error occurs, etc. `BaseCallbackHandler` is the class you subclass to hook
into these events for logging, debugging, or custom observability.

Callbacks can be attached in two ways:
    - per-call:    chain.invoke(input, config={"callbacks": [handler]})
    - constructor: ChatOllama(model="qwen2.5:1.5b", callbacks=[handler])

This example builds a simple handler that logs LLM calls and chain steps.
"""

from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class LoggingCallbackHandler(BaseCallbackHandler):
    """Custom handler that prints a line for each lifecycle event."""

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        print(f"[llm_start] {len(prompts)} prompt(s) sent to {serialized.get('name')}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # response.llm_output often contains token_usage / model_name metadata
        usage = (response.llm_output or {}).get("token_usage", {})
        print(f"[llm_end] tokens used: {usage}")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        print(f"[chain_start] inputs={inputs}")

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        print(f"[chain_end] outputs={outputs}")

    def on_chat_model_start(self, serialized: dict, messages: list[list[BaseMessage]], **kwargs: Any) -> None:
        print(f"[chat_model_start] {len(messages[0])} message(s)")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"[llm_error] {error!r}")


handler = LoggingCallbackHandler()

prompt = ChatPromptTemplate.from_template("Give a one-sentence fun fact about {topic}.")
model = ChatOllama(model="qwen2.5:1.5b", callbacks=[handler])  # constructor-level callback
chain = prompt | model

# Passing callbacks=[handler] again here would double-log; constructor-level
# callbacks apply to every invocation of this model automatically.
result = chain.invoke({"topic": "octopuses"})
print(result.content)
# -> [chat_model_start] 1 message(s)
# -> [llm_end] tokens used: {'prompt_tokens': 18, 'completion_tokens': 22, 'total_tokens': 40}
# -> "Octopuses have three hearts and blue blood."
