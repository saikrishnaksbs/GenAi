"""
STREAMING CALLBACKS
=====================
Streaming callbacks let you react to model output token-by-token as it is
generated, rather than waiting for the full response. This is how chat UIs
show text appearing incrementally.

Two common approaches:
    1. `.stream()` on a Runnable -> iterate over chunks directly (preferred).
    2. `on_llm_new_token` callback -> fired for each token when
       `streaming=True` is set on the model, useful when you need side
       effects (e.g. pushing tokens over a websocket) rather than a
       generator you control yourself.
"""

from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class StreamingPrintHandler(BaseCallbackHandler):
    """Prints each token as it arrives, e.g. to simulate pushing to a socket."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        print("\n[stream complete]")


prompt = ChatPromptTemplate.from_template("Write a two-line haiku about {subject}.")
model = ChatOllama(
    model="qwen2.5:1.5b",
    streaming=True,  # required for on_llm_new_token to fire
    callbacks=[StreamingPrintHandler()],
)
chain = prompt | model | StrOutputParser()

# Approach 1: idiomatic LCEL streaming, iterate over the generator yourself.
for chunk in chain.stream({"subject": "the ocean at dawn"}):
    print(chunk, end="", flush=True)
# -> Waves whisper softly
# -> Gold light spills across the tide

# Approach 2: use .invoke() but let the callback handler do the printing.
# Both the token-by-token callback output AND the final return value happen;
# .invoke() still blocks until generation is complete.
final_text = chain.invoke({"subject": "a quiet library"})
print("\nfinal:", final_text)
