"""
TOKEN USAGE TRACKING
======================
When you're paying per token, it's useful to measure exactly how many
tokens a chain consumes. LangChain provides `get_openai_callback()`, a
context manager that aggregates prompt/completion tokens and estimated
cost across every OpenAI call made inside its `with` block.

For non-OpenAI models, token counts are usually read off
`response.usage_metadata` on the returned AIMessage instead.
"""

from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {concept} in two sentences.")
model = ChatOllama(model="qwen2.5:1.5b")
chain = prompt | model | StrOutputParser()

# Everything invoked inside this block has its token usage tallied.
with get_openai_callback() as cb:
    result = chain.invoke({"concept": "vector embeddings"})
    print(result)
    print(cb)
    # -> Tokens Used: 142
    # ->     Prompt Tokens: 24
    # ->     Completion Tokens: 118
    # -> Successful Requests: 1
    # -> Total Cost (USD): $0.000085

    print(f"total tokens: {cb.total_tokens}")
    print(f"prompt tokens: {cb.prompt_tokens}")
    print(f"completion tokens: {cb.completion_tokens}")
    print(f"total cost: ${cb.total_cost:.6f}")

# The callback accumulates across multiple calls within the same `with` block,
# which is handy for measuring the cost of a whole multi-step chain or loop.
with get_openai_callback() as cb:
    for concept in ["gradient descent", "attention mechanisms"]:
        chain.invoke({"concept": concept})
    print(f"total tokens across {2} calls: {cb.total_tokens}")

# For chat models that expose usage_metadata directly on the AIMessage
# (works across providers, not just OpenAI):
raw_response = model.invoke("What is a token in NLP?")
print(raw_response.usage_metadata)
# -> {'input_tokens': 9, 'output_tokens': 34, 'total_tokens': 43}
