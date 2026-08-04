"""
with_retry() and with_fallbacks()
====================================
- .with_retry(): automatically retries a Runnable on failure (network errors,
  rate limits) with configurable retry count and exponential backoff.
- .with_fallbacks(): if the primary Runnable raises an exception, tries the
  next Runnable in the fallback list instead of failing the whole chain.
Both are essential for production reliability.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOllama

primary_model = ChatOllama(model="qwen2.5:1.5b")

# Retry up to 3 times, with exponential backoff, only on specific exception types.
resilient_model = primary_model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

prompt = ChatPromptTemplate.from_template("Summarize: {text}")
chain = prompt | resilient_model | StrOutputParser()
print(chain.invoke({"text": "LangChain and reliability patterns."}))

# --- Fallbacks: fall back to a cheaper/different provider if the primary fails ---
backup_model = ChatOllama(model="qwen2.5:1.5b")

model_with_fallback = primary_model.with_fallbacks([backup_model])

fallback_chain = prompt | model_with_fallback | StrOutputParser()
result = fallback_chain.invoke({"text": "If Anthropic is down, OpenAI answers instead."})
print(result)

# Retry and fallback can be combined — the whole chain (not just the model) can
# also have fallbacks, e.g. falling back to a simpler prompt if a complex one fails.
chain_with_fallback = chain.with_fallbacks([prompt | backup_model | StrOutputParser()])
