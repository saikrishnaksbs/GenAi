"""
CACHING: SET_LLM_CACHE, IN-MEMORY / SQLITE CACHE
====================================================
Calling a model with the exact same prompt repeatedly wastes money and
latency. LangChain lets you install a global cache so identical
(model, prompt, params) calls are served from cache instead of hitting the
provider again.

Two common cache backends:

    InMemoryCache -> lives only for the current process; fast, non-persistent
    SQLiteCache   -> persists to a local .db file across process restarts

Set the cache once globally with `set_llm_cache()`; every model call in
the process then automatically checks/populates it.
"""

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --- In-memory cache: fastest, cleared when the process exits ------------
set_llm_cache(InMemoryCache())

response1 = model.invoke("What is the boiling point of water in Celsius?")
print(response1.content)
# -> "100 degrees Celsius." (first call: hits the API, ~1s)

response2 = model.invoke("What is the boiling point of water in Celsius?")
print(response2.content)
# -> "100 degrees Celsius." (second call: served from cache, ~0ms)

# --- SQLite cache: persists across runs -----------------------------------
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

response = model.invoke("Name the largest planet in our solar system.")
print(response.content)
# -> "Jupiter is the largest planet in our solar system."

# Running this script again later will reuse the same .langchain_cache.db
# file, so identical prompts skip the API call entirely -- useful for
# repeated test runs or notebooks you re-execute often.

# --- Caching is keyed on the exact prompt + model + params ---------------
# Changing temperature, max_tokens, or the prompt text even slightly
# produces a cache miss and triggers a fresh API call.
uncached_model = ChatOllama(model="qwen2.5:1.5b", temperature=0.5)
response = uncached_model.invoke("Name the largest planet in our solar system.")
print(response.content)
# -> "Jupiter is the largest planet in our solar system." (different temperature -> cache miss)

# Note: caching is best for deterministic, repeatable workloads (tests,
# demos, idempotent batch jobs) -- avoid it for anything that should
# genuinely vary between calls, like temperature > 0 creative generation.
