"""
RATE LIMITERS: INMEMORYRATELIMITER
=====================================
When calling a model API repeatedly (e.g. in a loop over many documents),
you can easily exceed a provider's requests-per-minute quota. LangChain's
`InMemoryRateLimiter` throttles outgoing requests to a configured rate by
making `.invoke()` block until a "token" becomes available.

It implements a token-bucket algorithm: tokens refill at a steady rate,
and each request consumes one token, waiting if none are available.
"""

import time

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.chat_models import ChatOllama

# Allow roughly 1 request every 2 seconds, with a small burst allowance.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,      # refill rate: 1 token every 2 seconds
    check_every_n_seconds=0.1,    # how often to poll for an available token
    max_bucket_size=2,            # allow short bursts of up to 2 requests
)

model = ChatOllama(model="qwen2.5:1.5b", rate_limiter=rate_limiter)

# --- Looping over many prompts: each .invoke() call self-throttles -------
prompts = [
    "Summarize the word 'ocean' in five words.",
    "Summarize the word 'mountain' in five words.",
    "Summarize the word 'desert' in five words.",
]

start = time.monotonic()
for prompt in prompts:
    response = model.invoke(prompt)
    print(response.content)
    # -> "Vast, deep, salty, teeming, blue"
    # -> "Tall, rocky, cold, majestic, remote"
    # -> "Dry, sandy, hot, sparse, silent"

elapsed = time.monotonic() - start
print(f"Total time: {elapsed:.1f}s")
# -> "Total time: 4.1s" (throttled to roughly one call every 2 seconds)

# --- Rate limiters are attached per-model, not global ---------------------
# If you have multiple ChatOpenAI instances hitting the same account quota,
# share one InMemoryRateLimiter instance across them so their combined
# request rate stays under the limit.
shared_limiter = InMemoryRateLimiter(requests_per_second=1)
model_a = ChatOllama(model="qwen2.5:1.5b", rate_limiter=shared_limiter)
model_b = ChatOllama(model="qwen2.5:1.5b", rate_limiter=shared_limiter)
# Both model_a and model_b now draw from the same token bucket.
