"""
MODEL PARAMS: TEMPERATURE, MAX_TOKENS, STOP SEQUENCES
========================================================
Chat model constructors accept generation parameters that control how
text is sampled and when it stops. These are the most commonly tuned:

    temperature      -> randomness of sampling (0 = deterministic, 1+ = creative)
    max_tokens       -> hard cap on the length of the generated response
    stop             -> substrings that, once generated, immediately end output

Parameters can be set at construction time (applies to every call) or
overridden per-call by rebinding the model with `.bind()`.
"""

from langchain_community.chat_models import ChatOllama

# --- Set params at construction time -------------------------------------
deterministic_model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,       # always pick the highest-probability token
    max_tokens=50,        # truncate response after ~50 tokens
)

response = deterministic_model.invoke("Explain recursion in one paragraph.")
print(response.content)
# -> "Recursion is a technique where a function calls itself..." (cut off ~50 tokens)

# --- Higher temperature for more varied/creative output ------------------
creative_model = ChatOllama(model="qwen2.5:1.5b", temperature=1.2)

response = creative_model.invoke("Write a quirky product name for a smart mug.")
print(response.content)
# -> "MugMind: The Thermally Telepathic Cup"

# --- Stop sequences: end generation as soon as a marker appears ----------
stop_model = ChatOllama(model="qwen2.5:1.5b", stop=["\n\n", "END"])

response = stop_model.invoke("List two programming languages, then write END.")
print(response.content)
# -> "1. Python\n2. JavaScript" (generation halts right before "END")

# --- Overriding params per-call with .bind() ------------------------------
# Useful when most calls share a config but one call needs a tweak, without
# constructing a whole new model instance.
base_model = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)
short_and_strict = base_model.bind(max_tokens=20, temperature=0, stop=["."])

response = short_and_strict.invoke("Describe the ocean.")
print(response.content)
# -> "The ocean is a vast body of saltwater covering most of Earth's surface"

# --- Reproducible outputs with seed parameter -----------------------------
# Passing integer 'seed' forces the LLM's sampler to be deterministic.
# Useful for unit tests, debugging, and auditability.
seed_model = ChatOllama(model="qwen2.5:1.5b", temperature=0.7, seed=42)
res1 = seed_model.invoke("Generate a random 3-word slogan for space exploration.")
res2 = seed_model.invoke("Generate a random 3-word slogan for space exploration.")
print(f"Seed Run 1: {res1.content}")
print(f"Seed Run 2: {res2.content}")
# Both runs yield identical text because seed=42 locks pseudo-random sampling.

# Note: not every provider supports every parameter identically -- e.g.
# some expose `stop_sequences` instead of `stop`. LangChain normalizes the
# common ones, but check a provider's integration docs for edge cases.
