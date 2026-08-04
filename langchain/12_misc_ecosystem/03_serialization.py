"""
Serialization (dumps/loads, .dict(), saving/loading chains)
================================================================
LangChain objects (prompts, chains, some models) can be serialized to JSON
so they can be saved to disk, sent over a network, or version-controlled.
`dumps`/`loads` handle full round-tripping of LangChain objects (including
nested Runnables), while `.dict()` gives a plain-Python-dict snapshot useful
for logging/inspection.
"""

from langchain_core.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Answer this question: {question}")

# dumps() serializes a LangChain object (including type info) to a JSON string.
serialized = dumps(prompt, pretty=True)
print(serialized)

# loads() reconstructs the original LangChain object from that JSON string.
restored_prompt = loads(serialized)
print(restored_prompt.format(question="What is LCEL?"))

# .dict() gives a plain dict representation, handy for logging or diffing,
# but it is NOT guaranteed to be loadable back into an object the way dumps() is.
print(prompt.dict())

# Saving to / loading from a file:
with open("prompt.json", "w") as f:
    f.write(dumps(prompt))

with open("prompt.json") as f:
    loaded_prompt = loads(f.read())

# NOTE: objects containing API keys or arbitrary Python callables (like
# RunnableLambda wrapping a local function) generally cannot be fully
# serialized — dumps() will raise or fall back to a non-reconstructible
# representation for those pieces.
