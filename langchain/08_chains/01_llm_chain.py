"""
LCEL CHAIN (Modern Replacement for LLMChain)
=============================================
Instead of the legacy LLMChain class, modern LangChain uses LangChain Expression Language (LCEL).
You compose a pipeline using the pipe operator (`|`):

    chain = prompt | llm | output_parser

This returns a RunnableSequence, which natively supports streaming, batching, and async operations.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Initialize the modern ChatOllama model from langchain_ollama
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["product"],
    template="Suggest a catchy, one-line slogan for a company that makes {product}.",
)

# Modern LCEL Composition: prompt -> llm -> string output parser
chain = prompt | llm | StrOutputParser()

# 1. Single execution using .invoke() (replacing legacy .run())
slogan = chain.invoke({"product": "eco-friendly water bottles"})
print("--- Invoke Single ---")
print(slogan)

# 2. Batch execution using .batch()
print("\n--- Batch Run ---")
batch_results = chain.batch(
    [{"product": "electric bikes"}, {"product": "instant coffee"}]
)
for r in batch_results:
    print(r)

# 3. Streaming response using .stream() (natively supported by LCEL)
print("\n--- Streaming Run ---")
for chunk in chain.stream({"product": "standing desks"}):
    print(chunk, end="", flush=True)
print()
