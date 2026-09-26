"""
SEQUENTIAL LCEL CHAINS (Modern Replacement for SequentialChains)
==================================================================
Instead of the legacy SimpleSequentialChain and SequentialChain, modern LangChain
composes sequences using the pipe (`|`) operator and utility runnables like
RunnablePassthrough and RunnableParallel.

- Simple Sequential: Pipe the output of one runnable into the input of another.
- Complex Sequential (with memory/branching): Use RunnablePassthrough.assign()
  to run intermediate chains and add their results to the running context dictionary
  so subsequent steps can reference them.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.5)

# --- 1. Simple Sequential Chain (Linear Piping) ---
# Goal: business_type -> business_name -> business_tagline

name_prompt = ChatPromptTemplate.from_template(
    "Give a creative name for a {business}. Respond with just the name."
)
tagline_prompt = ChatPromptTemplate.from_template(
    "Write a short tagline for a business called {business}."
)

name_chain = name_prompt | llm | StrOutputParser()
tagline_chain = tagline_prompt | llm | StrOutputParser()

# The output of name_chain (a string) is mapped to the "business" key required by tagline_chain
simple_pipeline = (
    {"business": name_chain}
    | tagline_chain
)

print("--- Simple Sequential Chain ---")
final_tagline = simple_pipeline.invoke({"business": "artisanal coffee shop"})
print(final_tagline)


# --- 2. Complex Sequential Chain (Multiple Inputs/Outputs) ---
# Goal: (product_name, features) -> review -> short_summary
# We want the final output to contain the inputs, the review, and the summary.

review_prompt = ChatPromptTemplate.from_template(
    "Write a 2-sentence product review for {product_name}, highlighting: {features}."
)
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this review in 5 words or fewer:\n{review}"
)

review_chain = review_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()

# We use RunnablePassthrough.assign to carry forward intermediate inputs and outputs
overall_chain = (
    RunnablePassthrough.assign(review=review_chain)
    | RunnablePassthrough.assign(short_summary=summary_chain)
)

print("\n--- Complex Sequential Chain ---")
result = overall_chain.invoke(
    {
        "product_name": "AeroFlex Running Shoes",
        "features": "lightweight, breathable mesh, extra arch support",
    }
)
print("Keys in result:", result.keys())
print("Review:", result["review"])
print("Short Summary:", result["short_summary"])
