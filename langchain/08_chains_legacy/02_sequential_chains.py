"""
SEQUENTIAL CHAINS
===================
SimpleSequentialChain and SequentialChain both run a list of chains one
after another, feeding each chain's output into the next. The difference:

- SimpleSequentialChain: each chain has exactly ONE input and ONE output,
  and that single output is piped straight into the next chain's single input.
  Simple but inflexible.

- SequentialChain: chains can have MULTIPLE named inputs/outputs. You
  explicitly declare `input_variables` and `output_variables` for the whole
  pipeline, and intermediate outputs are tracked by name so later chains can
  reference any prior chain's output, not just the immediately preceding one.

Both are legacy; the LCEL equivalent is just composing runnables with `|`
and RunnableParallel/RunnablePassthrough for branching.
"""

from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.5)

# --- SimpleSequentialChain: single in, single out, chained linearly ---
name_prompt = PromptTemplate(
    input_variables=["business"],
    template="Give a creative name for a {business}. Respond with just the name.",
)
name_chain = LLMChain(llm=llm, prompt=name_prompt)

tagline_prompt = PromptTemplate(
    input_variables=["business"],  # simple chain always calls this var whatever it was defined as
    template="Write a short tagline for a business called {business}.",
)
tagline_chain = LLMChain(llm=llm, prompt=tagline_prompt)

simple_pipeline = SimpleSequentialChain(chains=[name_chain, tagline_chain], verbose=True)
final_tagline = simple_pipeline.run("artisanal coffee shop")
print(final_tagline)
# -> "Where Every Sip Tells a Story."   (tagline for whatever name the first chain invented)


# --- SequentialChain: multiple named inputs/outputs, more control ---
review_prompt = PromptTemplate(
    input_variables=["product_name", "features"],
    template="Write a 2-sentence product review for {product_name}, highlighting: {features}.",
)
review_chain = LLMChain(llm=llm, prompt=review_prompt, output_key="review")

summary_prompt = PromptTemplate(
    input_variables=["review"],
    template="Summarize this review in 5 words or fewer:\n{review}",
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="short_summary")

overall_chain = SequentialChain(
    chains=[review_chain, summary_chain],
    input_variables=["product_name", "features"],  # required initial inputs
    output_variables=["review", "short_summary"],   # which intermediate outputs to surface
    verbose=True,
)

result = overall_chain(
    {
        "product_name": "AeroFlex Running Shoes",
        "features": "lightweight, breathable mesh, extra arch support",
    }
)
print(result["review"])
# -> "AeroFlex Running Shoes are incredibly lightweight and the breathable mesh
#     keeps feet cool. The extra arch support makes long runs noticeably easier."
print(result["short_summary"])
# -> "Lightweight, breathable, supportive running shoes."
