"""
EXAMPLE SELECTORS
====================
When you have too many few-shot examples to fit in a prompt, an
ExampleSelector picks a relevant subset at runtime instead of hardcoding
which examples to use. Common strategies:

    LengthBasedExampleSelector          -> pick as many examples as fit
                                            within a token/length budget
    SemanticSimilarityExampleSelector   -> pick examples whose embeddings
                                            are most similar to the input
    MaxMarginalRelevanceExampleSelector -> like semantic similarity, but
                                            also favors diversity among
                                            the selected examples (MMR)

Selectors plug directly into FewShotPromptTemplate via `example_selector`
instead of a static `examples` list.
"""

from langchain_core.example_selectors import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

# --- LengthBasedExampleSelector: fit as many examples as the budget allows
length_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,  # rough word-count budget for selected examples
)

length_based_prompt = FewShotPromptTemplate(
    example_selector=length_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"],
)

# A long input leaves less room, so fewer examples get included.
print(length_based_prompt.invoke({"word": "big"}).to_string())
# -> prompt including as many examples as fit under the length budget

# --- SemanticSimilarityExampleSelector: pick the most relevant examples --
similarity_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OllamaEmbeddings(model="qwen3-embedding:8b"),
    FAISS,          # vector store used to index example embeddings
    k=2,            # return the top 2 most similar examples
)

print(similarity_selector.select_examples({"input": "cheerful"}))
# -> [{'input': 'happy', 'output': 'sad'}, {'input': 'sunny', 'output': 'gloomy'}]

# --- MaxMarginalRelevanceExampleSelector: relevant AND diverse -----------
mmr_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OllamaEmbeddings(model="qwen3-embedding:8b"),
    FAISS,
    k=2,
)

print(mmr_selector.select_examples({"input": "cheerful"}))
# -> [{'input': 'happy', 'output': 'sad'}, {'input': 'windy', 'output': 'calm'}]
# (MMR trades a bit of pure similarity for broader coverage of the example set)
