"""
STRING AND CRITERIA EVALUATORS
=================================
`langchain.evaluation` provides evaluators for grading model output. The
two simplest kinds are:

    - "string" evaluators: compare a prediction against a reference answer
      using exact/fuzzy matching or an LLM judge.
    - "criteria" evaluators: ask an LLM judge whether the output satisfies
      a named criterion (conciseness, correctness, harmfulness, etc.),
      with no reference answer required.

Under the hood, criteria evaluators are themselves LLM chains -- the judge
model is prompted to reason about the criterion and output a verdict.
"""

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.chat_models import ChatOllama

judge_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)  # low temp for consistent grading

# --- Exact-match style string evaluator (no LLM call needed) ---
exact_match_evaluator = load_evaluator("exact_match")
result = exact_match_evaluator.evaluate_strings(
    prediction="Paris",
    reference="Paris",
)
print(result)
# -> {'score': 1}

# --- QA evaluator: LLM judges correctness against a reference answer ---
qa_evaluator = load_evaluator("qa", llm=judge_model)
result = qa_evaluator.evaluate_strings(
    input="What is the capital of France?",
    prediction="The capital of France is Paris.",
    reference="Paris",
)
print(result)
# -> {'reasoning': 'The prediction matches the reference answer.', 'value': 'CORRECT', 'score': 1}

# --- Criteria evaluator: judge against a named quality, no reference needed ---
conciseness_evaluator = load_evaluator(
    EvaluatorType.CRITERIA, criteria="conciseness", llm=judge_model
)
result = conciseness_evaluator.evaluate_strings(
    input="What's 2+2?",
    prediction="Well, mathematically speaking, when you add the integer 2 to another integer 2, the sum you arrive at is 4.",
)
print(result)
# -> {'reasoning': '...too verbose for a simple question...', 'value': 'N', 'score': 0}

# --- Custom criteria: define your own rubric as a dict of {name: description} ---
custom_evaluator = load_evaluator(
    EvaluatorType.CRITERIA,
    criteria={"technical_accuracy": "Does the response use correct technical terminology?"},
    llm=judge_model,
)
result = custom_evaluator.evaluate_strings(
    input="Define a hash map.",
    prediction="A hash map stores key-value pairs and offers average O(1) lookup via hashing.",
)
print(result)
# -> {'reasoning': '...uses correct terminology...', 'value': 'Y', 'score': 1}
