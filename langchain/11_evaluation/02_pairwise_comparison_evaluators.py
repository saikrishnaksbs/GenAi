"""
PAIRWISE COMPARISON EVALUATORS
=================================
Pairwise (comparison) evaluators ask an LLM judge to pick the better of two
candidate outputs for the same input, e.g. when A/B testing two prompts,
two models, or two versions of a chain. This avoids needing an absolute
scoring rubric -- relative judgments are often more reliable for LLM judges.

Note: to avoid position bias (judges favoring whichever answer comes first),
it's good practice to run the comparison twice with the order swapped and
check for consistency.
"""

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.chat_models import ChatOllama

judge_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

pairwise_evaluator = load_evaluator(EvaluatorType.PAIRWISE_STRING, llm=judge_model)

question = "Explain what a race condition is."

response_a = (
    "A race condition happens when two threads access shared data at the "
    "same time and the outcome depends on the timing of their execution."
)
response_b = "It's a bug that happens with threads."

result = pairwise_evaluator.evaluate_string_pairs(
    input=question,
    prediction=response_a,
    prediction_b=response_b,
)
print(result)
# -> {'reasoning': 'Response A gives a precise, technically accurate definition...',
# ->  'value': 'A', 'score': 1}   # score 1 means A preferred, 0 means B preferred

# --- Swap order to check for position bias ---
swapped_result = pairwise_evaluator.evaluate_string_pairs(
    input=question,
    prediction=response_b,
    prediction_b=response_a,
)
print(swapped_result)
# -> {'reasoning': '...', 'value': 'B', 'score': 0}  # B here == response_a, so still consistent

# --- Pairwise comparison with custom criteria instead of the default rubric ---
custom_pairwise_evaluator = load_evaluator(
    EvaluatorType.PAIRWISE_STRING,
    criteria={"depth": "Which response demonstrates deeper technical understanding?"},
    llm=judge_model,
)
result = custom_pairwise_evaluator.evaluate_string_pairs(
    input=question,
    prediction=response_a,
    prediction_b=response_b,
)
print(result)
# -> {'reasoning': 'Response A discusses shared state and timing dependency...', 'value': 'A', 'score': 1}
