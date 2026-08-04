"""
MODEL ROUTING BY COST AND COMPLEXITY
====================================
This script demonstrates routing user queries to different LLM models dynamically 
to balance API cost and reasoning quality. 

Simple questions (e.g. greeting, factual lookups) are routed to a cheaper, faster 
model (gpt-4o-mini). Complex questions (e.g. writing code, math equations, logic puzzles) 
are routed to a more capable, expensive model (gpt-4o).

We implement this in two ways:
1. Dynamic routing based on prompt length / heuristic metrics.
2. Dynamic routing based on an LLM classifier step.
"""

from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Initialize the two models with different pricing/capability profiles
cheap_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)
expensive_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

prompt = ChatPromptTemplate.from_template("Answer the following query clearly: {query}")

# --------------------------------------------------------------------------
# Method 1: Heuristic Routing (Simple & Zero-cost classification)
# --------------------------------------------------------------------------
print("--- Method 1: Heuristic Routing ---")

def heuristic_classifier(inputs: dict) -> bool:
    """Returns True if the query is deemed 'complex' based on heuristic checks."""
    query = inputs["query"].lower()
    
    # 1. Length heuristic: questions over 200 characters usually require more capability
    if len(query) > 200:
        return True
    
    # 2. Keyword triggers: code, algorithm, write a function, math, solve
    complex_keywords = ["write a function", "algorithm", "python code", "optimize", "solve for x", "mathematical proof"]
    if any(kw in query for kw in complex_keywords):
        return True
        
    return False

# Build router chain using RunnableBranch
heuristic_router = RunnableBranch(
    (lambda x: heuristic_classifier(x), prompt | expensive_model | StrOutputParser()),
    prompt | cheap_model | StrOutputParser()
)

queries = [
    {"query": "Hi, what time is it?"}, # Simple -> Cheap Model
    {"query": "Write a python function to compute the edit distance between two strings using dynamic programming."} # Complex -> Expensive Model
]

for q in queries:
    # We add metadata to see which model gets run in logs/traces
    is_complex = heuristic_classifier(q)
    print(f"\nQuery: '{q['query']}'")
    print(f"Is Complex (Heuristic)? {is_complex} -> Routing to {'gpt-4o' if is_complex else 'gpt-4o-mini'}")
    # In a real environment, we'd invoke the router:
    # response = heuristic_router.invoke(q)


# --------------------------------------------------------------------------
# Method 2: LLM-Based Complexity Classifier
# --------------------------------------------------------------------------
print("\n--- Method 2: LLM Classifier Routing ---")

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user's query complexity. Respond with exactly 'SIMPLE' or 'COMPLEX'.\n"
               "Use 'COMPLEX' for programming, math, logic reasoning, and long detailed writing requests.\n"
               "Use 'SIMPLE' for greeting, trivia, simple fact checks, or short instructions."),
    ("human", "{query}")
])

# Classifier chain outputting a clean string
classifier_chain = classification_prompt | cheap_model | StrOutputParser()

# Route dynamically based on the classifier's output
llm_router = RunnableLambda(
    lambda inputs: (prompt | expensive_model) if classifier_chain.invoke(inputs).strip().upper() == "COMPLEX"
                  else (prompt | cheap_model)
)

test_query = {"query": "Explain quantum computing in one sentence."}
classification = classifier_chain.invoke(test_query).strip()
print(f"Query: '{test_query['query']}'")
print(f"LLM Classification: {classification} -> Routing accordingly")
