"""
DYNAMIC ROUTING / BRANCHING IN LCEL (Modern Replacement for MultiPromptChain/RouterChain)
========================================================================================
In legacy LangChain, routing requests dynamically to different sub-chains based on
user input required using complex `MultiPromptChain` or `RouterChain` classes.

In modern LCEL, routing can be achieved in two ways:
1. `RunnableBranch`: A declarative conditional branching utility.
2. Custom Routing Function: A standard Python function wrapped in `RunnableLambda`
   that evaluates inputs and returns the appropriate next chain. (Recommended for readability).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# Define sub-chains for different subjects
math_chain = (
    ChatPromptTemplate.from_template("You are a math expert. Answer this question:\n{question}")
    | llm
    | StrOutputParser()
)

physics_chain = (
    ChatPromptTemplate.from_template("You are a physics expert. Answer this question:\n{question}")
    | llm
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("Answer this general query:\n{question}")
    | llm
    | StrOutputParser()
)


# --- Approach 1: Custom Routing Function (Recommended) ---
# Why recommended: Standard Python functions are highly readable, easy to debug/step-through,
# and support standard Python control statements (if/else, logging, print, etc.).
print("=== Approach 1: Custom Routing Function ===")

# 1. Classifier Chain:
#    Takes the user's question and asks the LLM to output exactly one keyword category.
#    Output: A clean, parsed category string ('math', 'physics', or 'general').
classifier_prompt = ChatPromptTemplate.from_template(
    "Classify the following question into exactly one of these topics: 'math', 'physics', or 'general'. "
    "Do not include any other text.\n\n"
    "Question: {question}\n"
    "Topic:"
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# 2. Router Function:
#    A standard Python function wrapped in RunnableLambda.
#    - Input: A dictionary `info` (constructed in the LCEL pipeline).
#    - Output: Returns the next *Runnable* chain (NOT the execution result). LangChain
#      automatically executes whatever Runnable is returned from the router.
def router(info):
    category = info["category"].strip().lower()
    print(f"-> Categorized question as: '{category}'")
    if "math" in category:
        return math_chain
    elif "physics" in category:
        return physics_chain
    else:
        return general_chain

# 3. Pipeline Construction:
#    - Input dictionary: `{"question": "..."}`
#    - Step 1: Constructs a new dictionary:
#      - `"category"`: Runs classifier_chain (which processes x["question"]) and obtains the category.
#      - `"question"`: Passes the original query through using a lambda.
#    - Step 2: Pipes the constructed dictionary into `RunnableLambda(router)`.
#    - Step 3: LangChain evaluates the router function, gets the sub-chain, and executes it.
routing_chain = (
    {"category": classifier_chain, "question": lambda x: x["question"]}
    | RunnableLambda(router)
)

ans1 = routing_chain.invoke({"question": "What is the derivative of x^2 + 5x?"})
print("Answer:", ans1)

ans2 = routing_chain.invoke({"question": "Why is the sky blue?"})
print("\nAnswer:", ans2)


# --- Approach 2: RunnableBranch ---
# Why use it: A declarative, DSL-like syntax for conditional execution.
# How it works: Evaluates each condition sequentially. The first condition that returns True
# triggers the execution of its corresponding chain. If none match, it runs the fallback chain.
print("\n=== Approach 2: RunnableBranch ===")

# Define the branch logic:
# - Condition 1: If "math" is in category, execute math_chain.
# - Condition 2: If "physics" is in category, execute physics_chain.
# - Fallback: Otherwise, execute general_chain.
runnable_branch = RunnableBranch(
    (lambda x: "math" in x["category"].lower(), math_chain),
    (lambda x: "physics" in x["category"].lower(), physics_chain),
    general_chain,  # default fallback
)

# Pipe the constructed dictionary context into the runnable branch.
# Since it's a standard Runnable, it processes the input dictionary just like Approach 1.
branch_pipeline = (
    {"category": classifier_chain, "question": lambda x: x["question"]}
    | runnable_branch
)

ans3 = branch_pipeline.invoke({"question": "State Newton's second law of motion."})
print("Answer:", ans3)
