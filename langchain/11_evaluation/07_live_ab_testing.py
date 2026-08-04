"""
PRODUCTION EXPERIMENTATION: A/B TESTING PROMPTS ON LIVE TRAFFIC
=============================================================
A/B testing prompts offline on static evaluation sets is helpful, but final 
validation must occur on real-world traffic to observe user behavior, real inputs,
costs, and conversion metrics.

This script demonstrates a routing pattern for live prompt A/B testing:
1. Split traffic dynamically (e.g. 50/50 split based on random choice or user ID hash).
2. Execute Prompt A (Control) or Prompt B (Treatment) accordingly.
3. Attach metadata tags (`group_A` vs `group_B`) so LangSmith/log aggregators 
   can isolate and compare analytics (cost, response length, speed, and feedback).
"""

import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Initialize LLM
model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --------------------------------------------------------------------------
# Define Prompts for A/B Testing
# --------------------------------------------------------------------------
# Prompt A (Control): Standard customer service tone
prompt_a = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service assistant. Answer queries politely and concisely."),
    ("human", "{query}")
])

# Prompt B (Treatment): Highly friendly, structured, and includes a signature emoji
prompt_b = ChatPromptTemplate.from_messages([
    ("system", "You are an extremely helpful, cheerful customer helper. "
               "Structure your answer with bullet points if helpful, and always end "
               "with a positive emoji! 😊"),
    ("human", "{query}")
])

# Build LCEL chains
chain_a = prompt_a | model | StrOutputParser()
chain_b = prompt_b | model | StrOutputParser()


# --------------------------------------------------------------------------
# Traffic Router & Metadata Tagging
# --------------------------------------------------------------------------
def route_ab_test(user_id: str, query: str) -> str:
    """Routes user to group A or B based on user_id hash and executes the chain."""
    
    # 1. Deterministic hashing split (keeps a user pinned to the same version)
    # We take the last character of the user_id or compute a simple hash
    user_hash = hash(user_id)
    group = "B" if user_hash % 2 == 0 else "A"
    
    print(f"\nUser: {user_id} -> Assigned to Group: {group}")
    
    # 2. Configure metadata tracking (propagates directly to LangSmith)
    config = {
        "metadata": {
            "experiment_id": "exp_return_prompt_v2",
            "ab_group": group,
            "user_id": user_id
        },
        "tags": [f"group_{group}", "experiment_v2"]
    }
    
    # 3. Route execution
    if group == "A":
        response = chain_a.invoke({"query": query}, config=config)
    else:
        response = chain_b.invoke({"query": query}, config=config)
        
    return response


# --------------------------------------------------------------------------
# Simulated Production Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Test traffic routing across multiple simulated users
    test_traffic = [
        {"user_id": "usr_948", "query": "Can I return a shirt without tags?"},
        {"user_id": "usr_821", "query": "How do I check my order shipping status?"},
        {"user_id": "usr_104", "query": "Do you ship to Canada?"}
    ]
    
    for client in test_traffic:
        reply = route_ab_test(client["user_id"], client["query"])
        print(f"Reply:\n{reply}")
