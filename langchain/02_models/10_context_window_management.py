"""
CONTEXT WINDOW OVERFLOW HANDLING
================================
In production LLM applications, exceeding the model's maximum context window
causes API errors or heavy billing. This script demonstrates how to proactively
monitor token counts and dynamically trim/prune messages and documents to fit
within the context window constraints.

We demonstrate two techniques:
1. Native LangChain `trim_messages` helper to keep message history under a token threshold.
2. Custom document trimmer for large RAG context payloads.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_community.chat_models import ChatOllama

# --------------------------------------------------------------------------
# Technique 1: Native Message Trimming using trim_messages
# --------------------------------------------------------------------------
print("--- Technique 1: Message Trimming ---")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, my name is Alice. I live in Seattle and enjoy hiking."),
    AIMessage(content="Hi Alice! Seattle has some wonderful trails like Mt. Si and Rattlesnake Ledge."),
    HumanMessage(content="I also have two dogs and love cooking Italian food."),
    AIMessage(content="That's great! Italian food is delicious. What kind of pasta do you like to make?"),
    HumanMessage(content="Can you summarize everything we talked about so far?"),
]

# We configure a message trimmer.
# - max_tokens: maximum allowed tokens for the context window
# - strategy: 'last' keeps the most recent messages (highly common for chat)
# - token_counter: function or model used to calculate tokens
# - include_system: whether to always keep the system message at the beginning
# - start_on: the type of message the window must start on (e.g., HumanMessage)
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=ChatOllama(model="qwen2.5:1.5b"),
    include_system=True,
    start_on="human",
)

# Apply the trimmer to messages
trimmed_messages = trimmer.invoke(messages)

print(f"Original message count: {len(messages)}")
print(f"Trimmed message count: {len(trimmed_messages)}")
print("\nTrimmed conversation structure:")
for msg in trimmed_messages:
    print(f"[{msg.type.upper()}]: {msg.content}")


# --------------------------------------------------------------------------
# Technique 2: Custom Document Trimmer for Large RAG Contexts
# --------------------------------------------------------------------------
print("\n--- Technique 2: Custom Document Trimmer ---")

from langchain_core.documents import Document

documents = [
    Document(page_content="Document 1: Detailed guidelines about coding standards in Python. " * 10, metadata={"source": "coding_std"}),
    Document(page_content="Document 2: Database architecture, indexes, and connection pooling settings. " * 10, metadata={"source": "db_arch"}),
    Document(page_content="Document 3: Frontend deployment script and caching configurations on CDN. " * 10, metadata={"source": "fe_deploy"}),
]

def estimate_tokens(text: str) -> int:
    # Quick token estimation (approx 4 chars per token)
    return len(text) // 4

def trim_documents_to_budget(docs: list[Document], max_token_budget: int) -> list[Document]:
    """Prunes retrieved documents to stay strictly under a specified token budget."""
    allowed_docs = []
    current_tokens = 0
    
    for doc in docs:
        doc_tokens = estimate_tokens(doc.page_content)
        if current_tokens + doc_tokens <= max_token_budget:
            allowed_docs.append(doc)
            current_tokens += doc_tokens
        else:
            # Optionally split the last document to pack as much as possible
            remaining_budget = max_token_budget - current_tokens
            if remaining_budget > 15: # only split if it is worth it
                chars_to_keep = remaining_budget * 4
                truncated_content = doc.page_content[:chars_to_keep] + "... [TRUNCATED]"
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata={**doc.metadata, "truncated": True}
                )
                allowed_docs.append(truncated_doc)
                print(f"Truncated document {doc.metadata['source']} to fit budget.")
            break
            
    return allowed_docs

budget = 200
retained_docs = trim_documents_to_budget(documents, max_token_budget=budget)

print(f"\nInitial documents: {len(documents)}")
print(f"Budget-friendly documents: {len(retained_docs)}")
for i, d in enumerate(retained_docs):
    print(f"Retained Doc {i+1}: Source={d.metadata['source']}, Length={len(d.page_content)} chars")
