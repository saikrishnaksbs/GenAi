"""
CONVERSATIONAL RETRIEVAL (Modern Replacement for RetrievalQA & ConversationalRetrievalChain)
========================================================================================
In modern LangChain, RetrievalQA and ConversationalRetrievalChain are replaced by:
1. `create_stuff_documents_chain`: Combines retrieved documents into a context prompt.
2. `create_retrieval_chain`: Ties the retriever and the document combination chain together.
3. `create_history_aware_retriever`: Condenses conversational history and the new question
   into a query for the retriever.
4. `RunnableWithMessageHistory`: Automatically manages session-based chat history.
"""

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

# Create a self-contained in-memory vectorstore for demonstration
print("Indexing sample documents...")
docs = [
    Document(
        page_content="Our company's refund policy allows refunds within 30 days of purchase for unused items in original packaging. International orders are eligible, but return shipping fees are non-refundable.",
        metadata={"source": "policies/refunds.md"},
    ),
    Document(
        page_content="We offer standard shipping (3-5 business days) and expedited shipping (1-2 business days). All orders over $50 qualify for free standard shipping.",
        metadata={"source": "policies/shipping.md"},
    ),
]

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 1. Single-Turn RAG (Replacing Legacy RetrievalQA) ---
# Purpose: Given a user's question, retrieve relevant documents and generate an answer in one go.
# This approach does NOT maintain context or memory of previous chat turns.
print("\n=== Running Single-Turn RAG ===")

# Define the Prompt Template for answering questions using the retrieved context.
# The template requires two input variables:
# - `{context}`: The text content of the retrieved documents (stuffed by the document chain).
# - `{input}`: The user's original query.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question using only the context below.\n\n{context}"),
    ("human", "{input}"),
])

# 1. create_stuff_documents_chain:
#    Takes the LLM and the prompt template. It formats (or "stuffs") the retrieved
#    documents directly into the `{context}` variable of the prompt.
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# 2. create_retrieval_chain:
#    Links the retriever (which queries Chroma DB) and the document combination chain.
#    When invoked, it retrieves documents based on the input string, passes them to
#    combine_docs_chain, and returns a dictionary with keys: 'input', 'context', and 'answer'.
single_turn_rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Execute the single-turn pipeline
response = single_turn_rag_chain.invoke({"input": "What is our company's refund policy?"})
print("Answer:", response["answer"])
print("Sources:", [doc.metadata["source"] for doc in response["context"]])


# --- 2. Conversational RAG (Replacing Legacy ConversationalRetrievalChain) ---
# Purpose: Answer questions in a multi-turn chat format where the user's follow-up questions
# might reference earlier context (e.g., using pronouns like "that", "it", or "they").
print("\n=== Running Conversational RAG ===")

# --- Step A: Question Contextualization (History-Aware Retrieval) ---
# Goal: Rephrase ambiguous follow-up questions into standalone search queries.
# Example: If history is "Q: What is the refund policy? A: 30 days" and current query is
# "Does that apply to international orders?", this step rewrites it to:
# "Does the company's 30-day refund policy apply to international orders?".
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create_history_aware_retriever:
# This wrapper passes the chat history and latest user input to the LLM to get a reformulated
# standalone question, and then runs the underlying retriever using that reformulated question.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# --- Step B: Document Combining & QA Chain ---
# Goal: Construct the final system prompt that answers the user's question using the
# retrieved context documents AND allows the LLM to see the entire chat history.
qa_system_prompt = (
    "Answer the user's question using only the below context. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"), # Gives the LLM context of past exchanges for fluid dialog
    ("human", "{input}"),
])

# Combines the retrieved documents into the context prompt parameter.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combines the history-aware retriever with the QA chain to create a complete pipeline.
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

# --- Step C: Stateful Session-based Chat History Management ---
# Goal: Implement a session store that maps each user to their unique conversation history.
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or creates a message history object for a given session ID."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# RunnableWithMessageHistory:
# Wraps the conversational RAG chain to manage session history automatically.
# It uses the get_session_history helper to load past messages based on config.
# - input_messages_key: Maps user query to the "input" parameter.
# - history_messages_key: Maps history to the "chat_history" MessagesPlaceholder.
# - output_messages_key: Maps LLM response to the "answer" parameter to be saved to history.
stateful_rag_chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Define session configuration parameters. The session_id tracks the specific chat thread.
config = {"configurable": {"session_id": "user-session-123"}}

# Ask first question (initiating history)
resp1 = stateful_rag_chain.invoke(
    {"input": "What is the company's refund policy?"},
    config=config
)
print("Q1 Answer:", resp1["answer"])

# Ask a follow-up question.
# The history-aware retriever reformulates "Does that apply..." to "Does the refund policy apply..."
# so Chroma can fetch the correct document, while the LLM answers correctly.
resp2 = stateful_rag_chain.invoke(
    {"input": "Does that apply to international orders?"},
    config=config
)
print("Q2 (Follow-up) Answer:", resp2["answer"])
