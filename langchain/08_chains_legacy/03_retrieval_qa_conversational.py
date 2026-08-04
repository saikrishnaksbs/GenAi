"""
RETRIEVALQA AND CONVERSATIONALRETRIEVALCHAIN
================================================
RetrievalQA is the legacy "ask a question over your documents" chain: it
takes a retriever + an LLM, fetches relevant chunks for the question, stuffs
them into a prompt, and asks the LLM to answer using only that context.

ConversationalRetrievalChain extends this with chat memory: it first
condenses the chat history + new question into a standalone question (so
follow-ups like "what about the second one?" resolve correctly), then runs
retrieval + answering like RetrievalQA.

Both are legacy — modern code uses create_retrieval_chain +
create_stuff_documents_chain (see file 04) composed with RunnableWithMessageHistory.
"""

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

# Assume documents were already embedded and stored previously.
vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top-4 chunks per query

# --- RetrievalQA: single-turn Q&A over documents ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",       # "stuff" = concatenate all retrieved chunks into one prompt
    retriever=retriever,
    return_source_documents=True,  # also return which chunks were used
)

result = qa_chain.invoke({"query": "What is our company's refund policy?"})
print(result["result"])
# -> "Refunds are issued within 30 days of purchase for unused items..."
for doc in result["source_documents"]:
    print(doc.metadata.get("source"))
# -> "policies/refunds.md"


# --- ConversationalRetrievalChain: multi-turn Q&A with follow-up resolution ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversational_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

response_1 = conversational_qa.invoke({"question": "What is our refund policy?"})
print(response_1["answer"])
# -> "Refunds are issued within 30 days of purchase for unused items..."

# This follow-up is ambiguous on its own, but the chain rewrites it using
# chat_history into something like "What is the refund policy for international orders?"
response_2 = conversational_qa.invoke({"question": "Does that apply to international orders?"})
print(response_2["answer"])
# -> "Yes, the same 30-day refund window applies to international orders,
#     though shipping fees are non-refundable."
