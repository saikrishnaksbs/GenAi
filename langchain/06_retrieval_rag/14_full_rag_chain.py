"""
Full RAG Chain (create_retrieval_chain + create_stuff_documents_chain)
==========================================================================
Puts the RAG pieces together into one production-shaped chain:
retriever -> "stuff" all retrieved docs into the prompt -> model -> answer,
while also returning the source documents used, for citation purposes.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Build a retriever (in practice this comes from real documents/loaders/splitters).
vectorstore = FAISS.from_texts(
    [
        "LangGraph is built on top of LangChain for stateful, graph-based agents.",
        "RAG stands for Retrieval-Augmented Generation.",
        "LangSmith is LangChain's tracing and evaluation platform.",
    ],
    OllamaEmbeddings(model="qwen3-embedding:8b"),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. "Stuff" chain: takes retrieved Documents and stuffs their text directly
#    into the {context} slot of the prompt (works well for small doc sets).
model = ChatOllama(model="qwen2.5:1.5b")
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the given context.\n\n{context}"),
    ("human", "{input}"),
])
document_chain = create_stuff_documents_chain(model, qa_prompt)

# 3. Retrieval chain: wires the retriever's output into the document chain's
#    {context}, and also returns the raw source documents in the response.
rag_chain = create_retrieval_chain(retriever, document_chain)

response = rag_chain.invoke({"input": "What is LangGraph built on?"})
print(response["answer"])
print(response["context"])  # list[Document] — the sources used, for citations
