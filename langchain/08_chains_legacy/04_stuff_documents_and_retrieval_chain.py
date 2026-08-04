"""
CREATE_STUFF_DOCUMENTS_CHAIN AND CREATE_RETRIEVAL_CHAIN
===========================================================
These two functions are the modern (LCEL-based) replacement for RetrievalQA.
Instead of a monolithic legacy Chain class, you compose small Runnables:

- create_stuff_documents_chain: takes an LLM + a prompt (which must include a
  {context} placeholder) and returns a Runnable that "stuffs" a list of
  Documents into that placeholder before calling the LLM.

- create_retrieval_chain: wraps a retriever + a "combine documents" chain
  (like the one above) into a single Runnable that takes a user question,
  retrieves docs, and passes them into the combine step, returning both the
  answer and the retrieved context.

This is included here for contrast even though the folder is "legacy",
since it's the direct successor to RetrievalQA shown in file 03.
"""

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# The prompt MUST contain a {context} variable — that's where retrieved
# document text gets injected by create_stuff_documents_chain.
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the context below.\n\n{context}"),
        ("human", "{input}"),
    ]
)

# Builds a Runnable: (docs, input) -> answer string
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# Builds a Runnable: {"input": question} -> {"answer": ..., "context": [Document, ...]}
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = retrieval_chain.invoke({"input": "What is our company's refund policy?"})
print(response["answer"])
# -> "Refunds are issued within 30 days of purchase for unused items..."

print([doc.metadata.get("source") for doc in response["context"]])
# -> ["policies/refunds.md", "policies/faq.md", ...]

# Because it's a plain Runnable, streaming works out of the box
for chunk in retrieval_chain.stream({"input": "Summarize the shipping policy."}):
    if "answer" in chunk:
        print(chunk["answer"], end="")
# -> streams the answer token by token as it's generated
