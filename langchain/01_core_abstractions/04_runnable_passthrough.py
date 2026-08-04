"""
RunnablePassthrough
======================
RunnablePassthrough passes its input through unchanged (or lets you attach
extra keys to it via `.assign()`). It is most commonly used in RAG chains
to forward the user's original question alongside retrieved context.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b")
parser = StrOutputParser()


def fake_retriever(question: str) -> str:
    # Stand-in for a real vector store retriever.
    return "LangChain is a framework for building LLM-powered applications."


prompt = ChatPromptTemplate.from_template(
    "Answer the question using only this context:\n{context}\n\nQuestion: {question}"
)

# RunnableParallel builds the dict {"context": ..., "question": ...}.
# RunnablePassthrough() means "just forward whatever came in" (the raw question string).
# A plain function like fake_retriever is auto-wrapped as a RunnableLambda by LCEL.
rag_chain = (
    RunnableParallel(
        context=RunnableLambda(fake_retriever),
        question=RunnablePassthrough(),
    )
    | prompt
    | model
    | parser
)

print(rag_chain.invoke("What is LangChain?"))

# .assign() adds a new key to an existing dict without dropping the others.
add_length = RunnablePassthrough.assign(question_length=lambda x: len(x["question"]))
print(add_length.invoke({"question": "How long am I?"}))
# -> {"question": "How long am I?", "question_length": 15}
