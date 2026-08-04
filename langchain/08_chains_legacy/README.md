# Legacy LangChain Chains

This directory details LangChain's legacy **Chain** classes. While modern codebases use LCEL (LangChain Expression Language) and Runnable sequences, understanding these legacy classes is critical for maintaining older systems and understanding standard execution flows (e.g. map-reduce summarization, SQL generation, sequential pipelines).

---

## Table of Contents
1. [LLMChain (Prompt & Model Binding)](#1-llmchain-prompt--model-binding)
2. [Sequential Chains (Linear & Structured Pipelines)](#2-sequential-chains-linear--structured-pipelines)
3. [Legacy QA Retrieval: RetrievalQA & ConversationalRetrievalChain](#3-legacy-qa-retrieval-retrievalqa--conversationalretrievalchain)
4. [Modern LCEL QA Chains: Direct Successor](#4-modern-lcel-qa-chains-direct-successor)
5. [Summarization Strategies: Stuff, Map-Reduce, & Refine](#5-summarization-strategies-stuff-map-reduce--refine)
6. [SQL and API Chains](#6-sql-and-api-chains)

---

## 1. LLMChain (Prompt & Model Binding)
[LLMChain](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains_legacy/01_llm_chain.py#L4-L10) was the foundational class in early LangChain versions. It groups a prompt template, a model, and an optional output parser into a stateful object.

- **Legacy usage**: `chain = LLMChain(llm=model, prompt=prompt)`
- **Modern replacement**: `chain = prompt | model | parser`

---

## 2. Sequential Chains (Linear & Structured Pipelines)
To execute multiple steps in series where the output of one step serves as the input to the next, legacy LangChain used sequential chains (details in [02_sequential_chains.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains_legacy/02_sequential_chains.py)):
- **`SimpleSequentialChain`**: Linear chain where each step takes a single string input and produces a single string output.
- **`SequentialChain`**: Allows steps to declare multiple named inputs and outputs, storing intermediate results by key so they can be referenced by subsequent nodes.

In modern code, this is replaced by nested dictionaries, `RunnablePassthrough.assign()`, and `RunnableParallel`.

---

## 3. Legacy QA Retrieval: RetrievalQA & ConversationalRetrievalChain
For document question-answering, legacy code uses:
- **`RetrievalQA`**: Takes a retriever and an LLM, retrieves documents, and answers a query using the context (details in [03_retrieval_qa_conversational.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains_legacy/03_retrieval_qa_conversational.py#L4-L8)).
- **`ConversationalRetrievalChain`**: Adds conversational memory. It uses an LLM to condense the chat history and new question into a standalone query before invoking retrieval.

---

## 4. Modern LCEL QA Chains: Direct Successor
The direct successors to these legacy classes are LCEL-based helpers (covered in [04_stuff_documents_and_retrieval_chain.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains_legacy/04_stuff_documents_and_retrieval_chain.py#L4-L12)):
- **`create_stuff_documents_chain(model, prompt)`**: Maps retrieved documents to a `{context}` prompt key.
- **`create_retrieval_chain(retriever, document_chain)`**: Handles retrieval and passes content into the stuff chain.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

document_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)
```

---

## 5. Summarization Strategies: Stuff, Map-Reduce, & Refine
When generating summaries for large lists of documents, three main strategies are used (details in [05_summarization_chains.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains_legacy/05_summarization_chains.py)):
- **`stuff`**: Concatenates all documents into a single prompt. Fast, but will fail if the text size exceeds the model's context window.
- **`map_reduce`**: Summarizes each document chunk independently (**Map**), then combines and summarizes the summaries (**Reduce**). Allows parallel execution.
- **`refine`**: Summarizes the first chunk, then passes that summary along with the second chunk to refine the summary. Slow and sequential, but retains fine-grained detail.

---

## 6. SQL and API Chains
- **`SQLDatabaseChain`**: Renders database query generation from natural language prompts, executes the generated query against a connected database, and formulates a text answer (details in [06_sql_and_api_chains.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains_legacy/06_sql_and_api_chains.py#L4-L9)).
- **`APIChain`**: Constructs API calls from natural language requests, executes them, and translates the response back to the user.

In modern applications, these constraints are usually handled using tool-calling agents.
