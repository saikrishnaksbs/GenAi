# Modern LangChain Chains (LCEL)

This directory details how to build chains in LangChain using **LangChain Expression Language (LCEL)**, which is the modern standard replacing legacy `Chain` classes (like `LLMChain`, `SequentialChain`, and `RetrievalQA`). 

LCEL provides a declarative way to compose arbitrary chains, offering features like automatic streaming, parallel execution, batching, async support, and intermediate step-tracking out of the box.

---

## Table of Contents
1. [LCEL Chains (Basic Prompt & Model Binding)](#1-lcel-chains-basic-prompt--model-binding)
2. [Sequential Workflows (Linear & Branching Pipelines)](#2-sequential-workflows-linear--branching-pipelines)
3. [Retrieval QA & Conversational RAG](#3-retrieval-qa--conversational-rag)
4. [Modern Document Stuffing & Retrieval Chain](#4-modern-document-stuffing--retrieval-chain)
5. [Summarization Strategies: Stuff & Map-Reduce](#5-summarization-strategies-stuff--map-reduce)
6. [SQL and API Execution Workflows](#6-sql-and-api-execution-workflows)
7. [Dynamic Routing (Branching Pipelines)](#7-dynamic-routing-branching-pipelines)

---

## 1. LCEL Chains (Basic Prompt & Model Binding)
Instead of using the deprecated `LLMChain` class, modern chains compose prompt templates, models, and output parsers directly using the pipe (`|`) operator.

- **File**: [01_llm_chain.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/01_llm_chain.py)
- **Modern Pattern**:
  ```python
  chain = prompt | llm | StrOutputParser()
  response = chain.invoke({"variable": "value"})
  ```
- **Key Methods**: Use `.invoke()`, `.batch()`, and `.stream()` (and their async `a*` counterparts) which are natively implemented on all Runnables.

---

## 2. Sequential Workflows (Linear & Branching Pipelines)
To run multiple tasks or models in series:
- **File**: [02_sequential_chains.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/02_sequential_chains.py)
- **Simple Linear Pipeline**: Direct output-to-input composition:
  ```python
  chain = {"input_key": step_one_chain} | step_two_chain
  ```
- **Complex Branching Pipeline**: Use `RunnablePassthrough.assign()` to feed forward outputs to future steps:
  ```python
  overall_chain = (
      RunnablePassthrough.assign(step_one_output=chain_one)
      | RunnablePassthrough.assign(step_two_output=chain_two)
  )
  ```

---

## 3. Retrieval QA & Conversational RAG
Querying files/documents is built using modern retrieval chains rather than `RetrievalQA` or `ConversationalRetrievalChain`.
- **File**: [03_retrieval_qa_conversational.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/03_retrieval_qa_conversational.py)
- **Single-Turn QA**: Combine a retriever with `create_stuff_documents_chain` and `create_retrieval_chain`.
- **Conversational QA (with History)**: Rewrite user questions with history via `create_history_aware_retriever`, then feed the result into the QA retriever.
- **Session History**: Wrap the entire RAG pipeline in `RunnableWithMessageHistory` to automatically save and retrieve chat history.

---

## 4. Modern Document Stuffing & Retrieval Chain
Focuses on composing RAG components using explicit document combines.
- **File**: [04_stuff_documents_and_retrieval_chain.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/04_stuff_documents_and_retrieval_chain.py)
- Demonstrates `create_stuff_documents_chain` and `create_retrieval_chain` with direct response streaming.

---

## 5. Summarization Strategies: Stuff & Map-Reduce
Instead of the legacy `load_summarize_chain`, modern summarization utilizes explicit LCEL design:
- **File**: [05_summarization_chains.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/05_summarization_chains.py)
- **Stuff**: Collects page contents from a list of documents and injects them straight into a summarization prompt.
- **Map-Reduce**: Maps an LLM summarizer over each document chunk in parallel using `.map()` on the runnable chain, then reduces the summaries into a final response.

---

## 6. SQL and API Execution Workflows
Monolithic database/API chains are replaced with modular LCEL components that separate query generation from execution:
- **File**: [06_sql_and_api_chains.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/06_sql_and_api_chains.py)
- **SQL RAG**: Generates SQL using `create_sql_query_chain`, runs the SQL using `QuerySQLDatabaseTool`, and synthesizes a natural language answer with an LLM.
- **API Chain**: Generates request URLs using an LLM, invokes them safely via a Python `requests` handler, and processes the result.

---

## 7. Dynamic Routing (Branching Pipelines)
Allows conditional execution based on categorization of user inputs:
- **File**: [07_dynamic_routing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/08_chains/07_dynamic_routing.py)
- **Custom Router Function**: Process inputs using a classification prompt, then pass the category to a Python routing function wrapped in `RunnableLambda` to invoke the correct sub-chain dynamically.
- **RunnableBranch**: Declaratively routes input using boolean conditions.

