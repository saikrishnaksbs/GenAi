# LangChain Core Abstractions

This directory covers the foundational concepts and protocols of LangChain Core, focusing on the **Runnable** protocol, **LCEL (LangChain Expression Language)**, parallelism, context forwarding, configurations, argument binding, and fault-tolerance patterns.

---

## Table of Contents
1. [The Runnable Protocol](#1-the-runnable-protocol)
2. [LCEL Pipe Syntax & RunnableSequence](#2-lcel-pipe-syntax--runnablesequence)
3. [RunnableParallel & RunnableBranch](#3-runnableparallel--runnablebranch)
4. [RunnablePassthrough & Context Forwarding](#4-runnablepassthrough--context-forwarding)
5. [RunnableConfig & Callback propagation](#5-runnableconfig--callback-propagation)
6. [Dynamic Binding, Configuration, and Types](#6-dynamic-binding-configuration-and-types)
7. [Fault Tolerance: Retry & Fallbacks](#7-fault-tolerance-retry--fallbacks)
8. [Chain Topology & Graph Visualization](#8-chain-topology--graph-visualization)

---

## 1. The Runnable Protocol
Every standard component in LangChain (including Chat Models, LLMs, Prompts, Output Parsers, and Retrievers) implements the unified [Runnable](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/01_runnable_interface.py#L4-L15) interface. This guarantees a common interface to chain components together.

### Key Methods
The interface provides standard synchronous and asynchronous methods to interact with components:
- **`invoke(input, config=None)`**: Call the Runnable on a single input.
- **`ainvoke(input, config=None)`**: Asynchronous version of `invoke`.
- **`batch(inputs, config=None)`**: Run the Runnable concurrently on a list of inputs. It uses a thread pool to execute parallel requests.
- **`abatch(inputs, config=None)`**: Asynchronous version of `batch`.
- **`stream(input, config=None)`**: Stream back chunks of the output generator.
- **`astream(input, config=None)`**: Asynchronous version of `stream`.

In [01_runnable_interface.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/01_runnable_interface.py), you can see how to adapt any standard python function into a Runnable by using [RunnableLambda](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/01_runnable_interface.py#L25):
```python
from langchain_core.runnables import Runnable, RunnableLambda

def shout(text: str) -> str:
    return text.upper() + "!!!"

shout_runnable: Runnable = RunnableLambda(shout)
# Invocation
print(shout_runnable.invoke("hello world"))  # -> "HELLO WORLD!!!"
```

---

## 2. LCEL Pipe Syntax & RunnableSequence
**LangChain Expression Language (LCEL)** provides a declarative way to compose runnables using the pipe operator `|` (equivalent to a Unix pipe). Under the hood, this composition is represented as a [RunnableSequence](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/02_lcel_pipe_syntax.py#L5-L10).

In [02_lcel_pipe_syntax.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/02_lcel_pipe_syntax.py), a sequence is formed as:
```python
chain = prompt | model | parser
```
The output type of one Runnable must match the expected input type of the next:
- `prompt` outputs a `PromptValue`.
- `model` (e.g. ChatAnthropic) takes a `PromptValue` and outputs a `BaseMessage`.
- `parser` (e.g. StrOutputParser) takes a `BaseMessage` and outputs a `str`.

You can also construct it explicitly via [RunnableSequence](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/02_lcel_pipe_syntax.py#L30-L33):
```python
from langchain_core.runnables import RunnableSequence
explicit_chain = RunnableSequence(first=prompt, middle=[model], last=parser)
```

---

## 3. RunnableParallel & RunnableBranch

### RunnableParallel
[RunnableParallel](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/03_runnable_parallel_branch.py#L21-L24) allows executing multiple Runnables in parallel on the same input. The outputs are returned as a dictionary with matching keys. This is useful for running multiple tasks concurrently (e.g., summarizing text and detecting its sentiment at the same time).

```python
parallel_chain = RunnableParallel(
    summary=summarize_prompt | model | parser,
    sentiment=sentiment_prompt | model | parser,
)
```

### RunnableBranch
[RunnableBranch](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/03_runnable_parallel_branch.py#L42-L45) implements conditional routing. It takes a list of `(condition, runnable)` pairs and a default fallback. It evaluates conditions sequentially.

```python
branch = RunnableBranch(
    (lambda x: "code" in x["question"].lower(), code_chain),
    general_chain, # Default fallback
)
```

---

## 4. RunnablePassthrough & Context Forwarding
[RunnablePassthrough](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/04_runnable_passthrough.py#L4-L7) passes inputs through unmodified or adds new fields.
- **`RunnablePassthrough()`**: Returns the unmodified input it receives.
- **`RunnablePassthrough.assign(**kwargs)`**: Takes an input dictionary, runs the keyword-argument Runnables on it, and merges the resulting fields back into the input dictionary.

In [04_runnable_passthrough.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/04_runnable_passthrough.py#L30-L38), context is constructed for a RAG chain:
```python
rag_chain = (
    RunnableParallel(
        context=RunnableLambda(fake_retriever),
        question=RunnablePassthrough(),
    )
    | prompt
    | model
    | parser
)
```
And using `assign` in [04_runnable_passthrough.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/04_runnable_passthrough.py#L43-L44):
```python
add_length = RunnablePassthrough.assign(question_length=lambda x: len(x["question"]))
```

---

## 5. RunnableConfig & Callback Propagation
[RunnableConfig](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/05_runnable_config.py#L4-L8) is a standard Python dictionary containing metadata, tags, callbacks, and thread concurrency settings. It is passed as the second argument to invocation methods.

In [05_runnable_config.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/05_runnable_config.py#L25-L31):
```python
config: RunnableConfig = {
    "tags": ["greeting-chain", "demo"],
    "metadata": {"user_id": "u_123"},
    "run_name": "greet_user",
    "callbacks": [PrintTokenHandler()],
    "max_concurrency": 5,
}
```
Passing the config ensures that tracing platforms like **LangSmith** receive the context for all nested operations without manually injecting arguments to every function.

---

## 6. Dynamic Binding, Configuration, and Types
The Runnable protocol provides utilities to specialize or constrain executions:
- **`.bind(**kwargs)`**: Permanently attaches arguments to a Runnable (e.g., stop tokens, temperature, tool references).
- **`.with_config(**kwargs)`**: Binds a specific configuration to a Runnable.
- **`.with_types()`**: Overrides inferred input and output schemas, which is helpful when exporting services via LangServe.

Refer to [06_bind_with_config_with_types.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/06_bind_with_config_with_types.py) for examples.
```python
# Bind a stop sequence permanently to a model
model_with_stop = model.bind(stop=["\n\n"])
```

---

## 7. Fault Tolerance: Retry & Fallbacks
Production applications require high reliability. LangChain offers robust fault tolerance mechanisms:
- **`.with_retry()`**: Catches exceptions and retries the operation with exponential backoff.
- **`.with_fallbacks()`**: Specifies a sequence of alternative Runnables (such as backup models or fallback chains) to run if the primary fails.

In [07_with_retry_with_fallbacks.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/07_with_retry_with_fallbacks.py#L19-L22):
```python
resilient_model = primary_model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```
And fallback models in [07_with_retry_with_fallbacks.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/07_with_retry_with_fallbacks.py#L31):
```python
model_with_fallback = primary_model.with_fallbacks([backup_model])
```

---

## 8. Chain Topology & Graph Visualization
Complex chains with multiple parallel execution steps and branches can be difficult to inspect.
- **`.get_graph()`**: Any Runnable chain sequence can be queried to get a representation of its compiled node-and-edge topology.
- **`print_ascii()`**: Renders a text-based ASCII structure showing pipeline execution steps.
- **`draw_mermaid()`**: Generates Mermaid graph diagram code blocks.

See [08_get_graph_visualization.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/01_core_abstractions/08_get_graph_visualization.py) for examples.
