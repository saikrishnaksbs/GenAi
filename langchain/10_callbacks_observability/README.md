# LangChain Callbacks & Observability

This directory details LangChain's **Callback** system—the mechanism used to log, monitor, and trace executions. It covers custom event handlers, streaming callback execution, LangSmith platform integration, and token usage/cost measurement.

---

## Table of Contents
1. [The Callback Lifecycle & Custom Handlers](#1-the-callback-lifecycle--custom-handlers)
2. [Constructor vs. Invocation Scope](#2-constructor-vs-invocation-scope)
3. [Streaming Callbacks vs. Generator Streaming](#3-streaming-callbacks-vs-generator-streaming)
4. [LangSmith Tracing Integration](#4-langsmith-tracing-integration)
5. [Token Usage & Cost Tracking](#5-token-usage--cost-tracking)
6. [Structured Error Logging & Taxonomy](#6-structured-error-logging--taxonomy)

---

## 1. The Callback Lifecycle & Custom Handlers
LangChain fires events at every stage of a pipeline's lifecycle. You can tap into these events by subclassing **`BaseCallbackHandler`** (covered in [01_base_callback_handler.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/10_callbacks_observability/01_base_callback_handler.py)). Key hooks include:
- `on_llm_start` / `on_llm_end` / `on_llm_error`
- `on_chain_start` / `on_chain_end` / `on_chain_error`
- `on_tool_start` / `on_tool_end` / `on_tool_error`

```python
from langchain_core.callbacks import BaseCallbackHandler

class CustomLoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM starting with prompt: {prompts}")
```

---

## 2. Constructor vs. Invocation Scope
Callbacks can be registered at two different scopes:
- **Constructor-level**: Attached directly during object instantiation (e.g. `ChatOpenAI(callbacks=[handler])`). These execute for every single invocation of that specific instance.
- **Invocation-level**: Passed dynamically inside a `RunnableConfig` dict during a call (e.g. `chain.invoke(input, config={"callbacks": [handler]})`). These run only for that specific execution thread and automatically propagate down to nested sub-runs.

---

## 3. Streaming Callbacks vs. Generator Streaming
To handle token-by-token output delivery (e.g. to render typing effects in a Chat UI), LangChain offers two approaches (detailed in [02_streaming_callbacks.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/10_callbacks_observability/02_streaming_callbacks.py)):
1. **Piping/Streaming (`.stream()`)**: The preferred method. Standard python generators yield chunks sequentially.
2. **`on_llm_new_token` callback event**: Executed whenever a model receives a new token chunk from the provider (requires setting `streaming=True` on the model instantiation). This approach is useful when handling tokens via asynchronous side-effects (e.g. pushing values straight to a WebSocket connection).

---

## 4. LangSmith Tracing Integration
**LangSmith** is LangChain's observability platform. It records logs, execution times, token counts, and input/output parameters. Enabling tracing requires no code modifications and is configured entirely using environment variables (covered in [03_langsmith_tracing.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/10_callbacks_observability/03_langsmith_tracing.py)):

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__your_api_key
export LANGCHAIN_PROJECT=your_project_name
```
With these variables set, LangChain will automatically trace all executions and display them as interactive nested tree spans.

---

## 5. Token Usage & Cost Tracking
Monitoring API billing and usage is critical for production systems (details in [04_token_usage_tracking.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/10_callbacks_observability/04_token_usage_tracking.py)):
- **OpenAI Models**: Wrap calls inside the `get_openai_callback()` context manager to aggregate tokens and costs.
```python
from langchain_community.callbacks.manager import get_openai_callback

with get_openai_callback() as cb:
    response = model.invoke("Hi")
    print(cb.total_tokens) # Total tokens used
    print(cb.total_cost)   # Cost in USD
```
- **Other Providers**: Read the `usage_metadata` attribute directly off the returned `AIMessage` object:
```python
response = chat_model.invoke("Hi")
print(response.usage_metadata) # {'input_tokens': 12, 'output_tokens': 8, 'total_tokens': 20}
```

---

## 6. Structured Error Logging & Taxonomy
To easily trace and debug errors at scale, use custom callback handlers to format execution logs as JSON.
- **Error Categorization**: Separate issues by runtime taxonomy (e.g. `LLM_PROVIDER_ERROR`, `INPUT_VALIDATION_ERROR`, `TOOL_EXECUTION_ERROR`).
- **Trace Context**: Enrich JSON log payloads with run metadata, run IDs, parent run IDs, and invocation details.

See [05_structured_error_logging.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langchain/10_callbacks_observability/05_structured_error_logging.py) for examples.
