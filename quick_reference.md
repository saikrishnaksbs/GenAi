# 🚀 LangChain & LangGraph Quick Reference Guide

A comprehensive, easy-to-scan reference guide covering all keywords, concepts, and explicit short explanations for **every API, class, function, and symbol** across LangChain and LangGraph.

---

## 📌 Table of Contents
- [🔗 Part 1: LangChain Reference Guide](#-part-1-langchain-reference-guide)
  - [01. Core Abstractions](#01-core-abstractions-01_core_abstractions)
  - [02. Models](#02-models-02_models)
  - [03. Prompts](#03-prompts-03_prompts)
  - [04. Output Parsers](#04-output-parsers-04_output_parsers)
  - [05. Tools](#05-tools-05_tools)
  - [06. Retrieval & RAG](#06-retrieval--rag-06_retrieval_rag)
  - [07. Memory](#07-memory-07_memory)
  - [08. Chains](#08-chains-08_chains)
  - [09. Agents](#09-agents-09_agents)
  - [10. Callbacks & Observability](#10-callbacks--observability-10_callbacks_observability)
  - [11. Evaluation](#11-evaluation-11_evaluation)
  - [12. Ecosystem & Serving](#12-ecosystem--serving-12_misc_ecosystem)
- [⚡ Part 2: LangGraph Reference Guide](#-part-2-langgraph-reference-guide)
  - [01. Graph Fundamentals](#01-graph-fundamentals-01_graph_fundamentals)
  - [02. State Management](#02-state-management-02_state_management)
  - [03. Control Flow](#03-control-flow-03_control_flow)
  - [04. Persistence & Checkpointing](#04-persistence--checkpointing-04_persistence_checkpointing)
  - [05. Human-in-the-Loop](#05-human-in-the-loop-05_human_in_the_loop)
  - [06. Streaming](#06-streaming-06_streaming)
  - [07. Subgraphs & Reuse](#07-subgraphs--reuse-07_subgraphs_and_reuse)
  - [08. Time Travel & Replay](#08-time-travel--replay-08_time_travel_replay)
  - [09. Multi-Agent Systems](#09-multi-agent-systems-09_multi_agent_systems)
  - [10. Prebuilt Agents & Components](#10-prebuilt-agents--components-10_prebuilt_agents)
  - [11. Deployment Platform & Studio](#11-deployment-platform--studio-11_deployment_platform)

---

# 🔗 PART 1: LANGCHAIN REFERENCE GUIDE

### 01. Core Abstractions (`01_core_abstractions`)
> **Overview**: Core concepts of LangChain Expression Language (LCEL), the Runnable protocol, execution methods, and config propagation. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Standard Runnable Contract | `Runnable` | Abstract base interface protocol for all LCEL components (models, prompts, tools, parsers, chains). |
| Custom Function Wrapper | `RunnableLambda` | Converts custom Python functions or lambdas into LCEL-compatible `Runnable` execution steps. |
| Synchronous Execution | `.invoke(input)` | Processes a single input payload synchronously through a Runnable and returns a single output. |
| Asynchronous Execution | `.ainvoke(input)` | Non-blocking execution of a single input using Python's `asyncio` event loop. |
| Batch Execution | `.batch(inputs)` | Concurrently processes a list of inputs `[input1, input2, ...]` and returns a list of outputs. |
| Real-time Streaming | `.stream(input)` | Generator yielding output chunks iteratively as they are produced (token-by-token for LLMs). |
| Async Streaming | `.astream(input)` | Async generator streaming output chunks iteratively using async context. |
| Stream Transformation | `.transform(stream)` | Synchronously transforms an input stream into an output stream chunk by chunk. |
| Async Stream Transformation | `.atransform(stream)` | Asynchronously transforms an input stream into an output stream chunk by chunk. |
| LCEL Pipe Composition | `Prompt \| Model \| Parser` | Pipe operator (`\|`) chaining output of one Runnable into input of the next. |
| Concurrent Branching | `RunnableParallel` | Executes multiple Runnables concurrently in parallel on the exact same input dictionary. |
| Conditional Branching | `RunnableBranch` | Dynamically routes execution down different Runnable paths based on predicate functions. |
| Passthrough Input | `RunnablePassthrough` | Passes inputs through unchanged or allows adding key-value pairs into the payload. |
| Input Key Assignment | `RunnablePassthrough.assign()` | Injects or updates specific key-value pairs in payload while preserving all existing keys. |
| Runtime Configuration | `RunnableConfig` | Execution context dictionary containing `tags`, `metadata`, `callbacks`, and `run_name`. |
| Dynamic Field Configuration | `configurable_fields` | Declares parameters (e.g. temperature) that can be overridden dynamically at runtime. |
| Dynamic Runnable Swap | `configurable_alternatives` | Declares runtime alternative Runnable implementations (e.g. swapping model providers dynamically). |
| Parameter Binding | `.bind()` | Statically binds default parameters (e.g. `stop`, `tools`, `temperature`) to a Runnable. |
| Config Binding | `.with_config()` | Binds default runtime options (`tags`, `metadata`, `callbacks`) to a Runnable. |
| Schema Type Override | `.with_types()` | Explicitly overrides input and output Pydantic schema type specifications for a Runnable. |
| Automated Retries | `.with_retry()` | Wraps a Runnable with exponential backoff retry logic upon runtime exceptions. |
| Fallback Runnables | `.with_fallbacks()` | Defines fallback Runnables to try sequentially if the primary Runnable fails. |
| Graph AST Extraction | `runnable.get_graph()` | Extracts the internal Abstract Syntax Tree (AST) execution graph of a Runnable. |
| Mermaid Visualization | `draw_mermaid_png()` | Renders the Runnable AST graph visual structure as a PNG image byte stream. |
| ASCII Graph Printing | `print_ascii()` | Prints an ASCII representation of the Runnable graph structure to console stdout. |

---

### 02. Models (`02_models`)
> **Overview**: Chat models, legacy LLMs, caching, rate limiting, and context window trimming. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Local Ollama Model | `ChatOllama` | Chat model integration connecting to local Ollama LLM instances. |
| OpenAI Chat Model | `ChatOpenAI` | Chat model integration connecting to OpenAI GPT chat completions API. |
| Base Chat Model | `BaseChatModel` | Abstract base class for models operating on structured message lists. |
| Legacy LLM | `BaseLLM` | Abstract base class for legacy text completion models operating on raw strings. |
| User Message | `HumanMessage` | Represents input message payloads coming from human users. |
| Assistant Response | `AIMessage` | Represents message responses generated by the AI model. |
| System Instructions | `SystemMessage` | Sets background persona, system rules, and constraints for model behavior. |
| Tool Output Message | `ToolMessage` | Message conveying execution results returned by a tool back to the LLM. |
| Streaming Token Chunk | `AIMessageChunk` | Delta chunk object emitted during real-time LLM token streaming. |
| Randomness Hyperparameter | `temperature` | Controls sampling randomness (0.0 = deterministic, 1.0 = creative). |
| Nucleus Sampling | `top_p` | Considers tokens comprising top `p` cumulative probability mass. |
| Max Length Guard | `max_tokens` | Caps the maximum number of tokens generated in model completion output. |
| Reproducible Output | `seed` | Integer seed value for deterministic, reproducible LLM generations. |
| Frequency Penalty | `frequency_penalty` | Penalizes tokens based on their existing frequency in text to reduce repetition. |
| Tool Attachment | `.bind_tools()` | Binds tool/function schemas to chat model for function calling capability. |
| Structured Output Coercion | `.with_structured_output()` | Wraps model to guarantee output matching a Pydantic schema or JSON schema. |
| Schema Base | `BaseModel` | Pydantic base class used to construct strongly typed data validation models. |
| Global Response Cache | `set_llm_cache()` | Enables global caching layer to eliminate redundant LLM API calls. |
| In-Memory Cache | `InMemoryCache` | Fast in-memory key-value cache storing exact prompt-response matches. |
| Persistent SQLite Cache | `SQLiteCache` | SQLite database cache persisting LLM response pairs across app restarts. |
| API Rate Limiting | `InMemoryRateLimiter` | Token-bucket rate limiter preventing API quota limits and 429 errors. |
| Context Trimming | `trim_messages()` | Truncates chat message history to fit strictly within model context token window. |
| Semantic Vector Cache | `RedisSemanticCache` | Redis vector cache returning cached responses for semantically similar queries. |
| Local Embedding Model | `OllamaEmbeddings` | Text embedding generator using local Ollama embedding models. |

---

### 03. Prompts (`03_prompts`)
> **Overview**: Prompt design patterns, templates, few-shot examples, dynamic example selectors. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Text Prompt Template | `PromptTemplate.from_template()` | Creates text prompt template with variable interpolation (`{variable}`). |
| Chat Prompt Template | `ChatPromptTemplate.from_messages()` | Builds role-based chat prompt templates (`SystemMessage`, `HumanMessage`). |
| Dynamic Message Slot | `MessagesPlaceholder` | Slot in prompt template for inserting variable-length message arrays. |
| Text Few-Shot Template | `FewShotPromptTemplate` | Formats few-shot exemplar examples into raw prompt text for LLMs. |
| Chat Few-Shot Template | `FewShotChatMessagePromptTemplate` | Formats few-shot exemplars into structured chat message lists. |
| Semantic Example Selector | `SemanticSimilarityExampleSelector` | Uses vector embeddings to select top-k most relevant few-shot examples. |
| Length-Based Selector | `LengthBasedExampleSelector` | Selects few-shot examples dynamically based on remaining prompt token capacity. |
| Static Partialing | `prompt.partial()` | Pre-fills static variable values in prompt template ahead of execution. |
| Composite Prompt | `PipelinePromptTemplate` | Assembles multiple prompt templates hierarchically into a single output prompt. |

---

### 04. Output Parsers (`04_output_parsers`)
> **Overview**: Extracting structured data (Pydantic, JSON, List, XML) and auto-correcting errors. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| String Parser | `StrOutputParser` | Extracts plain string text from `AIMessage` output objects. |
| Pydantic JSON Parser | `PydanticOutputParser` | Parses model text output into a strongly typed Pydantic object. |
| Format Generator | `get_format_instructions()` | Generates string instructions telling LLM how to format JSON output for parser. |
| Lightweight JSON Parser | `StructuredOutputParser` | Parses structured JSON response into a Python dictionary. |
| Schema Definition | `ResponseSchema` | Defines expected key name, data type, and description for `StructuredOutputParser`. |
| List Parser | `CommaSeparatedListOutputParser` | Parses comma-separated string outputs into a list of strings. |
| XML Tag Parser | `XMLOutputParser` | Parses XML-tagged output strings into a structured Python dictionary. |
| Self-Correcting Parser | `OutputFixingParser` | Re-prompts the LLM with error details when initial parsing fails. |
| Error Retry Parser | `RetryWithErrorOutputParser` | Sends original prompt + bad output + error back to LLM to retry parsing. |

---

### 05. Tools (`05_tools`)
> **Overview**: Function tools, schemas, built-in toolkits, injection, and execution sandboxing. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Tool Decorator | `@tool` | Decorator converting a Python function into a LangChain tool with schema. |
| Structured Tool Factory | `StructuredTool.from_function()` | Factory creating tools with explicit multi-argument Pydantic schemas. |
| Tool Argument Schema | `args_schema` | Property defining Pydantic schema class for strict tool input validation. |
| Field Specifier | `Field` | Pydantic descriptor defining parameter descriptions, defaults, and constraints. |
| Web Search Tool | `DuckDuckGoSearchRun` | Built-in search tool executing DuckDuckGo search queries. |
| File System Toolkit | `FileManagementToolkit` | Toolkit providing tools for reading, writing, and listing files. |
| Extracted Tool Calls | `AIMessage.tool_calls` | List of requested tool invocations extracted from model response. |
| Tool Result Message | `ToolMessage` | Message containing tool execution output bound to a specific call ID. |
| Prebuilt Tool Executor | `ToolNode` | LangGraph prebuilt node that executes requested tool calls automatically. |
| Argument Injection | `InjectedToolArg` | Annotation for tool parameters injected at runtime rather than generated by LLM. |
| Tool Call Payload | `ToolCall` | Typed dictionary schema representing a requested tool call (`name`, `args`, `id`). |
| Sandboxed Code Exec | `PythonREPLTool` | Isolated Python REPL tool for executing dynamic code safely. |
| Safe Tool Error Handling | `handle_tool_error=True` | Configures tools to return error strings to model instead of raising exceptions. |

---

### 06. Retrieval & RAG (`06_retrieval_rag`)
> **Overview**: Advanced Retrieval-Augmented Generation (Loaders, Splitters, Embeddings, Hybrid Search, RERAG). 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Text File Ingestion | `TextLoader` | Loads plain text files into document objects. |
| PDF Ingestion | `PyPDFLoader` | Parses PDF documents into page-by-page document objects. |
| Web Page Ingestion | `WebBaseLoader` | Downloads web URLs and extracts main text content into documents. |
| Recursive Chunking | `RecursiveCharacterTextSplitter` | Splits document text recursively using characters (`\n\n`, `\n`, space). |
| Token Chunking | `TokenTextSplitter` | Splits text into chunks based on exact token count bounds. |
| Document Schema | `Document` | Standard data class containing `page_content` string and `metadata` dict. |
| Embedding Base Class | `Embeddings` | Abstract base class for text vector embedding models. |
| OpenAI Embeddings | `OpenAIEmbeddings` | Text embedding generator using OpenAI embedding API. |
| In-Memory Vector Store | `FAISS` | High-performance in-memory vector store for similarity search. |
| Open-Source Vector DB | `Chroma` | Open-source persistent vector database for similarity search. |
| Similarity Search | `similarity_search()` | Vector distance search returning top-k matching documents. |
| Diversity Search | `mmr()` | Maximal Marginal Relevance search balancing query similarity with diversity. |
| Base Retriever Contract | `BaseRetriever` | Abstract base class for document retrieval interfaces. |
| Vectorstore Conversion | `vectorstore.as_retriever()` | Converts a vector database instance into a standard `BaseRetriever`. |
| Query Expansion RAG | `MultiQueryRetriever.from_llm()` | Generates multiple query variations via LLM to boost recall. |
| Parent Document RAG | `ParentDocumentRetriever` | Indexes small chunk vectors while retrieving full parent context. |
| Parent Document Store | `InMemoryStore` | Key-value store holding full parent documents for `ParentDocumentRetriever`. |
| Self-Query Retriever | `SelfQueryRetriever.from_llm()` | Translates natural language into vector search + structured metadata filters. |
| Metadata Field Info | `AttributeInfo` | Attribute metadata descriptor used to build filter schemas for `SelfQueryRetriever`. |
| Context Compression | `ContextualCompressionRetriever` | Compresses retrieved document text to keep only relevant sentences. |
| LLM Sentence Extractor | `LLMChainExtractor` | Compression module using LLM to extract relevant sentences from chunks. |
| Hybrid Search | `EnsembleRetriever` | Combines sparse (BM25) and dense (Vector) retrievers via Reciprocal Rank Fusion. |
| Sparse Keyword Retriever | `BM25Retriever` | Keyword-based sparse text retriever implementing BM25 ranking algorithm. |
| Reranking Cross-Encoder | `CohereRerank` | Second-stage cross-encoder re-ordering retrieved documents by true relevance. |
| Pipeline Compressor | `DocumentCompressorPipeline` | Pipeline combining multiple document transformers and compressor modules. |
| Vector DB Indexing | `index()` | Synchronizes source documents with vector store (handles inserts, updates, deletes). |
| Document Hash Tracking | `SQLRecordManager` | Tracks document hashes in SQL database to prevent duplicate vector indexing. |
| Document Combination | `create_stuff_documents_chain` | Stuffs a list of document contents into prompt context for model synthesis. |
| End-to-End RAG Chain | `create_retrieval_chain` | Chains retriever and document synthesis chain into an end-to-end RAG pipeline. |
| Multi-Vector Indexing | `MultiVectorRetriever` | Indexes multiple vectors per document (e.g. summaries, parent chunks, images). |

---

### 07. Memory (`07_memory`)
> **Overview**: Managing conversation state across multiple turns (Buffer, Summary, Token-bound, Persistent Stores). 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Buffer Memory | `ConversationBufferMemory` | Memory tracking full raw transcript of all conversation turns. |
| Sliding Window Memory | `ConversationBufferWindowMemory` | Keeps a sliding window of the last K conversation turns. |
| Summary Memory | `ConversationSummaryMemory` | Uses an LLM to continuously summarize conversation history. |
| Summary Buffer Memory | `ConversationSummaryBufferMemory` | Combines buffer memory with LLM summarization when token limits overflow. |
| Token Bounded Memory | `ConversationTokenBufferMemory` | Prunes conversation history strictly based on total token count bounds. |
| LCEL History Wrapper | `RunnableWithMessageHistory` | LCEL wrapper dynamically injecting session chat history into chains. |
| Session History Factory | `get_session_history` | Factory function resolving chat message history stores per session/thread ID. |
| In-Memory History Store | `InMemoryChatMessageHistory` | In-memory store saving chat messages for a single session. |
| Redis History Store | `RedisChatMessageHistory` | Persistent Redis store storing chat history across app restarts. |
| PostgreSQL History Store | `PostgresChatMessageHistory` | Persistent PostgreSQL database store for chat message history. |

---

### 08. Chains (`08_chains`)
> **Overview**: Pre-built and LCEL sequential chains, conversational RAG, SQL generation, dynamic routing. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Legacy Chain | `LLMChain` | Legacy chain binding prompt template, model, and output parser. |
| History Reformulation | `create_history_aware_retriever` | Reformulates standalone user queries considering conversation history before retrieval. |
| Stuff Documents RAG | `create_stuff_documents_chain` | Passes formatted document context into prompt for RAG synthesis. |
| Conversational RAG Pipeline | `create_retrieval_chain` | Chains history-aware retriever with stuff-documents synthesis chain. |
| Summarization Chain | `load_summarize_chain` | Legacy factory for creating Map-Reduce, Refine, or Stuff summarization chains. |
| Text-to-SQL Chain | `create_sql_query_chain` | Creates LCEL chain translating natural text into SQL queries for database execution. |
| SQL Execution Tool | `QuerySQLDataBaseTool` | Executes SQL queries against connected database and returns raw results. |

---

### 09. Agents (`09_agents`)
> **Overview**: Evolution from legacy AgentExecutor to modern LangGraph ReAct agents. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| ReAct Agent | `create_react_agent` | Creates a ReAct reasoning-and-action agent logic chain. |
| Tool Calling Agent | `create_tool_calling_agent` | Creates an agent leveraging native model function calling capability. |
| Legacy Agent Runtime | `AgentExecutor` | Legacy execution runtime driving agent reasoning loops and tool dispatch. |
| LangGraph 1-Line Agent | `langgraph.prebuilt.create_react_agent` | Modern LangGraph prebuilt 1-line compiled ReAct agent graph. |
| Graph State Builder | `StateGraph` | Core class used to define and compile stateful multi-node execution graphs. |
| Node Registration | `add_node()` | Registers a named processing node into a state graph. |
| Dynamic Routing Edge | `add_conditional_edges()` | Registers a router function selecting the next destination node dynamically. |
| Infinite Loop Guard | `recursion_limit` | Runtime config capping maximum node transitions allowed in a single execution. |
| State Checkpointer | `MemorySaver` | In-memory checkpointer persisting agent graph state across multiple conversation turns. |
| Pre-Node Breakpoint | `interrupt_before` | Halts graph execution prior to executing specified nodes (e.g. human approval). |

---

### 10. Callbacks & Observability (`10_callbacks_observability`)
> **Overview**: Tracing, custom event listeners, streaming tokens, token counting, structured logging. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Callback Base Handler | `BaseCallbackHandler` | Base class for creating custom lifecycle listeners (`on_llm_start`, `on_tool_start`). |
| Async Streaming Handler | `AsyncIteratorCallbackHandler` | Callback handler streaming generated LLM tokens to an async iterator queue. |
| Token Event Hook | `on_llm_new_token` | Callback hook firing whenever a new token chunk is generated by an LLM. |
| LangSmith Tracing Decorator | `@traceable` | LangSmith decorator automatically tracing inputs, outputs, and execution metrics. |
| Tracing Environment Var | `LANGCHAIN_TRACING_V2` | Environment variable enabling automatic run logging to LangSmith dashboard. |
| OpenAI Cost Tracker | `get_openai_callback()` | Context manager capturing total prompt tokens, completion tokens, and dollar cost. |

---

### 11. Evaluation (`11_evaluation`)
> **Overview**: Automated scoring criteria, A/B testing, trajectory evaluation, and mock testing. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Evaluator Loader | `load_evaluator()` | Factory function loading pre-built LLM evaluation chains (Criteria, Trajectory, Pairwise). |
| Criteria Evaluator | `EvaluatorType.CRITERIA` | Evaluator type assessing responses against custom criteria (conciseness, accuracy). |
| Pairwise Evaluator | `EvaluatorType.PAIRWISE_STRING` | Evaluator comparing two LLM candidate responses side-by-side. |
| Trajectory Evaluator | `EvaluatorType.TRAJECTORY` | Evaluator scoring agent tool selection correctness and reasoning steps. |
| Mock Chat Model | `FakeMessagesListChatModel` | Mock chat model returning pre-configured message responses for unit testing. |

---

### 12. Ecosystem & Serving (`12_misc_ecosystem`)
> **Overview**: Package imports structure, REST API deployment via LangServe/FastAPI, security, guardrails. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Core Package | `langchain_core` | Lightweight base package containing core interfaces, schemas, and LCEL primitives. |
| Community Package | `langchain_community` | Third-party integrations (vector stores, toolkits, document loaders). |
| LangServe REST Binding | `langserve.add_routes()` | Binds LCEL Runnables directly to FastAPI REST app endpoints. |
| FastAPI Framework | `FastAPI` | High-performance Python web framework used for serving model APIs. |
| Chain Serialization | `dumps()` / `loads()` | Utility functions for serializing LCEL chain configurations to/from JSON/YAML. |
| SSE Streaming Responses | `StreamingResponse` / `EventSourceResponse` | FastAPI HTTP response classes for Server-Sent Events (SSE) token streaming. |

---

# ⚡ PART 2: LANGGRAPH REFERENCE GUIDE

### 01. Graph Fundamentals (`01_graph_fundamentals`)
> **Overview**: Core concepts of StateGraph, nodes, edges, reducers, and graph compilation. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| State Graph Builder | `StateGraph` | Core class used to define, wire, and compile stateful multi-node execution graphs. |
| Register Node | `add_node(name, action)` | Registers a named node function into the state graph. |
| Register Edge | `add_edge(start, end)` | Adds a deterministic transition edge connecting two graph nodes. |
| Graph Boundary Constants | `START` / `END` | Reserved graph constants marking the entry and terminal execution points. |
| Channel Reducer Typing | `Annotated` | Type annotation used to attach reducers to state fields (e.g. `Annotated[list, add_messages]`). |
| List Concatenation Reducer | `operator.add` | Standard Python addition operator used as a list-concatenation reducer. |
| Message Deduplicating Reducer | `add_messages` | Built-in state reducer merging message lists with ID deduplication and updates. |
| Dynamic Conditional Router | `add_conditional_edges()` | Registers a router function selecting the next destination node dynamically. |
| Graph Compiler | `graph.compile()` | Validates graph structure and returns an executable `CompiledStateGraph` application. |

---

### 02. State Management (`02_state_management`)
> **Overview**: State schemas (TypedDict, Pydantic), custom reducers, partial state updates, long-term memory. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Dictionary State Schema | `TypedDict` | Standard Python dictionary type definition used to define graph state schema. |
| Pydantic State Schema | `BaseModel` | Pydantic base class used for strict validation of graph state attributes. |
| Pydantic Field Validator | `@field_validator` | Pydantic decorator adding custom data validation logic to state fields. |
| Any Message Union | `AnyMessage` | Union type representing any valid message schema (`HumanMessage`, `AIMessage`, etc.). |
| Delete Message Control | `RemoveMessage` | Control message sent to `add_messages` reducer to purge a message by ID. |
| Long-Term Memory Store | `InMemoryStore` | Persistent key-value memory store for long-term cross-thread data storage. |
| Save Long-Term Record | `store.put()` | Saves a record under a specific namespace and key into the long-term store. |
| Query Long-Term Store | `store.search()` | Queries and retrieves records from long-term memory store by namespace filter. |

---

### 03. Control Flow (`03_control_flow`)
> **Overview**: Branching, dynamic routing, parallel execution (fan-out/fan-in), and Send API map-reduce. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Map-Reduce Dispatch Primitive | `Send(node_name, arg_payload)` | Map-Reduce primitive dispatching dynamic parallel task instances to worker nodes. |
| Explicit Router Type | `typing.Literal` | Specifies explicit allowed string return types for conditional edge routing functions. |

---

### 04. Persistence & Checkpointing (`04_persistence_checkpointing`)
> **Overview**: Checkpointers (Memory, SQLite, Postgres), thread persistence, state inspection. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| In-Memory Checkpointer | `MemorySaver` | In-memory state checkpointer keeping thread history during runtime. |
| Persistent SQLite Saver | `SqliteSaver` | SQLite-backed persistent checkpointer saving thread snapshots across app restarts. |
| Enterprise Postgres Saver | `PostgresSaver` | Enterprise PostgreSQL checkpointer for production thread state storage. |
| Session Thread Identifier | `thread_id` | Unique session key passed in `config` to segregate conversation state per user. |
| State Snapshot Inspector | `app.get_state(config)` | Fetches current state snapshot, pending targets, and checkpoint history for a thread. |
| Thread State Values | `state.values` | Dictionary containing current state channel values for a thread snapshot. |
| Scheduled Next Nodes | `state.next` | Tuple listing the next node(s) scheduled to execute in the thread. |

---

### 05. Human-in-the-Loop (`05_human_in_the_loop`)
> **Overview**: Interrupts (`interrupt_before`, `interrupt`), state editing, approval workflows, dynamic resume. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Static Node Breakpoints | `interrupt_before` / `interrupt_after` | Compile-time parameters halting execution before/after specified nodes. |
| State Edit Primitive | `app.update_state(config, values)` | Manually updates or overwrites state channel values for a paused thread. |
| Dynamic Inline Breakpoint | `interrupt(query_payload)` | Inline function pausing graph execution from inside a node body and returning data to client. |
| Dynamic Resume Signal | `Command(resume=value)` | Control object resuming execution of an interrupted thread with injected client input. |

---

### 06. Streaming (`06_streaming`)
> **Overview**: Real-time streaming modes (`values`, `updates`, `messages`, `astream_events`). 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Stream Full State Snapshots | `stream_mode="values"` | Streams full state dictionary snapshots after each node completion. |
| Stream State Deltas | `stream_mode="updates"` | Streams only the delta state updates emitted by each executed node. |
| Stream Real-Time LLM Tokens | `stream_mode="messages"` | Streams real-time LLM message token chunks as they are generated. |
| Granular Event Stream | `app.astream_events()` | Streams fine-grained granular events (node start/end, token deltas). |

---

### 07. Subgraphs & Reuse (`07_subgraphs_and_reuse`)
> **Overview**: Subgraph nesting, schema mapping between parent and child graphs, graph factories. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Embed Subgraph | `parent_graph.add_node("child", child_graph)` | Embeds a compiled child graph directly into a parent graph node. |

---

### 08. Time Travel & Replay (`08_time_travel_replay`)
> **Overview**: Inspecting state history, replaying execution from past checkpoints, and branching state forks. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| State History Iterator | `app.get_state_history(config)` | Returns an iterator over all historical state checkpoints for a thread ID. |
| Checkpoint Snapshot ID | `checkpoint_id` | Unique identifier referencing a specific historical execution state snapshot. |

---

### 09. Multi-Agent Systems (`09_multi_agent_systems`)
> **Overview**: Multi-agent architectures (Supervisor, Direct Handoffs using Command, Hierarchical Teams). 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Direct Agent Handoff | `Command(goto="target_agent", update=...)` | Multi-agent control signal performing direct handoff to another agent node. |

---

### 10. Prebuilt Agents & Components (`10_prebuilt_agents`)
> **Overview**: Prebuilt agent abstractions (`create_react_agent`, `ToolNode`, `tools_condition`). 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| 1-Line ReAct Graph Agent | `create_react_agent(model, tools)` | LangGraph prebuilt function constructing a complete ReAct agent graph in one call. |
| Prebuilt Tool Node | `ToolNode(tools)` | Prebuilt node that automatically inspects state messages and executes requested tool calls. |
| Prebuilt Tool Router | `tools_condition` | Prebuilt router function checking if the last message contains tool calls to decide next node. |

---

### 11. Deployment Platform & Studio (`11_deployment_platform`)
> **Overview**: Deploying graphs to LangGraph Platform/Cloud, visual debugging with Studio, Python SDK. 

| Topic / Concept | Key API / Symbol | Short Description & Usage |
| :--- | :--- | :--- |
| Manifest Config | `langgraph.json` | Configuration manifest file defining graph entry points, dependencies, and environment variables. |
| Cloud SDK Clients | `get_client()` / `get_sync_client()` | Client SDK constructors for connecting to LangGraph Cloud / Server APIs. |
