"""
PACKAGE STRUCTURE
===================
The LangChain ecosystem is split across several PyPI packages so that
projects only need to install what they actually use. Understanding the
split helps you know where an import should come from and why.

    langchain-core
        The foundation: base abstractions with (almost) no third-party
        dependencies. Runnable, BaseChatModel, BaseTool, BaseOutputParser,
        prompt templates, message types. Every other package depends on this.

    langchain
        Higher-level orchestration built on top of langchain-core: chains,
        agents, retrieval strategies, generic utilities that combine
        multiple primitives together (e.g. `langchain.evaluation`,
        `create_retrieval_chain`). Provider-agnostic by design.

    langchain-community
        Third-party integrations that the community maintains, bundled in
        one package: vector stores, document loaders, tools, less
        actively-maintained model providers. Broad but slower-moving and
        has more optional dependencies.

    Partner packages (langchain-openai, langchain-anthropic,
    langchain-pinecone, ...)
        First-class, independently-versioned integrations maintained (often
        jointly) by LangChain and the provider itself. Preferred over the
        equivalent langchain-community integration when one exists, since
        they get faster updates and stricter typing.

    langgraph
        A separate library for building stateful, multi-actor agent
        workflows as graphs -- more control than `langchain`'s prebuilt
        agents for complex branching/looping logic.

    langserve
        Deploys any Runnable as a REST API (FastAPI under the hood).
"""

# --- Core abstractions: no provider-specific code here ---
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

# --- Orchestration layer: built from langchain-core primitives ---
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- Partner packages: first-class, provider-maintained integrations ---
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOllama

# --- Community package: broad catalogue of third-party integrations ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback

# A quick sanity check of the layering: models from partner packages still
# implement the same langchain-core Runnable interface, so they're
# interchangeable in any chain built from langchain-core primitives.
openai_model: Runnable = ChatOllama(model="qwen2.5:1.5b")
anthropic_model: Runnable = ChatOllama(model="qwen2.5:1.5b")

prompt = ChatPromptTemplate.from_template("Translate '{text}' to French.")
parser = StrOutputParser()

# Same chain shape, just swap which partner package's model is plugged in.
openai_chain = prompt | openai_model | parser
anthropic_chain = prompt | anthropic_model | parser

print(type(openai_chain))
# -> <class 'langchain_core.runnables.base.RunnableSequence'>

# Rule of thumb when choosing an import:
#   1. Is it a core abstraction (Runnable, prompt, parser, message)?  -> langchain_core
#   2. Is it a specific provider's model/embeddings/vectorstore with
#      its own maintained package (openai, anthropic, pinecone, ...)? -> langchain_<provider>
#   3. Is it a broader third-party integration without a partner pkg? -> langchain_community
#   4. Is it orchestration logic combining multiple pieces?           -> langchain
