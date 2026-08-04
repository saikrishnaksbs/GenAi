import uuid
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger("my_app")
logging.basicConfig(level=logging.INFO)  # so you actually see log output


class ProductionLoggingHandler(BaseCallbackHandler):
    """Logs lifecycle events instead of just printing tokens — safer for prod."""
    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.info(f"Chain started | run_id={kwargs.get('run_id')}")

    def on_chain_end(self, outputs, **kwargs):
        logger.info(f"Chain finished | run_id={kwargs.get('run_id')}")

    def on_chain_error(self, error, **kwargs):
        logger.error(f"Chain failed | run_id={kwargs.get('run_id')} | error={error}")


def build_request_config(user_id: str, session_id: str) -> RunnableConfig:
    """Build a fresh, traceable config for each incoming request."""
    return {
        "tags": ["greeting-chain", "prod"],
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
        },
        "run_name": "greet_user",
        "run_id": uuid.uuid4(),
        "callbacks": [ProductionLoggingHandler()],
        "max_concurrency": 10,
    }


# --- THIS is what was missing: define the chain itself first ---
model = ChatOllama(model="qwen2.5:1.5b")
chain = ChatPromptTemplate.from_template("Say hi to {name}") | model | StrOutputParser()

# Now .with_config() has something real to attach to
base_chain = chain.with_config(tags=["myapp:v1"])


def handle_request(name: str, user_id: str, session_id: str) -> str:
    config = build_request_config(user_id, session_id)
    return base_chain.invoke({"name": name}, config=config)


if __name__ == "__main__":
    output = handle_request("Sai", user_id="u_123", session_id="s_456")
    print(output)