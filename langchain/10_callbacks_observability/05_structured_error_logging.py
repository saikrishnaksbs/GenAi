"""
STRUCTURED ERROR LOGGING & TAXONOMY
===================================
Standard Python tracebacks are difficult to parse in log aggregators (e.g., Datadog, ELK).
This script demonstrates implementing a structured JSON error logging system in LangChain 
using a custom **`BaseCallbackHandler`** to categorize and structure errors across chains, 
LLMs, and tools.
"""

import json
import logging
from typing import Any, Dict, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Setup structured logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StructuredLLMLogger")


class StructuredErrorLoggingHandler(BaseCallbackHandler):
    """Custom callback handler to capture and log execution errors as structured JSON."""
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._log_error("LLM_PROVIDER_ERROR", error, run_id, parent_run_id, kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Determine error taxonomy/category
        error_class = error.__class__.__name__
        if "ValidationError" in error_class or "ValueError" in error_class:
            category = "INPUT_VALIDATION_ERROR"
        else:
            category = "CHAIN_EXECUTION_ERROR"
            
        self._log_error(category, error, run_id, parent_run_id, kwargs)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._log_error("TOOL_EXECUTION_ERROR", error, run_id, parent_run_id, kwargs)

    def _log_error(
        self,
        category: str,
        error: BaseException,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        extra_ctx: Dict[str, Any]
    ) -> None:
        log_payload = {
            "event": "execution_failed",
            "taxonomy": category,
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "context": extra_ctx
        }
        # Print logs as structured JSON strings
        logger.error(json.dumps(log_payload, indent=2))


# --------------------------------------------------------------------------
# Demonstration of Custom Error Logging
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Demonstration: Input Validation Error Catch ---")
    
    # Define a simple function that crashes on purpose to test error categorization
    def buggy_node(inputs: dict) -> dict:
        val = inputs.get("val", "")
        if not val:
            raise ValueError("Input parameter 'val' cannot be empty.")
        if val == "crash":
            raise RuntimeError("Database connection failure occurred.")
        return {"output": f"Processed {val}"}
        
    prompt = ChatPromptTemplate.from_template("Process: {val}")
    chain = prompt | RunnableLambda(buggy_node)
    
    # Instantiate handler
    handler = StructuredErrorLoggingHandler()
    
    # 1. Trigger VALUE_ERROR (validation taxonomy)
    try:
        chain.invoke({"val": ""}, config={"callbacks": [handler]})
    except Exception:
        pass # Exception logged by handler
        
    # 2. Trigger RUNTIME_ERROR (execution taxonomy)
    try:
        chain.invoke({"val": "crash"}, config={"callbacks": [handler]})
    except Exception:
        pass # Exception logged by handler
