# LangGraph Deployment Platform & Studio

This directory details deploying, serving, and debugging LangGraph applications. It covers the `langgraph.json` manifest structure, visual debugging via LangGraph Studio, and REST API communication using the LangGraph SDK Client.

---

## Table of Contents
1. [LangGraph Platform & langgraph.json Manifest](#1-langgraph-platform--langgraphjson-manifest)
2. [LangGraph Studio (Visual Debugging)](#2-langgraph-studio-visual-debugging)
3. [LangGraph SDK Client (REST API Integration)](#3-langgraph-sdk-client-rest-api-integration)

---

## 1. LangGraph Platform & langgraph.json Manifest
To run graphs locally using the CLI or deploy them to the cloud-hosted LangGraph Platform, a project requires a **`langgraph.json`** manifest file at its root.
- The manifest declares dependencies, environment variables, and maps graph identifiers to compiled graph variables.

Example manifest structure (covered in [01_langgraph_json_config.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/11_deployment_platform/01_langgraph_json_config.py)):
```json
{
  "dependencies": ["."],
  "graphs": {
    "support_agent": "./11_deployment_platform/01_langgraph_json_config.py:graph"
  },
  "env": ".env"
}
```
The value is formatted as `path/to/module.py:variable_name`. The platform imports the module and exposes the compiled graph variable as a REST API endpoint.

---

## 2. LangGraph Studio (Visual Debugging)
**LangGraph Studio** is a visual IDE desktop application (and web interface served by `langgraph dev`) designed for debugging graphs (covered in [02_langgraph_studio_debugging.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/11_deployment_platform/02_langgraph_studio_debugging.py)).
- **Capabilities**:
  - Renders interactive node-edge diagrams.
  - Allows triggering manual runs with custom input payloads.
  - Displays state values and transitions step-by-step.
  - Inspects checkpoint histories.
  - Supports editing states mid-run to test human-in-the-loop approval gates.

---

## 3. LangGraph SDK Client (REST API Integration)
Once a graph is hosted, other services can interact with it. Rather than hand-rolling HTTP calls, you can use the official **`langgraph-sdk`** package (covered in [03_langgraph_sdk_client.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/11_deployment_platform/03_langgraph_sdk_client.py)):
- **`get_client`** / **`get_sync_client`**: Instantiates a client connection.
- Creates new threads.
- Starts execution runs and streams updates.
- Queries state history and updates checkpoints.

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:8123")

# Create a conversation thread
thread = client.threads.create()

# Start a streaming execution run
for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="support_agent",
    input={"messages": [{"role": "user", "content": "Hello"}]}
):
    print(chunk)
```
This decouples the frontend or client application from the graph implementation, allowing it to communicate with the hosted graph over standard API calls.
