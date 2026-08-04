# LangGraph Multi-Agent Systems

This directory details architectural patterns for orchestrating multi-agent systems in LangGraph. It covers the supervisor delegation model, decentralized agent handoffs using `Command(goto=...)`, hierarchical team structuring, and state boundaries design (shared vs. isolated contexts).

---

## Table of Contents
1. [The Supervisor Pattern (Centralized Routing)](#1-the-supervisor-pattern-centralized-routing)
2. [Decentralized Handoffs via Command(goto=...)](#2-decentralized-handoffs-via-commandgoto)
3. [Hierarchical Agent Teams (Supervisor of Supervisors)](#3-hierarchical-agent-teams-supervisor-of-supervisors)
4. [Shared vs. Isolated State Architecture](#4-shared-vs-isolated-state-architecture)

---

## 1. The Supervisor Pattern (Centralized Routing)
The **Supervisor Pattern** delegates tasks using a hub-and-spoke topology (covered in [01_supervisor_pattern.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/09_multi_agent_systems/01_supervisor_pattern.py)).
- A central **Supervisor Node** (typically an LLM call configured with structured output) inspects the conversation state.
- The supervisor routes execution to a specialized **Worker Node** (e.g. coder, research worker).
- When the worker node completes its task, it writes its output and routes execution **back to the supervisor**.
- This centralized loop repeats until the supervisor determines the task is complete and routes execution to `END`.

---

## 2. Decentralized Handoffs via Command(goto=...)
Centralized routing can add unnecessary overhead if workers need to communicate directly. LangGraph supports decentralized **Agent Handoffs** using the **`Command`** object (covered in [02_agent_handoffs_command.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/09_multi_agent_systems/02_agent_handoffs_command.py)).
- A worker node returns a `Command(update=..., goto="peer_node")`.
- This updates the state and routes execution directly to the target peer node, bypassing a central supervisor and eliminating the need for conditional edges.

```python
from langgraph.types import Command

def support_agent(state: State):
    if needs_billing_help(state):
        # Update billing context and route directly to billing agent
        return Command(
            update={"messages": [AIMessage(content="Transferring to billing.")]},
            goto="billing_agent"
        )
```

---

## 3. Hierarchical Agent Teams (Supervisor of Supervisors)
For complex workflows, you can scale the supervisor pattern recursively by nesting team subgraphs (covered in [03_hierarchical_agent_teams.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/09_multi_agent_systems/03_hierarchical_agent_teams.py)).
- A top-level supervisor delegates tasks to **Team Supervisors**.
- Each team supervisor is a nested compiled subgraph managing its own local workers.
- Because subgraphs are runnables, the top-level parent graph treats each team as a single node, encapsulating the team's internal routing logic.

---

## 4. Shared vs. Isolated State Architecture
Designing state sharing between agents is a key architectural decision (covered in [04_shared_vs_isolated_state.py](file:///Users/saikrishnakuchimanchi/Downloads/Test/GenAi/langgraph/09_multi_agent_systems/04_shared_vs_isolated_state.py)):
- **Shared State**: All agents read and write to the same state object (e.g., sharing a single message list).
  - *Pros*: Simple to build; agents have full visibility into the conversation history.
  - *Cons*: Prompts can grow large; risk of cross-talk or confusion if agents see irrelevant intermediate messages.
- **Isolated State**: Each agent maintains its own private state. Communication occurs via adapter nodes that pass high-level summaries.
  - *Pros*: Reduces prompt sizes; decouples agent implementations.
  - *Cons*: Requires writing custom adapter functions to map inputs and outputs between state schemas.
