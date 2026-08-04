"""
Intermediate Steps and Scratchpad Formatting
================================================
As an agent loops through tool calls, each (AgentAction, observation) pair
is accumulated into a list of "intermediate steps". Before the next model
call, these steps are formatted back into the prompt's `agent_scratchpad`
so the model can see what it already tried and what happened.
"""

from langchain.agents.format_scratchpad import format_to_openai_tool_messages
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, ToolMessage

# intermediate_steps: list[tuple[AgentAction, str]] accumulated by AgentExecutor
intermediate_steps = [
    (
        AgentAction(tool="get_weather", tool_input={"city": "Delhi"}, log="calling get_weather"),
        "32°C, sunny",  # the observation returned by the tool
    ),
]

# For tool-calling agents, the scratchpad is formatted as real ToolMessage
# objects appended to the message list (not raw text, unlike classic ReAct).
scratchpad_messages = format_to_openai_tool_messages(intermediate_steps)
for m in scratchpad_messages:
    print(type(m).__name__, m.content)

# For classic ReAct agents, the scratchpad is instead built as plain text like:
def format_react_scratchpad(steps) -> str:
    lines = []
    for action, observation in steps:
        lines.append(action.log)
        lines.append(f"Observation: {observation}")
    return "\n".join(lines)

print(format_react_scratchpad(intermediate_steps))
# -> "calling get_weather\nObservation: 32°C, sunny"

# This scratchpad text/messages is what gets injected into the
# MessagesPlaceholder("agent_scratchpad") slot of the agent's prompt on each loop.
