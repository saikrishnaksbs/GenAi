"""
AgentAction and AgentFinish
==============================
These are the two possible outputs of a single agent "decision" step:

- AgentAction: "call this tool with these inputs" — has `tool`, `tool_input`,
  and `log` (the raw reasoning text that led to this decision).
- AgentFinish: "I'm done, here is the final answer" — has `return_values`
  (typically {"output": "..."}) and `log`.

AgentExecutor's loop is essentially: call agent -> if AgentAction, run the
tool and loop again; if AgentFinish, stop and return.
"""

from langchain_core.agents import AgentAction, AgentFinish

# Constructed manually here to show their shape — normally these come from
# the agent's `.invoke()` / `.plan()` call, parsed from the model's output.

action = AgentAction(
    tool="get_weather",
    tool_input={"city": "Bengaluru"},
    log="Thought: I need the current weather.\nAction: get_weather\nAction Input: {\"city\": \"Bengaluru\"}",
)
print(action.tool, action.tool_input)

finish = AgentFinish(
    return_values={"output": "The weather in Bengaluru is 27°C and sunny."},
    log="Thought: I now have the final answer.\nFinal Answer: The weather in Bengaluru is 27°C and sunny.",
)
print(finish.return_values["output"])


# A minimal illustration of the executor loop's core logic:
def run_step(agent_output):
    if isinstance(agent_output, AgentAction):
        # look up and call the real tool by name
        observation = f"[result of calling {agent_output.tool} with {agent_output.tool_input}]"
        return ("continue", observation)
    elif isinstance(agent_output, AgentFinish):
        return ("stop", agent_output.return_values["output"])


print(run_step(action))
print(run_step(finish))
