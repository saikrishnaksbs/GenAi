"""
TRAJECTORY EVALUATORS
========================
Agents don't just produce a final answer -- they take a sequence of actions
(tool calls) to get there. A trajectory evaluator judges the *whole path*
an agent took: did it call reasonable tools, in a sensible order, without
redundant or harmful steps, before arriving at its final answer.

This is different from string/criteria evaluators, which only look at the
final output text.
"""

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama

judge_model = ChatOllama(model="qwen2.5:1.5b", temperature=0)


# Tools available to the agent being evaluated. The evaluator needs their
# schemas to reason about whether each call in the trajectory was sensible.
@tool
def search_flights(origin: str, destination: str) -> str:
    """Search for available flights between two cities."""
    return f"2 flights found from {origin} to {destination}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: sunny, 72F"


trajectory_evaluator = load_evaluator(
    EvaluatorType.AGENT_TRAJECTORY, llm=judge_model
)

# The trajectory is recorded as a list of (AgentAction, observation) tuples,
# typically captured via `intermediate_steps` when running an AgentExecutor.
from langchain_core.agents import AgentAction

agent_trajectory = [
    (
        AgentAction(tool="search_flights", tool_input={"origin": "SFO", "destination": "JFK"}, log=""),
        "2 flights found from SFO to JFK",
    ),
    (
        AgentAction(tool="get_weather", tool_input={"city": "New York"}, log=""),
        "Weather in New York: sunny, 72F",
    ),
]

result = trajectory_evaluator.evaluate_agent_trajectory(
    input="Find me a flight from SFO to JFK and tell me the weather there.",
    prediction="I found 2 flights from SFO to JFK. The weather in New York is sunny and 72F.",
    agent_trajectory=agent_trajectory,
    tools=[search_flights, get_weather],
)
print(result)
# -> {'score': 1.0,
# ->  'reasoning': 'The agent used both tools appropriately and in a logical '
# ->               'order, and the final answer accurately reflects the tool outputs.'}

# A trajectory with an unnecessary or redundant tool call would typically
# score lower even if the final answer is still correct, since the
# evaluator penalizes wasted/irrelevant steps along the way.
