"""Central model selection."""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Cheap and fast. The default for routine work: single-purpose subagents,
# tool-calling loops, anything run once per leg where cost multiplies.
LLM = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="low",
)

# Slower and pricier. Reserved for the reasoning that carries the conversation:
# the orchestrator, which juggles state, itinerary rules and the user.
POWERFUL_LLM = "openai:gpt-5"
