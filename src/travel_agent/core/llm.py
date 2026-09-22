"""Central model selection.

Every agent picks one of these two rather than naming a model itself, so
swapping models is a one-line change here.

Both are provider-prefixed strings that LangChain's create_agent resolves via
init_chat_model. Only OpenAI is configured, so both need OPENAI_API_KEY.
"""

# Cheap and fast. The default for routine work: single-purpose subagents,
# tool-calling loops, anything run once per leg where cost multiplies.
LLM = "openai:gpt-5-mini"

# Slower and pricier. Reserved for the reasoning that carries the conversation:
# the orchestrator, which juggles state, itinerary rules and the user.
POWERFUL_LLM = "openai:gpt-5"
