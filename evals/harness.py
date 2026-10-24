"""Decision and conversation harnesses with injectable agent and traveller behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evals.contracts import Scenario, Turn
from evals.deterministic import evaluate_session
from evals.state import validate_state


class AgentUnderTest(Protocol):
    def __call__(self, user_message: str, state: dict, provider_fixtures: dict) -> tuple[Turn, dict]: ...


class Traveller(Protocol):
    def reply(self, assistant_message: str) -> str | None: ...


@dataclass
class ScriptedTraveller:
    """Synthetic traveller with private scenario messages; it has no evaluator access."""

    messages: list[str]

    def reply(self, assistant_message: str) -> str | None:
        return self.messages.pop(0) if self.messages else None


def run_decision(scenario: Scenario, user_message: str, agent: AgentUnderTest):
    """Evaluate a single decision from a labelled state and conversation prefix."""
    validate_state(scenario.initial_state, requires_return_home=scenario.requires_return_home)
    turn, final_state = agent(user_message, dict(scenario.initial_state), scenario.provider_fixtures)
    return evaluate_session(
        scenario, [turn], completion_reason="decision_complete", final_state=final_state
    )


def run_conversation(scenario: Scenario, agent: AgentUnderTest, traveller: Traveller | None = None):
    """Run up to the scenario limit; the production agent is supplied by the caller."""
    traveller = traveller or ScriptedTraveller(list(scenario.user_turns))
    validate_state(scenario.initial_state, requires_return_home=scenario.requires_return_home)
    state = dict(scenario.initial_state)
    turns: list[Turn] = []
    user_message = traveller.reply("")
    while user_message is not None and len(turns) < scenario.turn_limit:
        turn, state = agent(user_message, state, scenario.provider_fixtures)
        turns.append(turn)
        user_message = traveller.reply(turn.assistant_message)
    reason = "turn_limit" if len(turns) >= scenario.turn_limit else "traveller_complete"
    return evaluate_session(scenario, turns, completion_reason=reason, final_state=state)
