from datetime import date

from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from travel_agent.agents.orchestrator import orchestrator_dynamic_prompt


def test_prompt_refreshes_current_date_between_model_calls(monkeypatch):
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[],
        state={"phase": "intake"},
    )
    current_date = date(2026, 10, 23)

    class CurrentDate(date):
        @classmethod
        def today(cls):
            return current_date

    monkeypatch.setattr("travel_agent.agents.orchestrator.date", CurrentDate)
    first_prompt = orchestrator_dynamic_prompt.wrap_model_call(
        request, lambda updated: updated.system_message.content,
    )
    current_date = date(2026, 10, 24)
    next_prompt = orchestrator_dynamic_prompt.wrap_model_call(
        request, lambda updated: updated.system_message.content,
    )

    assert first_prompt.startswith("Current date for this turn: 2026-10-23.")
    assert next_prompt.startswith("Current date for this turn: 2026-10-24.")
    assert "2026-10-23" not in next_prompt
