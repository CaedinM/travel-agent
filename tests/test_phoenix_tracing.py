from travel_agent.infrastructure.phoenix import runtime_chat_turn


def test_runtime_tracing_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "false")

    with runtime_chat_turn("opaque-thread", "hello") as span:
        assert span is None
