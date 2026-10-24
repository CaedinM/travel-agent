import pytest

from travel_agent.infrastructure.resources import (
    MAX_CHAT_MESSAGE_LENGTH,
    validate_chat_message_length,
)


def test_validate_chat_message_length_allows_the_limit() -> None:
    validate_chat_message_length("x" * MAX_CHAT_MESSAGE_LENGTH)


def test_validate_chat_message_length_rejects_messages_over_the_limit() -> None:
    with pytest.raises(ValueError, match="at most 400 characters"):
        validate_chat_message_length("x" * (MAX_CHAT_MESSAGE_LENGTH + 1))
