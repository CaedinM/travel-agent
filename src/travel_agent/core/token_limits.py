"""Configuration for monthly OpenAI token entitlements."""

import os

from dotenv import load_dotenv


load_dotenv()


def _positive_int_from_environment(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


# The free allowance is sized for one typical end-to-end planning conversation,
# including itinerary research and flight recommendations. These remain environment
# settings so pricing can be tuned without a code release.
FREE_MONTHLY_TOKEN_LIMIT = _positive_int_from_environment(
    "FREE_MONTHLY_TOKEN_LIMIT", 75_000
)
PAID_MONTHLY_TOKEN_LIMIT = _positive_int_from_environment(
    "PAID_MONTHLY_TOKEN_LIMIT", 750_000
)

# Reservations prevent simultaneous requests from spending the same remaining
# allowance. Actual provider-reported usage replaces the reservation on completion.
CHAT_TOKEN_RESERVATION = _positive_int_from_environment(
    "CHAT_TOKEN_RESERVATION", 8_000
)
FLIGHT_SELECTION_TOKENS_PER_LEG = _positive_int_from_environment(
    "FLIGHT_SELECTION_TOKENS_PER_LEG", 4_000
)
