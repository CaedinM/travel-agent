from contextlib import asynccontextmanager

MAX_CHAT_MESSAGE_LENGTH = 400


def validate_chat_message_length(message: str) -> None:
    """Reject chat messages that would exceed the public API input limit."""
    if len(message) > MAX_CHAT_MESSAGE_LENGTH:
        raise ValueError(
            f"message must be at most {MAX_CHAT_MESSAGE_LENGTH} characters"
        )


@asynccontextmanager
async def open_resources():
    """Open the direct Duffel and bounded-selector services used in production."""
    from travel_agent.core.llm import FLIGHT_SELECTOR_LLM
    from travel_agent.core.models import Resources
    from travel_agent.flights.duffel import DuffelClient
    from travel_agent.flights.prefetch import FlightPrefetchCache

    duffel_client = DuffelClient.from_environment()
    flight_prefetch = FlightPrefetchCache()
    try:
        yield Resources(
            pipeline="deterministic",
            duffel_client=duffel_client,
            flight_selector_model=FLIGHT_SELECTOR_LLM,
            flight_prefetch=flight_prefetch,
        )
    finally:
        flight_prefetch.close()
        await duffel_client.aclose()
