"""Flight-search subagent."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool

from llm import LLM
from models import BestOffer


def build_flights_agent(search_leg: BaseTool, get_offer_details: BaseTool):
    """Build the specialist that selects the best offer for one flight leg."""
    system_prompt = """
    You are a flight-search specialist. You are given exactly one one-way hop and
    you decide which flight is best for it.

    Call search_leg ONCE. It returns every distinct option for that leg in departure
    order, with price, times, duration, stops and carrier. Nothing in that list is
    ranked or recommended — the judgment is entirely yours.

    Weigh the tradeoffs:
    - Price matters most, but it is not the only thing. A cheaper fare is a bad deal
      if it costs the traveller hours of their day or lands at an unusable hour.
    - A departure before 06:00, or an arrival after 23:00, is a real cost. Prefer a
      civilised time unless the saving is substantial.
    - A non-stop beats a connection at a similar price. One long layover can be worth
      a large saving; two connections rarely are.
    - Watch for outliers. The cheapest row is sometimes a 20-hour multi-stop routing.

    If the request states the traveller's preferences, they outrank the guidance above
    — they are what this particular traveller wants. Apply them to the options you can
    see. If no option satisfies them, do not keep searching: choose the closest one and
    say plainly in your reasoning which preference you could not meet and why.

    If two or three options are genuinely close, you may call get_offer_details on
    those specific offers to compare baggage allowance and fare conditions before
    deciding. Use it sparingly; never walk the whole list.

    Then answer with the offer_id, price and currency copied exactly from the row you
    chose — all three are required, and price is the number alone without the currency
    code — plus one sentence naming the specific tradeoff that decided it. Never invent
    an offer_id. You do not plan itineraries and you do not book anything.
    """

    return create_agent(
        model=LLM,
        tools=[search_leg, get_offer_details],
        system_prompt=system_prompt,
        response_format=ToolStrategy(
            BestOffer,
            handle_errors=(
                "That didn't match the required format. Return the exact offer_id "
                "string from one of your search results."
            ),
        ),
        middleware=[
            # Backstops. The prompt should keep these unreachable; they exist so a
            # confused run degrades into a reported failure instead of a hang.
            ToolCallLimitMiddleware(
                tool_name="search_leg", run_limit=2, exit_behavior="continue"
            ),
            ToolCallLimitMiddleware(
                tool_name="get_offer_details", run_limit=3, exit_behavior="continue"
            ),
            ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ],
    )
