import pytest
from pydantic import ValidationError

from travel_agent.core.responses import DestinationSuggestion, PlannerResponse
from travel_agent.core.state import TripSnapshot


def test_planner_response_rejects_more_than_three_destination_suggestions():
    suggestion = DestinationSuggestion(
        country="Thailand", city="Bangkok", description="Food and nightlife."
    )

    with pytest.raises(ValidationError, match="at most 3 items"):
        PlannerResponse(response="Here are some ideas.", destination_suggestions=[suggestion] * 4)


def test_trip_snapshot_rejects_non_concrete_dates():
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        TripSnapshot(departure_date="Winter 2027")


@pytest.mark.parametrize("phase", [None, "intake", "flights", "accommodations"])
def test_destination_cards_are_withheld_outside_research(phase):
    reply = PlannerResponse(
        response="Which airport are you departing from?",
        destination_suggestions=[DestinationSuggestion(
            country="Thailand", city="Bangkok", description="Food and nightlife."
        )],
    )
    state = dict(phase=phase, origin="LAX", departure_date="2027-01-10",
                 return_date="2027-01-20", travelers=1)

    filtered = reply.for_trip_state(state)

    assert filtered.destination_suggestions == []
    assert filtered.response == reply.response
    assert len(reply.destination_suggestions) == 1


@pytest.mark.parametrize("missing", ["origin", "departure_date", "return_date", "travelers"])
def test_destination_cards_require_saved_intake_details_even_in_research(missing):
    reply = PlannerResponse(
        response="One more detail first.",
        destination_suggestions=[DestinationSuggestion(
            country="Thailand", city="Bangkok", description="Food and nightlife."
        )],
    )
    state = dict(phase="research", origin="LAX", departure_date="2027-01-10",
                 return_date="2027-01-20", travelers=1)
    state[missing] = None

    assert reply.for_trip_state(state).destination_suggestions == []


def test_destination_cards_are_preserved_when_trip_is_ready_for_research():
    reply = PlannerResponse(
        response="Would Bangkok suit you?",
        destination_suggestions=[DestinationSuggestion(
            country="Thailand", city="Bangkok", description="Food and nightlife."
        )],
    )
    state = dict(phase="research", origin="LAX", departure_date="2027-01-10",
                 return_date="2027-01-20", travelers=1)

    assert reply.for_trip_state(state).destination_suggestions == reply.destination_suggestions
