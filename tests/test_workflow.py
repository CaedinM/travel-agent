from travel_agent.core.state import TripPhase
from travel_agent.core.workflow import next_phase


def test_next_phase_only_advances_when_each_phase_prerequisite_is_complete():
    assert next_phase(
        {
            "phase": "intake",
            "origin": "SFO",
            "departure_date": "2027-05-10",
            "return_date": "2027-05-17",
        }
    ) is TripPhase.RESEARCH

    assert next_phase(
        {
            "phase": "research",
            "legs": [{"origin": "SFO", "destination": "", "departure_date": "2027-05-10"}],
        }
    ) is TripPhase.RESEARCH

    assert next_phase(
        {
            "phase": "research",
            "legs": [{"origin": "SFO", "destination": "CDG", "departure_date": "2027-05-10"}],
        }
    ) is TripPhase.FLIGHTS

    assert next_phase(
        {
            "phase": "flights",
            "flights_confirmed": True,
            "legs": [{"origin": "SFO", "destination": "CDG", "offer": {}}],
        }
    ) is TripPhase.FLIGHTS

    assert next_phase(
        {
            "phase": "flights",
            "flights_confirmed": True,
            "legs": [{"origin": "SFO", "destination": "CDG", "offer": {"offer_id": "offer-1"}}],
        }
    ) is TripPhase.ACCOMMODATIONS
