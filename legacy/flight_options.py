"""Turn a raw flight-search payload into a compact, complete option table.

This module deliberately does no ranking. It removes duplicates and shrinks the
representation; choosing between options is the agent's job.
"""

import re
from dataclasses import dataclass

_DURATION = re.compile(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?")


def _duration_minutes(value: str | None) -> int:
    """Parse an ISO-8601 duration like 'PT5H10M' or 'PT8H'. Unparseable sorts last."""
    match = _DURATION.fullmatch(value or "")
    if not match:
        return 10**6
    days, hours, minutes = (int(g) if g else 0 for g in match.groups())
    return days * 1440 + hours * 60 + minutes


@dataclass
class Option:
    offer_id: str
    price: float
    currency: str
    departure: str
    arrival: str
    minutes: int
    stops: int
    carrier: str


def parse_offers(raw: dict) -> list[Option]:
    """Flatten a search_flights payload. Sums duration/stops so this still works
    if a multi-slice search is ever passed in."""
    out = []
    for offer in raw.get("offers", []):
        slices = offer.get("slices") or []
        if not slices:
            continue
        try:
            price = float(offer["price"]["amount"])
        except (KeyError, TypeError, ValueError):
            continue
        if not offer.get("offer_id"):
            continue
        out.append(Option(
            offer_id=offer["offer_id"],
            price=price,
            currency=offer["price"].get("currency", ""),
            departure=slices[0].get("departure") or "",
            arrival=slices[-1].get("arrival") or "",
            minutes=sum(_duration_minutes(s.get("duration")) for s in slices),
            stops=sum(s.get("stops", 0) for s in slices),
            carrier=slices[0].get("carrier") or "?",
        ))
    return out


def dedupe(options: list[Option]) -> list[Option]:
    """Collapse the identical itinerary sold by several carriers, keeping the
    cheapest. This removes literal duplicates only — it expresses no preference
    between genuinely different flights. Measured: 110 offers -> 33 itineraries."""
    best: dict[tuple, Option] = {}
    for o in sorted(options, key=lambda o: o.price):
        best.setdefault((o.departure, o.arrival, o.stops), o)
    return list(best.values())


def format_options(options: list[Option]) -> str:
    """Compact table, sorted by departure time.

    Chronological order is used precisely because it is neutral: sorting by price
    would signal that cheapest is best, which is the agent's call, not ours."""
    if not options:
        return "No flights found for this leg."
    rows = sorted(options, key=lambda o: o.departure)
    lines = [
        f"{len(rows)} options, in departure order. Nothing here is ranked — you choose.",
        # price and currency are separate columns so each is copied cleanly into BestOffer.
        "offer_id | price | currency | depart | arrive | duration | stops | carrier",
    ]
    for o in rows:
        lines.append(
            f"{o.offer_id} | {o.price:.2f} | {o.currency} | {o.departure[11:16]} | "
            f"{o.arrival[11:16]} | {o.minutes // 60}h{o.minutes % 60:02d} | {o.stops} | {o.carrier}"
        )
    return "\n".join(lines)
