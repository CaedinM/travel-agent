"""Backward-compatible ASGI entry point.

The application now lives in :mod:`travel_agent.api.app`, but keeping this
module lets existing ``uvicorn api:app --app-dir src`` commands continue to
work during the package-layout migration.
"""

from travel_agent.api.app import app

__all__ = ["app"]
