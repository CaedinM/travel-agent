"""Readiness checks for the API's required persistence services."""

import asyncio
import logging

import psycopg
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from travel_agent.infrastructure.database import check_database_connection

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", include_in_schema=False)
async def health(request: Request) -> JSONResponse:
    """Report readiness without exposing service addresses or credentials."""
    try:
        await asyncio.wait_for(request.app.state.redis.ping(), timeout=3)
        await asyncio.to_thread(check_database_connection)
    except (RedisError, psycopg.Error, TimeoutError):
        # Provider exception messages can contain connection details.
        logger.warning("health.persistence_unavailable")
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ok"})
