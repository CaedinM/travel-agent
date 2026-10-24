from unittest.mock import AsyncMock, Mock

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from travel_agent.api import health


@pytest.mark.parametrize("failure", [None, "redis", "postgres", "timeout"])
def test_health_checks_persistence_without_auth_or_error_details(monkeypatch, failure):
    database_check = Mock()
    redis_client = Mock(ping=AsyncMock(return_value=True))
    if failure == "postgres":
        database_check.side_effect = psycopg.OperationalError("secret-db-url")
    elif failure == "redis":
        redis_client.ping.side_effect = RedisConnectionError("secret-redis-url")
    elif failure == "timeout":
        redis_client.ping.side_effect = TimeoutError("secret-redis-url")
    monkeypatch.setattr(health, "check_database_connection", database_check)
    app = FastAPI()
    app.state.redis = redis_client
    app.include_router(health.router)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == (503 if failure else 200)
    assert response.json() == {"status": "unavailable" if failure else "ok"}
    redis_client.ping.assert_awaited_once()
    if failure in {"redis", "timeout"}:
        database_check.assert_not_called()
    else:
        database_check.assert_called_once_with()
