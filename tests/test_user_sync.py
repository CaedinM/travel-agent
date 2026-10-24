import asyncio
import base64
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from fastapi import HTTPException
from starlette.requests import Request

from travel_agent.api import webhooks
from travel_agent.infrastructure import database


def test_create_user_generates_uuid_and_uses_idempotent_insert(monkeypatch):
    captured = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query, parameters):
            captured["query"] = query
            captured["parameters"] = parameters

        def fetchone(self):
            return (captured["parameters"][0],)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def cursor(self):
            return Cursor()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(database.psycopg, "connect", lambda _: Connection())

    user_id = database.create_user("user_123")

    assert isinstance(user_id, uuid.UUID)
    assert captured["parameters"][1] == "user_123"
    assert "ON CONFLICT (clerk_user_id) DO NOTHING" in captured["query"]


def test_create_thread_links_thread_to_application_user(monkeypatch):
    captured = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query, parameters):
            captured["query"] = query
            captured["parameters"] = parameters

        def fetchone(self):
            return (captured["parameters"][2],)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def cursor(self):
            return Cursor()

    thread_id = str(uuid.uuid4())
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(database.psycopg, "connect", lambda _: Connection())

    assert database.create_thread("user_123", thread_id)
    assert captured["parameters"][1] == "user_123"
    assert captured["parameters"][2] == uuid.UUID(thread_id)
    assert "INSERT INTO threads (thread_id, user_id)" in captured["query"]


def test_reserve_monthly_tokens_uses_the_paid_customer_allowance(monkeypatch):
    queries = []
    user_id = uuid.uuid4()

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query, parameters):
            queries.append((query, parameters))

        def fetchone(self):
            if len(queries) == 1:
                return (user_id, True)
            if len(queries) == 3:
                return (8_000,)
            return None

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def cursor(self):
            return Cursor()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(database.psycopg, "connect", lambda _: Connection())
    month = database.date(2026, 9, 1)
    monkeypatch.setattr(database, "current_usage_month", lambda: month)

    reservation = database.reserve_monthly_tokens(
        "user_123", reservation_tokens=8_000, free_monthly_limit=75_000,
        paid_monthly_limit=750_000,
    )

    assert reservation == database.TokenReservation(usage_month=month, tokens=8_000)
    assert queries[2][1][-1] == 750_000
    assert "tokens_reserved + %s" in queries[2][0]


def test_settle_monthly_tokens_records_actual_usage(monkeypatch):
    captured = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query, parameters):
            captured["query"] = query
            captured["parameters"] = parameters

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def cursor(self):
            return Cursor()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(database.psycopg, "connect", lambda _: Connection())
    reservation = database.TokenReservation(database.date(2026, 9, 1), 8_000)

    database.settle_monthly_tokens("user_123", reservation, actual_tokens=1_234)

    assert captured["parameters"] == (8_000, 1_234, "user_123", database.date(2026, 9, 1))
    assert "tokens_used = tokens_used + %s" in captured["query"]


def _request() -> Request:
    async def receive():
        return {"type": "http.request", "body": b'{"test": true}', "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [
                (b"svix-id", b"msg_123"),
                (b"svix-timestamp", b"123"),
                (b"svix-signature", b"v1,test"),
            ],
        },
        receive,
    )


def test_user_created_webhook_creates_application_user(monkeypatch):
    captured = {}

    class FakeWebhook:
        def __init__(self, secret):
            captured["secret"] = secret

        def verify(self, payload, headers):
            captured["headers"] = headers
            assert payload == b'{"test": true}'
            return {"type": "user.created", "data": {"id": "user_123"}}

    monkeypatch.setenv("CLERK_WEBHOOK_SIGNING_SECRET", "whsec_test")
    monkeypatch.setattr(webhooks, "Webhook", FakeWebhook)
    monkeypatch.setattr(webhooks, "create_user", lambda clerk_user_id: uuid.UUID(int=1))

    result = asyncio.run(webhooks.handle_clerk_webhook(_request()))

    assert result == {"status": "created"}
    assert captured["secret"] == "whsec_test"
    assert captured["headers"]["svix-id"] == "msg_123"


def test_webhook_requires_signing_secret(monkeypatch):
    monkeypatch.delenv("CLERK_WEBHOOK_SIGNING_SECRET", raising=False)

    try:
        asyncio.run(webhooks.handle_clerk_webhook(_request()))
    except HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected an HTTPException")


@pytest.mark.parametrize("signature", ["v1,invalid", "no-comma", "v1,AAAA"])
def test_webhook_rejects_malformed_signatures_without_writing_users(monkeypatch, signature):
    monkeypatch.setenv("CLERK_WEBHOOK_SIGNING_SECRET", "whsec_" + base64.b64encode(bytes(32)).decode())
    create_user = Mock()
    monkeypatch.setattr(webhooks, "create_user", create_user)
    request = _request()
    request.scope["headers"] = [
        (b"svix-id", b"msg_123"),
        (b"svix-timestamp", str(int(datetime.now(timezone.utc).timestamp())).encode()),
        (b"svix-signature", signature.encode()),
    ]
    with pytest.raises(HTTPException) as error:
        asyncio.run(webhooks.handle_clerk_webhook(request))
    assert error.value.status_code == 400
    create_user.assert_not_called()
