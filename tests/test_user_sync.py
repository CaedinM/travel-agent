import asyncio
import uuid

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
