import base64

import pytest
from starlette.requests import Request

from travel_agent.api.security import client_ip, clerk_frontend_api, content_security_policy


def _request(headers: dict[str, str], client=("192.0.2.1", 12345)) -> Request:
    return Request({
        "type": "http", "client": client,
        "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
    })


@pytest.mark.parametrize("address", ["203.0.113.7", "2001:db8::7"])
def test_client_ip_uses_railway_edge_header(monkeypatch, address):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "railway-test")
    assert client_ip(_request({"X-Real-IP": address, "X-Forwarded-For": "spoofed"})) == address


@pytest.mark.parametrize("address", ["invalid", "203.0.113.1, 203.0.113.2", ""])
def test_client_ip_falls_back_for_invalid_or_missing_edge_ip(monkeypatch, address):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "railway-test")
    assert client_ip(_request({"X-Real-IP": address})) == "192.0.2.1"


def test_client_ip_ignores_headers_outside_railway(monkeypatch):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT_ID", raising=False)
    assert client_ip(_request({"X-Real-IP": "203.0.113.7"})) == "192.0.2.1"
    assert client_ip(_request({}, client=None)) == "unknown"


def _publishable_key(host: str) -> str:
    encoded = base64.urlsafe_b64encode(f"{host}$".encode()).decode().rstrip("=")
    return f"pk_test_{encoded}"


def test_clerk_frontend_api_reads_a_valid_publishable_key() -> None:
    assert (
        clerk_frontend_api(_publishable_key("clerk.example.com")) == "clerk.example.com"
    )


def test_clerk_frontend_api_rejects_invalid_values() -> None:
    assert clerk_frontend_api("not-a-publishable-key") is None
    assert clerk_frontend_api(_publishable_key("bad host")) is None


def test_content_security_policy_allows_only_required_script_sources() -> None:
    policy = content_security_policy(_publishable_key("clerk.example.com"))

    assert (
        "script-src 'self' https://challenges.cloudflare.com https://*.protect.clerk.com https://clerk.example.com"
        in policy
    )
    assert (
        "connect-src 'self' https://*.protect.clerk.com:* https://clerk.example.com"
        in policy
    )
    assert "object-src 'none'" in policy
    assert "script-src 'self' 'unsafe-eval'" not in policy
    assert "https://unpkg.com" not in policy
    assert "https://esm.sh" not in policy
