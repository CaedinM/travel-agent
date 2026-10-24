import base64

from travel_agent.api.security import clerk_frontend_api, content_security_policy


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
