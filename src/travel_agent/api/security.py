"""HTTP security headers for the browser client."""

import base64
import os
import re
from ipaddress import ip_address

from starlette.requests import Request

_HOSTNAME = re.compile(
    r"(?=.{1,253}\Z)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}\Z"
)


def client_ip(request: Request) -> str:
    """Read Railway's edge-set client IP without trusting forwarding chains."""
    if os.environ.get("RAILWAY_ENVIRONMENT_ID"):
        forwarded_ip = request.headers.get("x-real-ip")
        if forwarded_ip:
            try:
                return str(ip_address(forwarded_ip))
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def clerk_frontend_api(publishable_key: str | None) -> str | None:
    """Safely extract Clerk's Frontend API host from a publishable key."""
    if not publishable_key:
        return None
    parts = publishable_key.split("_", 2)
    if len(parts) != 3 or parts[0] != "pk":
        return None
    try:
        encoded_host = parts[2] + "=" * (-len(parts[2]) % 4)
        host = base64.urlsafe_b64decode(encoded_host).decode("ascii")
    except (UnicodeDecodeError, ValueError):
        return None
    host = host.rstrip("$")
    return host if _HOSTNAME.fullmatch(host) else None


def content_security_policy(publishable_key: str | None) -> str:
    """Return a least-privilege CSP compatible with the bundled Clerk client."""
    clerk_host = clerk_frontend_api(publishable_key)
    clerk_origin = f"https://{clerk_host}" if clerk_host else None
    script_sources = [
        "'self'",
        "https://challenges.cloudflare.com",
        "https://*.protect.clerk.com",
    ]
    connect_sources = ["'self'", "https://*.protect.clerk.com:*"]
    if clerk_origin:
        script_sources.append(clerk_origin)
        connect_sources.append(clerk_origin)

    directives = {
        "default-src": ["'self'"],
        "base-uri": ["'self'"],
        "object-src": ["'none'"],
        "frame-ancestors": ["'none'"],
        "form-action": ["'self'"],
        "script-src": script_sources,
        # Clerk injects component styles at runtime. This does not authorize scripts.
        "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
        "font-src": ["'self'", "https://fonts.gstatic.com"],
        "img-src": ["'self'", "data:", "https://img.clerk.com"],
        "connect-src": connect_sources,
        "worker-src": ["'self'", "blob:"],
        "frame-src": [
            "'self'",
            "https://challenges.cloudflare.com",
            "https://*.protect.clerk.com",
        ],
    }
    return "; ".join(
        f"{name} {' '.join(values)}" for name, values in directives.items()
    )
