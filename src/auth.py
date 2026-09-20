"""Clerk authentication helpers for the FastAPI application."""

import os
from typing import Annotated

from clerk_backend_api import AuthenticateRequestOptions, authenticate_request
from clerk_backend_api.security.types import RequestState
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


http_bearer = HTTPBearer(auto_error=False)


def _authorized_parties() -> list[str]:
    """Read the comma-separated browser origins allowed to use this API."""
    value = os.environ.get("CLERK_AUTHORIZED_PARTIES", "http://localhost:8000")
    return [party.strip() for party in value.split(",") if party.strip()]


def require_auth(
    request: Request,
    _credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(http_bearer)
    ] = None,
) -> RequestState:
    """Verify a Clerk session token and return its signed request state."""
    secret_key = os.environ.get("CLERK_SECRET_KEY")
    if not secret_key:
        raise HTTPException(status_code=503, detail="Clerk authentication is not configured.")

    state = authenticate_request(
        request,
        AuthenticateRequestOptions(
            secret_key=secret_key,
            jwt_key=os.environ.get("CLERK_JWT_KEY"),
            authorized_parties=_authorized_parties(),
            accepts_token=["session_token"],
        ),
    )
    if not state.is_signed_in:
        reason = state.reason.name if state.reason else "unauthorized"
        raise HTTPException(
            status_code=401,
            detail=reason,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return state


def current_user_id(state: Annotated[RequestState, Depends(require_auth)]) -> str:
    """Extract the Clerk user ID from an authenticated session token."""
    return state.payload["sub"]
