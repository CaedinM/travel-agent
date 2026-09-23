"""Verified inbound webhooks from Clerk."""

import asyncio
import os

from fastapi import HTTPException, Request
from svix.webhooks import Webhook, WebhookVerificationError

from travel_agent.infrastructure.database import create_user


async def handle_clerk_webhook(request: Request) -> dict[str, str]:
    """Store a newly created Clerk user after verifying the Svix signature."""
    signing_secret = os.environ.get("CLERK_WEBHOOK_SIGNING_SECRET")
    if not signing_secret:
        raise HTTPException(status_code=503, detail="Clerk webhooks are not configured.")

    payload = await request.body()
    headers = {
        header: value
        for header, value in request.headers.items()
        if header.lower() in {"svix-id", "svix-timestamp", "svix-signature"}
    }
    try:
        event = Webhook(signing_secret).verify(payload, headers)
    except WebhookVerificationError as error:
        raise HTTPException(status_code=400, detail="Invalid Clerk webhook signature.") from error

    if event.get("type") != "user.created":
        return {"status": "ignored"}

    clerk_user_id = event.get("data", {}).get("id")
    if not isinstance(clerk_user_id, str) or not clerk_user_id:
        raise HTTPException(status_code=400, detail="Clerk webhook has no user ID.")

    user_id = await asyncio.to_thread(create_user, clerk_user_id)
    return {"status": "created" if user_id else "already_exists"}
