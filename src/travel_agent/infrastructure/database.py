"""PostgreSQL storage for application-owned user records."""

import os
import uuid

import psycopg


_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    clerk_user_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _database_url() -> str:
    """Return the required Render Postgres connection string."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return database_url


def initialize_database() -> None:
    """Create application tables before the API starts accepting requests."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_USERS_TABLE)


def create_user(clerk_user_id: str) -> uuid.UUID | None:
    """Create a user record once, tolerating Clerk's webhook retries.

    Returns the generated application UUID for a new row and ``None`` when the
    Clerk user was already stored.
    """
    user_id = uuid.uuid4()
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (user_id, clerk_user_id)
                VALUES (%s, %s)
                ON CONFLICT (clerk_user_id) DO NOTHING
                RETURNING user_id
                """,
                (user_id, clerk_user_id),
            )
            row = cursor.fetchone()
    return row[0] if row else None
