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

_CREATE_THREADS_TABLE = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
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
            cursor.execute(_CREATE_THREADS_TABLE)


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


def create_thread(clerk_user_id: str, thread_id: str) -> bool:
    """Record ownership when a new chat thread is created.

    No conversation or trace content is persisted here. The user upsert makes
    this safe if a user starts chatting before Clerk's webhook delivery arrives.
    """
    thread_uuid = uuid.UUID(thread_id)
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH app_user AS (
                    INSERT INTO users (user_id, clerk_user_id)
                    VALUES (%s, %s)
                    ON CONFLICT (clerk_user_id) DO UPDATE
                    SET clerk_user_id = EXCLUDED.clerk_user_id
                    RETURNING user_id
                )
                INSERT INTO threads (thread_id, user_id)
                SELECT %s, user_id FROM app_user
                ON CONFLICT (thread_id) DO NOTHING
                RETURNING thread_id
                """,
                (uuid.uuid4(), clerk_user_id, thread_uuid),
            )
            row = cursor.fetchone()
    return row is not None
