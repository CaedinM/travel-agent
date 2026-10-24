"""PostgreSQL storage for application-owned user records."""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

import psycopg
from psycopg.types.json import Jsonb


logger = logging.getLogger(__name__)


_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    clerk_user_id TEXT NOT NULL UNIQUE,
    is_paying_customer BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_ADD_USERS_IS_PAYING_CUSTOMER = """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS is_paying_customer BOOLEAN NOT NULL DEFAULT FALSE
"""

_CREATE_MONTHLY_TOKEN_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS monthly_token_usage (
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    usage_month DATE NOT NULL,
    tokens_used BIGINT NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
    tokens_reserved BIGINT NOT NULL DEFAULT 0 CHECK (tokens_reserved >= 0),
    PRIMARY KEY (user_id, usage_month)
)
"""

_CREATE_THREADS_TABLE = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_TRIPS_TABLE = """
CREATE TABLE IF NOT EXISTS trips (
    trip_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'Untitled trip',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    state_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_ADD_TRIP_TO_THREADS = """
ALTER TABLE threads
ADD COLUMN IF NOT EXISTS trip_id UUID REFERENCES trips(trip_id) ON DELETE SET NULL
"""

_CREATE_TRIPS_USER_INDEX = """
CREATE INDEX IF NOT EXISTS trips_user_updated_idx ON trips (user_id, updated_at DESC)
"""

_CREATE_THREADS_TRIP_INDEX = """
CREATE INDEX IF NOT EXISTS threads_trip_idx ON threads (trip_id)
"""

_BACKFILL_TRIPS = """
INSERT INTO trips (trip_id, user_id, title)
SELECT threads.thread_id, threads.user_id, 'Imported trip'
FROM threads
WHERE threads.trip_id IS NULL
ON CONFLICT (trip_id) DO NOTHING
"""

_LINK_LEGACY_THREADS = """
UPDATE threads SET trip_id = thread_id WHERE trip_id IS NULL
"""


@dataclass(frozen=True)
class TokenReservation:
    """A portion of a user's monthly budget held while an AI request runs."""

    usage_month: date
    tokens: int


def current_usage_month() -> date:
    """Return the UTC calendar month used for monthly token entitlements."""
    now = datetime.now(timezone.utc)
    return date(now.year, now.month, 1)


def _database_url() -> str:
    """Return the required Postgres connection string."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return database_url


def initialize_database() -> None:
    """Create application tables before the API starts accepting requests."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_USERS_TABLE)
            cursor.execute(_ADD_USERS_IS_PAYING_CUSTOMER)
            cursor.execute(_CREATE_MONTHLY_TOKEN_USAGE_TABLE)
            cursor.execute(_CREATE_THREADS_TABLE)
            cursor.execute(_CREATE_TRIPS_TABLE)
            cursor.execute(_ADD_TRIP_TO_THREADS)
            cursor.execute(_BACKFILL_TRIPS)
            cursor.execute(_LINK_LEGACY_THREADS)
            cursor.execute(_CREATE_TRIPS_USER_INDEX)
            cursor.execute(_CREATE_THREADS_TRIP_INDEX)


def check_database_connection() -> None:
    """Check database readiness with a short timeout and no application writes."""
    with psycopg.connect(_database_url(), connect_timeout=3) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '3s'")
            cursor.execute("SELECT 1")


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
    created_user_id = row[0] if row else None
    if created_user_id:
        logger.info(
            "database.user_created clerk_user_id=%s user_id=%s",
            clerk_user_id,
            created_user_id,
        )
    else:
        logger.info("database.user_exists clerk_user_id=%s", clerk_user_id)
    return created_user_id


def reserve_monthly_tokens(
    clerk_user_id: str,
    reservation_tokens: int,
    free_monthly_limit: int,
    paid_monthly_limit: int,
) -> TokenReservation | None:
    """Reserve token budget before a model call without exceeding the monthly cap."""
    if min(reservation_tokens, free_monthly_limit, paid_monthly_limit) < 1:
        raise ValueError("Token reservation and limits must be positive")

    usage_month = current_usage_month()
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (user_id, clerk_user_id)
                VALUES (%s, %s)
                ON CONFLICT (clerk_user_id) DO UPDATE
                SET clerk_user_id = EXCLUDED.clerk_user_id
                RETURNING user_id, is_paying_customer
                """,
                (uuid.uuid4(), clerk_user_id),
            )
            user_id, is_paying_customer = cursor.fetchone()
            token_limit = paid_monthly_limit if is_paying_customer else free_monthly_limit
            cursor.execute(
                """
                INSERT INTO monthly_token_usage (user_id, usage_month)
                VALUES (%s, %s)
                ON CONFLICT (user_id, usage_month) DO NOTHING
                """,
                (user_id, usage_month),
            )
            cursor.execute(
                """
                UPDATE monthly_token_usage
                SET tokens_reserved = tokens_reserved + %s
                WHERE user_id = %s AND usage_month = %s
                    AND tokens_used + tokens_reserved + %s <= %s
                RETURNING tokens_reserved
                """,
                (reservation_tokens, user_id, usage_month, reservation_tokens, token_limit),
            )
            reserved = cursor.fetchone()
    if reserved is None:
        return None
    return TokenReservation(usage_month=usage_month, tokens=reservation_tokens)


def settle_monthly_tokens(
    clerk_user_id: str, reservation: TokenReservation, actual_tokens: int
) -> None:
    """Replace a reservation with the token usage reported by the completed call."""
    if actual_tokens < 0:
        raise ValueError("actual_tokens must not be negative")

    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE monthly_token_usage
                SET tokens_reserved = GREATEST(tokens_reserved - %s, 0),
                    tokens_used = tokens_used + %s
                FROM users
                WHERE monthly_token_usage.user_id = users.user_id
                    AND users.clerk_user_id = %s
                    AND monthly_token_usage.usage_month = %s
                """,
                (reservation.tokens, actual_tokens, clerk_user_id, reservation.usage_month),
            )


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
    created = row is not None
    if created:
        logger.info(
            "database.thread_created clerk_user_id=%s thread_id=%s",
            clerk_user_id,
            thread_uuid,
        )
    else:
        logger.info(
            "database.thread_exists clerk_user_id=%s thread_id=%s",
            clerk_user_id,
            thread_uuid,
        )
    return created


def create_trip_and_thread(clerk_user_id: str, thread_id: str) -> uuid.UUID:
    """Create a blank durable trip and its first chat thread atomically."""
    trip_id = uuid.uuid4()
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
                ), created_trip AS (
                    INSERT INTO trips (trip_id, user_id)
                    SELECT %s, user_id FROM app_user
                    RETURNING trip_id
                )
                INSERT INTO threads (thread_id, user_id, trip_id)
                SELECT %s, app_user.user_id, created_trip.trip_id
                FROM app_user CROSS JOIN created_trip
                RETURNING trip_id
                """,
                (uuid.uuid4(), clerk_user_id, trip_id, thread_uuid),
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Could not create trip thread")
    logger.info("database.trip_created clerk_user_id=%s trip_id=%s thread_id=%s", clerk_user_id, trip_id, thread_uuid)
    return row[0]


def create_resumed_thread(clerk_user_id: str, trip_id: str, thread_id: str) -> bool:
    """Create a new chat thread only when the caller owns the active trip."""
    trip_uuid = uuid.UUID(trip_id)
    thread_uuid = uuid.UUID(thread_id)
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO threads (thread_id, user_id, trip_id)
                SELECT %s, users.user_id, trips.trip_id
                FROM users JOIN trips ON trips.user_id = users.user_id
                WHERE users.clerk_user_id = %s AND trips.trip_id = %s
                    AND trips.status = 'active'
                ON CONFLICT (thread_id) DO NOTHING
                RETURNING thread_id
                """,
                (thread_uuid, clerk_user_id, trip_uuid),
            )
            row = cursor.fetchone()
    if row:
        logger.info("database.trip_resumed clerk_user_id=%s trip_id=%s thread_id=%s", clerk_user_id, trip_uuid, thread_uuid)
    return row is not None


def trip_for_thread(clerk_user_id: str, thread_id: str) -> uuid.UUID | None:
    """Resolve a thread's trip after enforcing ownership."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT threads.trip_id FROM threads
                JOIN users ON users.user_id = threads.user_id
                WHERE users.clerk_user_id = %s AND threads.thread_id = %s
                """,
                (clerk_user_id, uuid.UUID(thread_id)),
            )
            row = cursor.fetchone()
    return row[0] if row and row[0] else None


def load_trip_state(clerk_user_id: str, trip_id: str) -> dict | None:
    """Return the current JSON snapshot only to the trip owner."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT trips.state FROM trips JOIN users ON users.user_id = trips.user_id
                WHERE users.clerk_user_id = %s AND trips.trip_id = %s
                    AND trips.status = 'active'
                """,
                (clerk_user_id, uuid.UUID(trip_id)),
            )
            row = cursor.fetchone()
    return dict(row[0]) if row else None


def save_trip_state(clerk_user_id: str, trip_id: str, state: dict, title: str) -> int | None:
    """Persist a validated planning snapshot and return its new version."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE trips SET state = %s, title = %s, state_version = state_version + 1,
                    updated_at = NOW()
                FROM users
                WHERE trips.user_id = users.user_id AND users.clerk_user_id = %s
                    AND trips.trip_id = %s AND trips.status = 'active'
                RETURNING trips.state_version
                """,
                (Jsonb(state), title, clerk_user_id, uuid.UUID(trip_id)),
            )
            row = cursor.fetchone()
    if row:
        logger.info("database.trip_saved clerk_user_id=%s trip_id=%s state_version=%s", clerk_user_id, trip_id, row[0])
        return row[0]
    return None


def list_trips(clerk_user_id: str) -> list[dict]:
    """List a user's saved trips without exposing their raw state."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT trips.trip_id, trips.title, trips.status, trips.state_version,
                    trips.created_at, trips.updated_at
                FROM trips JOIN users ON users.user_id = trips.user_id
                WHERE users.clerk_user_id = %s
                ORDER BY trips.updated_at DESC
                """,
                (clerk_user_id,),
            )
            rows = cursor.fetchall()
    return [
        {"trip_id": row[0], "title": row[1], "status": row[2], "state_version": row[3], "created_at": row[4], "updated_at": row[5]}
        for row in rows
    ]


def set_trip_status(clerk_user_id: str, trip_id: str, status: str) -> bool:
    """Archive or reactivate an owned trip."""
    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE trips SET status = %s, updated_at = NOW()
                FROM users WHERE trips.user_id = users.user_id
                    AND users.clerk_user_id = %s AND trips.trip_id = %s
                RETURNING trips.trip_id
                """,
                (status, clerk_user_id, uuid.UUID(trip_id)),
            )
            row = cursor.fetchone()
    return row is not None
