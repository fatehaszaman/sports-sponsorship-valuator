"""
data/cache_db.py — SQLite Deterministic Cache and ETL Validation Layer.

Business summary
----------------
API calls are expensive (rate-limited, latency, cost). This module caches all
external API responses in a local SQLite database with deterministic keys.
A cache hit returns the stored result instantly; a miss fetches and stores.

The ETL validation layer checks every fetched payload against a schema before
storing — catching malformed API responses before they corrupt downstream models.

Developer notes
---------------
Cache key design: SHA-256 hash of (endpoint + parameters), so identical
requests always resolve to the same key regardless of call order.

SQLite was chosen over Redis or file-based caches because:
  1. Zero external dependency (stdlib sqlite3)
  2. ACID-compliant (no partial writes)
  3. Queryable for debugging (SELECT * FROM cache WHERE ...)
  4. Built-in TTL support via expires_at column

Validation uses a lightweight schema-checking approach rather than Pydantic
(to avoid dependency in the data layer) — field presence + type checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_DB_PATH   = Path("~/.sports_sponsorship_valuator/cache.db").expanduser()
DEFAULT_TTL_SECS  = 86_400          # 24 hours
SCHEMA_VERSION    = "v1"


# ── Validation Schemas ────────────────────────────────────────────────────────

_VALIDATION_SCHEMAS: dict[str, dict] = {
    "newsapi_article": {
        "required_fields": ["title", "url", "publishedAt", "source"],
        "field_types":     {"title": str, "url": str, "publishedAt": str},
        "numeric_ranges":  {},
    },
    "twitter_tweet": {
        "required_fields": ["id", "text", "created_at"],
        "field_types":     {"id": str, "text": str, "created_at": str},
        "numeric_ranges":  {},
    },
    "newsapi_response": {
        "required_fields": ["status", "totalResults", "articles"],
        "field_types":     {"status": str, "totalResults": int, "articles": list},
        "numeric_ranges":  {"totalResults": (0, 1_000_000)},
    },
    "twitter_search_response": {
        "required_fields": ["data", "meta"],
        "field_types":     {"data": list, "meta": dict},
        "numeric_ranges":  {},
    },
}


class ValidationError(Exception):
    """Raised when a fetched payload fails schema validation."""
    pass


def _make_cache_key(endpoint: str, params: dict) -> str:
    """
    Deterministic SHA-256 cache key from endpoint + sorted parameters.

    Sorting params ensures that {"q": "pepsi", "lang": "en"} and
    {"lang": "en", "q": "pepsi"} produce the same cache key.
    """
    canonical = f"{SCHEMA_VERSION}:{endpoint}:" + json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_payload(payload: Any, schema_name: str) -> list[str]:
    """
    Validate a fetched payload against a named schema.

    Returns a list of validation error messages (empty = valid).

    Checks performed:
      1. Required fields present
      2. Field types correct
      3. Numeric fields within expected range
    """
    schema = _VALIDATION_SCHEMAS.get(schema_name)
    if schema is None:
        return []  # unknown schema → skip validation

    errors: list[str] = []

    if not isinstance(payload, dict):
        return [f"Payload is {type(payload).__name__}, expected dict"]

    # Required fields
    for field in schema.get("required_fields", []):
        if field not in payload:
            errors.append(f"Missing required field: '{field}'")

    # Field types
    for field, expected_type in schema.get("field_types", {}).items():
        if field in payload and not isinstance(payload[field], expected_type):
            errors.append(
                f"Field '{field}': expected {expected_type.__name__}, "
                f"got {type(payload[field]).__name__}"
            )

    # Numeric ranges
    for field, (lo, hi) in schema.get("numeric_ranges", {}).items():
        if field in payload:
            val = payload[field]
            if isinstance(val, (int, float)) and not (lo <= val <= hi):
                errors.append(f"Field '{field}': value {val} outside range [{lo}, {hi}]")

    return errors


class APICache:
    """
    SQLite-backed deterministic cache for external API responses.

    Business summary
    ----------------
    Wraps any external API call with a cache layer:
      - On cache hit (within TTL): returns stored response instantly.
      - On cache miss: caller fetches, validates, and stores via set().
      - On validation failure: raises ValidationError before storing.

    Developer notes
    ---------------
    Thread-safety: SQLite's WAL mode is enabled for concurrent readers.
    Each connection uses check_same_thread=False for use in async contexts.

    Usage
    -----
    >>> cache = APICache()
    >>> key = cache.make_key("newsapi/everything", {"q": "Emirates Arsenal"})
    >>> cached = cache.get(key)
    >>> if cached is None:
    ...     data = requests.get(...).json()
    ...     cache.set(key, data, schema="newsapi_response", ttl=3600)
    """

    def __init__(self, db_path: Optional[Path] = None, default_ttl: int = DEFAULT_TTL_SECS) -> None:
        self.db_path    = db_path or DEFAULT_DB_PATH
        self.default_ttl = default_ttl
        self._ensure_db()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _ensure_db(self) -> None:
        """Create the cache database and table if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key      TEXT PRIMARY KEY,
                    endpoint       TEXT NOT NULL,
                    params_json    TEXT NOT NULL,
                    payload_json   TEXT NOT NULL,
                    schema_name    TEXT,
                    fetched_at     REAL NOT NULL,
                    expires_at     REAL NOT NULL,
                    hit_count      INTEGER DEFAULT 0,
                    last_hit_at    REAL,
                    is_valid       INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_log (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key      TEXT NOT NULL,
                    schema_name    TEXT,
                    errors_json    TEXT,
                    logged_at      REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON api_cache(expires_at)
            """)

    def _conn(self) -> sqlite3.Connection:
        """Open a new SQLite connection to the cache database."""
        return sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(endpoint: str, params: dict) -> str:
        """Compute deterministic cache key. Wrapper around module-level function."""
        return _make_cache_key(endpoint, params)

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve a cached response if it exists and has not expired.

        Parameters
        ----------
        cache_key : str
            SHA-256 hash key from make_key().

        Returns
        -------
        Any
            Deserialized payload, or None on miss/expired.
        """
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload_json, expires_at FROM api_cache "
                "WHERE cache_key = ? AND is_valid = 1",
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        payload_json, expires_at = row
        if now > expires_at:
            logger.debug("Cache entry expired: %s", cache_key[:16])
            return None

        # Update hit stats
        with self._conn() as conn:
            conn.execute(
                "UPDATE api_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                "WHERE cache_key = ?",
                (now, cache_key),
            )

        return json.loads(payload_json)

    def set(
        self,
        cache_key: str,
        payload: Any,
        endpoint: str = "",
        params: Optional[dict] = None,
        schema: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a fetched payload in the cache after validation.

        Validation
        ----------
        If `schema` is provided, the payload is validated against that schema
        before storage. On validation failure, a ValidationError is raised
        AND the failure is logged to the validation_log table with the error list.

        Parameters
        ----------
        cache_key : str
        payload : Any
            JSON-serializable response object.
        endpoint : str
            API endpoint (for audit log).
        params : dict, optional
            Request parameters (for audit log).
        schema : str, optional
            Schema name to validate against (e.g. "newsapi_response").
        ttl : int, optional
            Cache lifetime in seconds. Defaults to self.default_ttl.

        Raises
        ------
        ValidationError
            If schema validation fails.
        """
        now        = time.time()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        # ── ETL Validation ────────────────────────────────────────────────────
        errors: list[str] = []
        if schema:
            errors = validate_payload(payload, schema)
            if errors:
                # Log the failure
                with self._conn() as conn:
                    conn.execute(
                        "INSERT INTO validation_log (cache_key, schema_name, errors_json, logged_at) "
                        "VALUES (?, ?, ?, ?)",
                        (cache_key, schema, json.dumps(errors), now),
                    )
                raise ValidationError(
                    f"Payload failed schema '{schema}' validation:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        payload_json = json.dumps(payload, ensure_ascii=False)
        params_json  = json.dumps(params or {}, sort_keys=True)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO api_cache
                    (cache_key, endpoint, params_json, payload_json,
                     schema_name, fetched_at, expires_at, hit_count, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1)
                """,
                (cache_key, endpoint, params_json, payload_json,
                 schema, now, expires_at),
            )

    def invalidate(self, cache_key: str) -> bool:
        """Manually invalidate a cache entry (marks is_valid=0)."""
        with self._conn() as conn:
            n = conn.execute(
                "UPDATE api_cache SET is_valid = 0 WHERE cache_key = ?",
                (cache_key,),
            ).rowcount
        return n > 0

    def purge_expired(self) -> int:
        """Delete all expired entries. Returns count deleted."""
        now = time.time()
        with self._conn() as conn:
            n = conn.execute(
                "DELETE FROM api_cache WHERE expires_at < ?", (now,)
            ).rowcount
        logger.info("Purged %d expired cache entries.", n)
        return n

    def stats(self) -> dict:
        """Return cache statistics for monitoring."""
        now = time.time()
        with self._conn() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
            valid    = conn.execute("SELECT COUNT(*) FROM api_cache WHERE is_valid=1").fetchone()[0]
            expired  = conn.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at < ?", (now,)).fetchone()[0]
            total_hits = conn.execute("SELECT SUM(hit_count) FROM api_cache").fetchone()[0] or 0
            val_errors = conn.execute("SELECT COUNT(*) FROM validation_log").fetchone()[0]

        return {
            "total_entries":    total,
            "valid_entries":    valid,
            "expired_entries":  expired,
            "total_cache_hits": total_hits,
            "validation_errors": val_errors,
            "db_path":          str(self.db_path),
        }

    def reconcile(self, expected_keys: list[str]) -> dict:
        """
        ETL reconciliation: verify that all expected cache keys are present and valid.

        Business summary
        ----------------
        After a batch fetch run, this method checks that every expected API
        response has been successfully cached. Missing or invalid entries are
        flagged for re-fetch. This is the "reconciliation" step in the ETL pipeline.

        Parameters
        ----------
        expected_keys : list[str]
            List of cache keys that should be in the cache.

        Returns
        -------
        dict
            {present: list, missing: list, expired: list, invalid: list}
        """
        now = time.time()
        present, missing, expired_keys, invalid = [], [], [], []

        with self._conn() as conn:
            for key in expected_keys:
                row = conn.execute(
                    "SELECT expires_at, is_valid FROM api_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    missing.append(key)
                elif not row[1]:
                    invalid.append(key)
                elif row[0] < now:
                    expired_keys.append(key)
                else:
                    present.append(key)

        return {
            "present":  present,
            "missing":  missing,
            "expired":  expired_keys,
            "invalid":  invalid,
            "coverage_pct": round(len(present) / len(expected_keys) * 100, 1) if expected_keys else 100.0,
        }
