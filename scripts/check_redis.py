"""
Verify Redis connectivity using the same REDIS_URL loading path as the FastAPI app.

Run from the bookQnA directory:
  python scripts/check_redis.py

Loads app/.env first (see app/main.py). If REDIS_URL is unset, prints what .env.example recommends.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"


def _load_env() -> None:
    load_dotenv(APP_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")


def _redis_url() -> str | None:
    url = os.getenv("REDIS_URL", "").strip()
    return url or None


def _describe_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.hostname or "?"
        port = p.port or 6379
        db = (p.path or "/").lstrip("/") or "0"
        return f"host={host!r} port={port} db_index={db}"
    except Exception:
        return "(could not parse URL)"


def main() -> int:
    _load_env()
    url = _redis_url()
    if not url:
        print("REDIS_URL is not set (empty after loading app/.env and project .env).")
        print("Example from .env.example: REDIS_URL=redis://localhost:6379/0")
        print("Docker Desktop: publish 6379 from the Redis container to the host (e.g. -p 6379:6379).")
        return 2

    print("Resolved REDIS_URL:", url.split("@")[-1] if "@" in url else url)
    print("Parsed:", _describe_url(url))

    try:
        import redis
    except ImportError:
        print("redis package not installed. Run: pip install redis")
        return 2

    client = redis.from_url(url, socket_connect_timeout=5, decode_responses=True)
    try:
        if client.ping():
            print("PING: OK")
        info = client.info("server")
        ver = info.get("redis_version", "?")
        print(f"SERVER redis_version: {ver}")
        test_key = "bookqna:check_redis:ping"
        client.setex(test_key, 10, "ok")
        v = client.get(test_key)
        print(f"SETEX/GET probe key={test_key!r} value={v!r}")
        client.delete(test_key)
        print("Redis is reachable from this Python process.")
        return 0
    except redis.exceptions.ConnectionError as e:
        print("Connection failed:", e)
        print()
        print("WinError 10061 / connection refused usually means:")
        print("  - No process is listening on that host:port on Windows (Redis not running or wrong port).")
        print("  - Docker: ensure Ports show 0.0.0.0:6379->6379/tcp (or 127.0.0.1:6379) for the Redis container.")
        print("  - If Redis listens only inside a Docker network, a host app must use the published host port,")
        print("    not the service name 'redis' (that name works only from other containers on the same network).")
        return 1
    except Exception as e:
        print("Unexpected error:", e)
        return 1
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
