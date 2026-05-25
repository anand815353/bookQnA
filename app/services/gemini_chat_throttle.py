"""Process-wide spacing between Gemini chat API calls (reduces free-tier 429 bursts)."""

from __future__ import annotations

import threading
import time

from app.settings import GEMINI_CHAT_MIN_INTERVAL_SECONDS

_lock = threading.Lock()
_last_slot_monotonic: float | None = None


def wait_gemini_chat_slot() -> None:
    """Sleep if needed so consecutive invokes are at least GEMINI_CHAT_MIN_INTERVAL_SECONDS apart."""
    interval = GEMINI_CHAT_MIN_INTERVAL_SECONDS
    if interval <= 0:
        return
    global _last_slot_monotonic
    with _lock:
        now = time.monotonic()
        if _last_slot_monotonic is not None:
            elapsed = now - _last_slot_monotonic
            if elapsed < interval:
                time.sleep(interval - elapsed)
        _last_slot_monotonic = time.monotonic()
