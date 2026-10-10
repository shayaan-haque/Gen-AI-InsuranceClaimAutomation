"""Shared request pacing for answer generation and evaluation API calls."""

import asyncio
import os
import threading
import time


# Conservative default for the observed 1,000-output / 7,000-input TPM limits.
# Hooks run for every HTTP attempt, including SDK/structured-output retries.
REQUEST_INTERVAL_SECONDS = float(os.getenv("EVAL_REQUEST_INTERVAL_SECONDS", "65"))
_lock = threading.Lock()
_next_request_at = 0.0


def _reserve_delay():
    global _next_request_at
    with _lock:
        now = time.monotonic()
        scheduled = max(now, _next_request_at)
        _next_request_at = scheduled + REQUEST_INTERVAL_SECONDS
        return scheduled - now


def pace_request(request):
    time.sleep(_reserve_delay())


async def pace_async_request(request):
    await asyncio.sleep(_reserve_delay())
