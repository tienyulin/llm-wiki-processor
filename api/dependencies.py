"""Request-level dependencies shared by routers."""

import os
import secrets

from fastapi import Header, HTTPException


async def require_api_key(x_api_key: str = Header(default="")):
    """Reject calls without a valid X-API-Key when auth is enabled.

    Read at request time (not import time) so tests can toggle the env var.
    """
    expected = os.getenv("PROCESSOR_API_KEY")
    if not expected:
        return  # auth disabled (dev mode)
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
