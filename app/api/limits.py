"""Shared rate limiter.

One Limiter instance is created here so every router registers against the same
in-memory store. The backend is per-process, which suits a single-instance
deployment; behind multiple workers this needs a shared Redis store or each
process enforces its own separate budget.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

#: Client identity is the remote address. Behind a proxy this needs the
#: forwarded-for header instead, or every client collapses into one bucket.
limiter = Limiter(key_func=get_remote_address)

UPLOAD_LIMIT = "10/minute"
READ_LIMIT = "30/minute"
QUERY_LIMIT = "20/minute"
