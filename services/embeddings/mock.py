"""Deterministic mock embeddings (feature hashing).

Not a whole-string hash: each token is hashed to a (dimension, sign) bucket
and the buckets are summed, so texts sharing tokens get high cosine
similarity. This makes semantic-search assertions meaningful in tests —
mock search for "inventory health" really does rank the inventory health
entry first — while staying fully deterministic and dependency-free.

IMPORTANT: mcp-server/services/embeddings.py carries a byte-identical copy
(query vectors must match index vectors). Golden-value tests in both test
suites pin the output; change both copies together.
"""

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def mock_embed(text: str, dim: int) -> list[float]:
    """Map text to a deterministic L2-normalized vector of length dim."""
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(c * c for c in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [c / norm for c in vec]
