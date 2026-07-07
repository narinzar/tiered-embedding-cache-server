"""Deterministic hashing embedder.

This turns a text key into a fixed-dimension float32 vector without any model
download or network access. Tokens are hashed into buckets of a fixed-width
vector (the hashing trick), then the vector is L2 normalized.

This is a stand-in for a real embedding model. It is deterministic, offline,
and dependency-light so the cache server and benchmark run anywhere. In a real
deployment you would swap `embed` for a call into e.g. sentence-transformers
(SentenceTransformer(...).encode) or an API embedding endpoint; the cache and
server code do not care where the vectors come from as long as the output is a
fixed-length float32 numpy array.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

DEFAULT_DIM = 128

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercase word/number tokens. Falls back to the whole string if empty."""
    toks = _TOKEN_RE.findall(text.lower())
    return toks if toks else [text.lower()]


def _bucket_and_sign(token: str, dim: int) -> tuple[int, float]:
    """Map a token to a (bucket_index, sign) pair via a stable hash.

    Using a signed hashing trick keeps the collisions from all pushing the
    vector in the same direction, which preserves more information per bucket.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    bucket = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return bucket, sign


def embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Return a deterministic L2-normalized float32 embedding for `text`.

    Args:
        text: the key to embed.
        dim: output dimension.

    Returns:
        A 1-D float32 numpy array of length `dim`. If the input has no usable
        tokens the returned vector is all zeros (norm 0 stays 0).
    """
    if dim <= 0:
        raise ValueError("dim must be positive")

    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokens(text):
        bucket, sign = _bucket_and_sign(tok, dim)
        vec[bucket] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec
