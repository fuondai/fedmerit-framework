"""Canonical serialization and hashing for certificate fields."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("certificate fields must be finite")
        return {"float64": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Encode ``value`` deterministically without relying on object ordering."""
    return json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest(value: Any) -> str:
    """Return a SHA-256 hexadecimal digest of a canonical value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def merkle_root(leaves: list[Any]) -> str:
    """Return a deterministic binary Merkle root over ordered leaves."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(digest(leaf)) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def merkle_path(leaves: list[Any], index: int) -> tuple[tuple[str, bool], ...]:
    """Return ``(sibling_hash, sibling_is_left)`` steps for one ordered leaf."""
    if not leaves or isinstance(index, bool) or not 0 <= index < len(leaves):
        raise ValueError("Merkle path index is outside a non-empty leaf set")
    level = [bytes.fromhex(digest(leaf)) for leaf in leaves]
    position = index
    path: list[tuple[str, bool]] = []
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sibling_index = position - 1 if position % 2 else position + 1
        path.append((level[sibling_index].hex(), bool(position % 2)))
        level = [
            hashlib.sha256(level[offset] + level[offset + 1]).digest()
            for offset in range(0, len(level), 2)
        ]
        position //= 2
    return tuple(path)


def verify_merkle_path(
    leaf: Any,
    path: tuple[tuple[str, bool], ...],
    expected_root: str,
) -> bool:
    """Verify an ordered binary Merkle authentication path."""
    try:
        node = bytes.fromhex(digest(leaf))
        for sibling_hash, sibling_is_left in path:
            sibling = bytes.fromhex(sibling_hash)
            if len(sibling) != 32 or not isinstance(sibling_is_left, bool):
                return False
            node = (
                hashlib.sha256(sibling + node).digest()
                if sibling_is_left
                else hashlib.sha256(node + sibling).digest()
            )
        return node.hex() == expected_root
    except (TypeError, ValueError):
        return False
