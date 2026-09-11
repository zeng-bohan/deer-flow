"""Deterministic JSON-safe projection for persisted run-event content.

The ``RunEventStore`` backends disagreed about nested values that are not
directly JSON serializable. ``MemoryRunEventStore`` retained the original
Python container, while ``JsonlRunEventStore`` and ``DbRunEventStore`` coerced
those values through ``json.dumps(..., default=str)``. The same
``run.end.content`` therefore read back as three different values depending on
which backend served it, and ``str()`` of an object without a stable
``__str__`` embeds a memory address, so even the two JSON backends were not
reproducible across processes.

``canonical_event_content()`` is the single projection a producer applies
before a value enters any store, so every backend persists the same structure
and restores it identically.

Projection rules
----------------

===============================  ==================================================
Input                            Projection
===============================  ==================================================
``None``, ``bool``, ``int``,     unchanged
``str``, finite ``float``
non-finite ``float``             ``{"__deerflow_type__": "float", "value": ...}``
``bytes`` / ``bytearray`` /      ``{"__deerflow_type__": "bytes", "base64": ...}``
``memoryview``
``datetime`` / ``date`` /        ISO-8601 string
``time``
``timedelta``                    ``{"__deerflow_type__": "timedelta", "seconds": ...}``
``UUID`` / ``Decimal`` /         ``str(value)``
``PurePath``
``enum.Enum``                    projection of ``value.value``
mappings                         ``dict``; non-string keys are stringified
``set`` / ``frozenset``          list sorted by canonical JSON text
other sequences                  list
objects exposing ``model_dump``  projection of the dump (LangChain messages,
                                 pydantic models)
dataclasses                      projection of ``dataclasses.asdict``
anything else                    ``{"__deerflow_type__": "module.QualName"}``
reference cycle                  ``{"__deerflow_type__": "circular-reference"}``
===============================  ==================================================

The result depends only on the value: never on ``repr()``, memory addresses,
the iteration order of an unordered container (sets are sorted), or the current
time. Two runs in two processes project the same input to the same output.

``__deerflow_type__`` is reserved for the runtime. A genuine mapping that
already carries that key is indistinguishable from a marker, which is why the
key is namespaced rather than generic.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import decimal
import enum
import json
import math
import pathlib
import uuid
from collections.abc import Mapping, Sequence, Set
from typing import Any, Final

CANONICAL_TYPE_KEY: Final[str] = "__deerflow_type__"

#: Sentinel returned by :func:`_model_dump` when a value has no usable dump, so
#: a legitimate dump of ``None`` is never confused with "no dump available".
_NO_DUMP: Final[object] = object()


def canonical_event_content(value: Any) -> Any:
    """Return the deterministic JSON-safe projection of *value*."""
    return _project(value, set())


def _marker(kind: str, **fields: Any) -> dict[str, Any]:
    return {CANONICAL_TYPE_KEY: kind, **fields}


def _guard(value: Any, active: set[int]) -> bool:
    """Track *value* as an in-progress container; ``False`` means a cycle."""
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    return True


def _project_sequence(value: Sequence, active: set[int]) -> Any:
    if not _guard(value, active):
        return _marker("circular-reference")
    try:
        return [_project(item, active) for item in value]
    finally:
        active.discard(id(value))


def _project_mapping(value: Mapping, active: set[int]) -> Any:
    if not _guard(value, active):
        return _marker("circular-reference")
    try:
        return {key if isinstance(key, str) else str(key): _project(item, active) for key, item in value.items()}
    finally:
        active.discard(id(value))


def _project_set(value: Set, active: set[int]) -> Any:
    if not _guard(value, active):
        return _marker("circular-reference")
    try:
        projected = [_project(item, active) for item in value]
    finally:
        active.discard(id(value))
    # An unordered container is only deterministic once its projected members
    # are ordered by their canonical text.
    return sorted(projected, key=_canonical_text)


def _canonical_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        return _NO_DUMP
    try:
        return dump()
    except Exception:  # noqa: BLE001 - a model that cannot dump falls back to the type marker
        return _NO_DUMP


def _project(value: Any, active: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # ``json.dumps`` emits bare NaN/Infinity, which is not valid JSON and
        # does not survive an equality round-trip (NaN != NaN).
        return value if math.isfinite(value) else _marker("float", value=repr(value))
    if isinstance(value, enum.Enum):
        return _project(value.value, active)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _marker("bytes", base64=base64.b64encode(bytes(value)).decode("ascii"))
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return _marker("timedelta", seconds=value.total_seconds())
    if isinstance(value, (uuid.UUID, decimal.Decimal, pathlib.PurePath)):
        return str(value)
    if isinstance(value, Mapping):
        return _project_mapping(value, active)
    if isinstance(value, Set):
        return _project_set(value, active)
    if isinstance(value, (list, tuple, range)) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview))):
        return _project_sequence(value, active)
    dumped = _model_dump(value)
    if dumped is not _NO_DUMP:
        return _project(dumped, active)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _project(dataclasses.asdict(value), active)
    # No structured form is available. Recording the type is deterministic;
    # ``str(value)`` would embed a memory address for most objects.
    return _marker(f"{type(value).__module__}.{type(value).__qualname__}")
