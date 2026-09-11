"""Unit tests for the canonical run-event content projection.

Covers the projection rules and the determinism guarantee documented in
``deerflow/runtime/events/canonical.py``. The cross-backend conformance case
lives in ``test_run_event_stream_contract.py``.
"""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import pathlib
import uuid
from dataclasses import dataclass
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.runtime.events.canonical import CANONICAL_TYPE_KEY, canonical_event_content
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.journal import RunJournal


class _Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class _Payload:
    name: str
    count: int


class _Opaque:
    """An object whose only sane representation is its type."""


def _strict_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def test_scalars_and_finite_floats_pass_through_unchanged():
    for value in (None, True, False, 0, -3, 1.5, "", "text", "中文"):
        assert canonical_event_content(value) == value


@pytest.mark.parametrize("value,expected", [("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf"))])
def test_non_finite_floats_become_markers(value, expected):
    projected = canonical_event_content(expected)

    assert projected == {CANONICAL_TYPE_KEY: "float", "value": value}
    # ``json.dumps`` would otherwise emit bare NaN/Infinity, which is not JSON.
    _strict_dump(projected)


def test_nested_containers_project_recursively():
    projected = canonical_event_content({"a": [1, {"b": (2, 3)}], "c": {"d": None}})

    assert projected == {"a": [1, {"b": [2, 3]}], "c": {"d": None}}


def test_non_string_mapping_keys_are_stringified():
    projected = canonical_event_content({1: "one", (1, 2): "tuple", None: "none"})

    assert projected == {"1": "one", "(1, 2)": "tuple", "None": "none"}
    _strict_dump(projected)


def test_sets_project_to_a_sorted_list_independent_of_iteration_order():
    first = canonical_event_content({"tags": {"beta", "alpha", "gamma"}})
    second = canonical_event_content({"tags": {"gamma", "alpha", "beta"}})

    assert first == {"tags": ["alpha", "beta", "gamma"]}
    assert _strict_dump(first) == _strict_dump(second)


def test_bytes_project_to_a_base64_marker():
    projected = canonical_event_content(b"\x00\xffbinary")

    assert projected == {CANONICAL_TYPE_KEY: "bytes", "base64": "AP9iaW5hcnk="}
    _strict_dump(projected)


def test_temporal_values_project_to_stable_text():
    moment = dt.datetime(2026, 9, 11, 3, 4, 5, tzinfo=dt.UTC)

    assert canonical_event_content(moment) == "2026-09-11T03:04:05+00:00"
    assert canonical_event_content(dt.date(2026, 9, 11)) == "2026-09-11"
    assert canonical_event_content(dt.time(3, 4, 5)) == "03:04:05"
    assert canonical_event_content(dt.timedelta(seconds=90)) == {CANONICAL_TYPE_KEY: "timedelta", "seconds": 90.0}


def test_scalar_like_objects_project_to_text():
    identifier = uuid.UUID("00000000-0000-0000-0000-0000000000ab")

    assert canonical_event_content(identifier) == "00000000-0000-0000-0000-0000000000ab"
    assert canonical_event_content(decimal.Decimal("1.50")) == "1.50"
    assert canonical_event_content(pathlib.PurePosixPath("/mnt/user-data/outputs/report.md")) == "/mnt/user-data/outputs/report.md"


def test_enums_project_to_their_value():
    assert canonical_event_content({"colour": _Colour.RED}) == {"colour": "red"}


def test_langchain_messages_project_to_their_model_dump():
    projected = canonical_event_content({"messages": [AIMessage(content="final answer", id="final-message")]})

    message = projected["messages"][0]
    assert isinstance(message, dict)
    assert message["content"] == "final answer"
    assert message["id"] == "final-message"
    assert message["type"] == "ai"
    _strict_dump(projected)


def test_other_message_types_project_without_losing_their_identity():
    projected = canonical_event_content([HumanMessage(content="hi", id="h1"), ToolMessage(content="ok", tool_call_id="call-1", id="t1")])

    assert [message["type"] for message in projected] == ["human", "tool"]
    assert projected[1]["tool_call_id"] == "call-1"


def test_dataclasses_project_to_their_field_mapping():
    assert canonical_event_content(_Payload(name="report", count=2)) == {"name": "report", "count": 2}


def test_unrepresentable_objects_project_to_a_type_marker_without_an_address():
    projected = canonical_event_content({"value": _Opaque()})

    assert projected == {"value": {CANONICAL_TYPE_KEY: f"{__name__}._Opaque"}}
    dumped = _strict_dump(projected)
    assert "0x" not in dumped


def test_reference_cycles_project_to_a_marker():
    cyclic: dict = {"name": "root"}
    cyclic["self"] = cyclic

    projected = canonical_event_content(cyclic)

    assert projected == {"name": "root", "self": {CANONICAL_TYPE_KEY: "circular-reference"}}
    _strict_dump(projected)


def test_projection_is_json_safe_and_reproducible():
    payload = {
        "messages": [AIMessage(content=[{"type": "text", "text": "hi"}], id="m1")],
        "started_at": dt.datetime(2026, 9, 11, tzinfo=dt.UTC),
        "durations": {dt.timedelta(seconds=1)},
        "blob": b"\x01\x02",
        "opaque": _Opaque(),
        "ratio": 0.1,
    }

    first = canonical_event_content(payload)
    second = canonical_event_content(payload)

    assert _strict_dump(first) == _strict_dump(second)
    # A JSON round trip through either JSON backend must return the same value.
    assert json.loads(_strict_dump(first)) == first


@pytest.mark.anyio
async def test_run_journal_projects_root_outputs_before_persisting():
    store = MemoryRunEventStore()
    journal = RunJournal("run-canonical", "thread-canonical", store, flush_threshold=100)

    journal.on_chain_end(
        {"messages": [AIMessage(content="final answer", id="final-message")], "artifacts": [pathlib.PurePosixPath("/mnt/user-data/outputs/a.md")]},
        run_id=uuid4(),
        parent_run_id=None,
    )
    await journal.flush()

    events = await store.list_events("thread-canonical", "run-canonical", event_types=["run.end"])
    assert len(events) == 1
    content = events[0]["content"]

    assert isinstance(content["messages"][0], dict)
    assert content["messages"][0]["content"] == "final answer"
    assert content["artifacts"] == ["/mnt/user-data/outputs/a.md"]
    _strict_dump(content)
