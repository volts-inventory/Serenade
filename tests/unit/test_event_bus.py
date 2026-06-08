"""Unit tests for EventBus -- verify in-process asyncio pub/sub."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from orchestrator.events.bus import EventBus


@pytest.mark.asyncio
async def test_publish_writes_entry():
    bus = EventBus()
    eng = uuid4()
    entry_id = await bus.publish(engagement_id=eng, kind="stage.start", payload={"stage": "recon"})
    assert entry_id != ""
    # History should contain the event
    topic = str(eng)
    assert len(bus._history[topic]) == 1
    event = bus._history[topic][0]
    assert event["kind"] == "stage.start"
    assert event["payload"]["stage"] == "recon"


@pytest.mark.asyncio
async def test_publish_returns_monotonic_ids():
    bus = EventBus()
    eng = uuid4()
    id1 = await bus.publish(engagement_id=eng, kind="a", payload={})
    id2 = await bus.publish(engagement_id=eng, kind="b", payload={})
    assert int(id1) < int(id2)


@pytest.mark.asyncio
async def test_subscribe_receives_live_events():
    """A subscriber started with from_id='$' receives events published
    after the subscription begins."""
    bus = EventBus()
    eng = uuid4()

    received: list[dict] = []

    async def _consumer():
        async for event in bus.subscribe(eng, from_id="$"):
            if event["kind"] == "_heartbeat":
                continue
            received.append(event)
            if len(received) >= 2:
                break

    task = asyncio.create_task(_consumer())
    # Give the consumer a moment to register
    await asyncio.sleep(0.01)

    await bus.publish(engagement_id=eng, kind="x", payload={"n": 1})
    await bus.publish(engagement_id=eng, kind="y", payload={"n": 2})

    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 2
    assert received[0]["kind"] == "x"
    assert received[1]["kind"] == "y"


@pytest.mark.asyncio
async def test_subscribe_replays_history():
    """A subscriber with from_id='0' replays all historical events before
    receiving live events."""
    bus = EventBus()
    eng = uuid4()

    # Publish two events BEFORE subscribing
    await bus.publish(engagement_id=eng, kind="old_a", payload={})
    await bus.publish(engagement_id=eng, kind="old_b", payload={})

    received: list[dict] = []

    async def _consumer():
        async for event in bus.subscribe(eng, from_id="0"):
            if event["kind"] == "_heartbeat":
                continue
            received.append(event)
            if len(received) >= 3:
                break

    task = asyncio.create_task(_consumer())
    await asyncio.sleep(0.01)

    # Publish one more AFTER subscribing
    await bus.publish(engagement_id=eng, kind="new_c", payload={})

    await asyncio.wait_for(task, timeout=2.0)

    assert len(received) == 3
    assert received[0]["kind"] == "old_a"
    assert received[1]["kind"] == "old_b"
    assert received[2]["kind"] == "new_c"


@pytest.mark.asyncio
async def test_subscribe_heartbeat_on_idle():
    """When no events arrive within idle_timeout_s, the subscriber
    yields a _heartbeat event."""
    bus = EventBus()
    eng = uuid4()

    async for event in bus.subscribe(eng, from_id="$", idle_timeout_s=0.05):
        assert event["kind"] == "_heartbeat"
        break  # got it, done


@pytest.mark.asyncio
async def test_publish_swallows_internal_error():
    """Bus failures must not block the agent loop."""
    bus = EventBus()
    # Sabotage the counter to trigger an unexpected error path
    bus._counter = None  # type: ignore[assignment]
    entry_id = await bus.publish(engagement_id=uuid4(), kind="x", payload={})
    assert entry_id == ""  # silent failure surface


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_event():
    """Fan-out: all subscribers on the same topic see the event."""
    bus = EventBus()
    eng = uuid4()

    results_a: list[dict] = []
    results_b: list[dict] = []

    async def _sub(out: list[dict]):
        async for event in bus.subscribe(eng, from_id="$"):
            if event["kind"] == "_heartbeat":
                continue
            out.append(event)
            break

    ta = asyncio.create_task(_sub(results_a))
    tb = asyncio.create_task(_sub(results_b))
    await asyncio.sleep(0.01)

    await bus.publish(engagement_id=eng, kind="shared", payload={"v": 42})

    await asyncio.wait_for(asyncio.gather(ta, tb), timeout=2.0)

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert results_a[0]["kind"] == "shared"
    assert results_b[0]["kind"] == "shared"


@pytest.mark.asyncio
async def test_concurrent_publishes_get_unique_ordered_ids():
    """async-01 regression: a burst of publishes scheduled together must
    each get a UNIQUE, monotonically-increasing id. The counter bump +
    history append run under the bus lock so two publishes can't read the
    same counter value and emit duplicate ids (which the abort/SSE
    consumers de-dup/order on)."""
    bus = EventBus()
    eng = uuid4()
    n = 200

    ids = await asyncio.gather(
        *(bus.publish(engagement_id=eng, kind="burst", payload={"i": i}) for i in range(n))
    )

    int_ids = [int(x) for x in ids]
    # Unique.
    assert len(set(int_ids)) == n, "duplicate event ids under concurrent publish"
    # Contiguous + monotonic.
    assert sorted(int_ids) == list(range(1, n + 1))
    # History order matches id order.
    hist_ids = [int(e["id"]) for e in bus._history[str(eng)]]
    assert hist_ids == sorted(hist_ids)
    assert len(hist_ids) == n


@pytest.mark.asyncio
async def test_subscribe_replay_delivers_each_event_exactly_once():
    """async-02 regression: when a publish lands in the window between the
    subscriber registering its queue and snapshotting history, the event
    must be delivered EXACTLY ONCE (not once via the snapshot and again via
    the live queue). We drive many interleaved publishes while a replay
    subscriber attaches and assert no id is seen twice and ordering holds."""
    bus = EventBus()
    eng = uuid4()

    # Seed some history so from_id="0" has something to replay.
    for i in range(5):
        await bus.publish(engagement_id=eng, kind="seed", payload={"i": i})

    seen_ids: list[int] = []
    total_expected = 5 + 20

    async def _consumer():
        async for event in bus.subscribe(eng, from_id="0"):
            if event["kind"] == "_heartbeat":
                continue
            seen_ids.append(int(event["id"]))
            if len(seen_ids) >= total_expected:
                break

    task = asyncio.create_task(_consumer())
    await asyncio.sleep(0)  # let the subscriber register + snapshot

    # Fire a burst concurrently to maximise the chance of landing in the
    # register/snapshot window.
    await asyncio.gather(
        *(bus.publish(engagement_id=eng, kind="live", payload={"i": i}) for i in range(20))
    )

    await asyncio.wait_for(task, timeout=3.0)

    # Each event seen exactly once.
    assert len(seen_ids) == len(set(seen_ids)), f"duplicate delivery: {seen_ids}"
    # Strictly increasing → in order, no reordering across replay/live seam.
    assert seen_ids == sorted(seen_ids)
    assert len(seen_ids) == total_expected


@pytest.mark.asyncio
async def test_heartbeat_carries_id():
    """async-07 regression: the idle-timeout heartbeat dict must include an
    ``id`` key (consumers track last-seen id for Last-Event-ID resume and
    would KeyError on a heartbeat without one)."""
    bus = EventBus()
    eng = uuid4()
    # Publish one event so the counter is non-zero.
    await bus.publish(engagement_id=eng, kind="x", payload={})

    async for event in bus.subscribe(eng, from_id="$", idle_timeout_s=0.05):
        assert event["kind"] == "_heartbeat"
        assert "id" in event, "heartbeat must carry an id for resume parity"
        # Non-empty because the counter advanced.
        assert event["id"] == "1"
        break


@pytest.mark.asyncio
async def test_aclose_clears_state():
    """aclose() should clean up topics and history."""
    bus = EventBus()
    eng = uuid4()
    await bus.publish(engagement_id=eng, kind="x", payload={})
    assert len(bus._history) > 0
    await bus.aclose()
    assert len(bus._history) == 0
    assert len(bus._topics) == 0
