"""Capacity-aware periodic scheduling for one tile body.

High-level
----------
A tile body is one iteration of a tiled FX scope. Its physical events are
compute operations and data transfers. Modulo scheduling lets iteration ``i``
start at ``i * II``, where ``II`` is the initiation interval, so successive
iterations overlap in time.

For a candidate ``II`` the scheduler must answer two questions:

1. Timing: can every event fit on its resources without violating
   dependencies?
2. Capacity: if the body repeats every ``II`` seconds, does its resident data
   fit in every finite memory?

Capacity is checked by resolving each buffer's live interval against the
candidate placement, folding those intervals onto ``II``, and measuring the
peak concurrent bytes. A buffer that spans a whole body is therefore counted
more than once when successive iterations overlap.

The public entry point, :func:`schedule_modulo_event_body`, performs:

    1. Infer the body's single scope path from node metadata.
    2. Collect compute, transfer, and zero-duration storage events.
    3. Collect internal dependencies between those events.
    4. Extract anchored storage specs from the storage events.
    5. Compute a resource lower bound on ``II``.
    6. Find the smallest timing-only ``II``.
    7. Probe one full body for a structurally oversized buffer.
    8. Search for the smallest capacity-feasible ``II``.
    9. If no interval fits, classify and raise ``MemoryCapacityError``.
   10. Return the placement, including whether capacity forced ``II``.

All times are seconds. The interval search works in integer nanoseconds so
doubling and bisection are exact.

Helper map
----------
``_search_periodic_interval``
    Grows and bisects the interval, calling the placement check below.
``_schedule_periodic``
    Tries one interval: placement, dependencies, recurrences, capacity.
``_first_periodic_slot``
    Finds a free slot in the periodic resource calendars.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil
from types import MappingProxyType

import networkx as nx

from dataflow_mapper.scheduling.model import (
    EventInterval,
    MemoryCapacityError,
    TilingScope,
    capacity_violations,
    periodic_capacity_peak,
    resolve_storage_windows,
)

_TIME_QUANTUM_SECONDS = 1e-9
_EPSILON = 1e-15


@dataclass(frozen=True)
class EventModuloPlacement:
    """Result of scheduling one tile body modulo its initiation interval.

    ``capacity`` is binding when the interval is larger than the timing-only
    minimum; ``capacity_bound`` records that fact for diagnostics and tests.
    """

    scope_path: tuple[TilingScope, ...]
    initiation_interval_seconds: float
    body_finish_seconds: float
    starts: Mapping[str, float]
    capacity_bound: bool = False

    def __post_init__(self) -> None:
        """Freeze placement data and validate the summarized tile body."""
        if (
            not self.scope_path
            or self.initiation_interval_seconds <= 0
            or self.body_finish_seconds < 0
        ):
            raise ValueError("invalid FX modulo placement")
        object.__setattr__(self, "starts", MappingProxyType(dict(self.starts)))


@dataclass(frozen=True)
class _PeriodicEvent:
    """One normalized compute or transfer event used by the periodic placer."""

    node_id: str
    resource_ids: tuple[str, ...]
    duration_seconds: float


def _search_periodic_interval(
    events: tuple[_PeriodicEvent, ...],
    dependencies: Mapping[str, tuple[str, ...]],
    recurrences: tuple[tuple[str, str], ...],
    lower_quanta: int,
    storage_specs: Iterable[
        tuple[
            str,
            str,
            int,
            tuple[str | None, str],
            tuple[str | None, str],
        ]
    ] = (),
    memory_capacities: Mapping[str, int] = MappingProxyType({}),
) -> tuple[int, dict[str, float] | None]:
    """Find the smallest feasible interval by exponential search and bisection.

    Steps:
        1. Try the resource lower bound first.
        2. Double the interval until a placement succeeds, up to 64 attempts.
        3. If no attempt succeeds, return the largest tried interval and
           ``None``.
        4. Bisect between the last failure and the first success to find the
           smallest feasible interval.

    Args:
        events: One iteration's periodic events with their resources and durations.
        dependencies: Intra-iteration ``event -> predecessors`` edges.
        recurrences: Loop-carried ``(producer, consumer)`` edges across iterations.
        lower_quanta: Lower bound on the interval, in time quanta.
        storage_specs: Anchored residency specs; empty disables the capacity check.
        memory_capacities: Finite bytes per memory id; empty disables the check.

    Returns:
        ``(upper_quanta, starts)`` where ``upper_quanta`` is the smallest feasible
        interval in quanta (or the largest tried when infeasible) and ``starts`` is
        ``event_id -> start_seconds``, or ``None`` when no tried interval works.

    Example:
        A timing-only search returns the smallest feasible interval::

            quanta, starts = _search_periodic_interval(events, deps, (), 1)
    """
    # Step 1: Try the lower bound before growing the interval.
    upper_quanta = lower_quanta
    starts = _schedule_periodic(
        events,
        dependencies,
        recurrences,
        upper_quanta * _TIME_QUANTUM_SECONDS,
        storage_specs,
        memory_capacities,
    )
    # Step 2: Double until a placement fits or the search budget is exhausted.
    for _ in range(64):
        if starts is not None:
            break
        upper_quanta *= 2
        starts = _schedule_periodic(
            events,
            dependencies,
            recurrences,
            upper_quanta * _TIME_QUANTUM_SECONDS,
            storage_specs,
            memory_capacities,
        )
    # Step 3: No tried interval fits; report the largest one tried.
    if starts is None:
        return upper_quanta, None
    # Step 4: Bisect between the last failure and the feasible upper bound.
    failed_quanta = lower_quanta - 1
    while failed_quanta + 1 < upper_quanta:
        middle = (failed_quanta + upper_quanta) // 2
        candidate = _schedule_periodic(
            events,
            dependencies,
            recurrences,
            middle * _TIME_QUANTUM_SECONDS,
            storage_specs,
            memory_capacities,
        )
        # Infeasible raises the floor; feasible lowers the ceiling and is kept.
        if candidate is None:
            failed_quanta = middle
        else:
            upper_quanta = middle
            starts = candidate
    return upper_quanta, starts


def schedule_modulo_event_body(
    body_graph: nx.DiGraph,
    memory_capacities: Mapping[str, int],
) -> EventModuloPlacement:
    """Schedule one scoped tile body at the smallest capacity-feasible interval.

    The caller supplies a graph containing only this body's events, so the
    scheduler does not need to filter a larger multi-scope graph. Storage nodes
    are included as zero-duration pass-through events and carry the anchored
    residency specs used for capacity checks.

    Steps:
        1. Infer the body's single scope path from node metadata.
        2. Collect compute, transfer, and zero-duration storage events.
        3. Collect internal dependencies between those events.
        4. Extract anchored storage specs from the storage events.
        5. Compute the resource lower bound on the interval.
        6. Find the smallest timing-only interval.
        7. Probe one full body for a buffer that cannot fit at any interval.
        8. Search for the smallest interval that also satisfies capacity.
        9. If no interval fits, classify the capacity failure and raise.
       10. Return the placement, including whether capacity forced the interval.

    Args:
        body_graph: Scoped event subgraph for one tile body, including compute,
            transfer, and pass-through storage nodes.
        memory_capacities: Finite capacity in bytes per memory id; memories
            absent from the mapping are unconstrained.

    Returns:
        The tile body's modulo placement (initiation interval, body finish, and
        per-event starts), or a structured ``MemoryCapacityError`` when no
        interval can fit.

    Example:
        >>> schedule_modulo_event_body(body, {"gpu": 40_000_000_000})
    """
    # Step 1: Infer the body's scope. The caller has already restricted the
    # graph to one body, so every node must agree on the same scope path.
    scoped = {
        event_id: data for event_id, data in body_graph.nodes(data=True)
    }
    scopes = {tuple(data["scopes"]) for data in scoped.values()}
    if len(scopes) != 1:
        raise ValueError("a modulo body must contain exactly one scope")
    scope_path = scopes.pop()
    if not scope_path:
        raise ValueError("an FX modulo body requires a non-empty tiling scope path")
    # Step 2: Normalize nodes into periodic events. Storage nodes are included
    # with zero duration and no resources so they participate in dependencies,
    # but never consume scheduling capacity.
    events = tuple(
        _PeriodicEvent(
            event_id,
            ()
            if data.get("kind") == "storage"
            else tuple(data.get("resources") or (data["resource_id"],)),
            0.0
            if data.get("kind") == "storage"
            else float(data["duration"]),
        )
        for event_id, data in sorted(scoped.items())
        if data.get("kind") == "storage" or float(data["duration"]) > 0
    )
    if not events:
        raise ValueError("FX modulo tile body has no positive-duration events")
    # Step 3: Keep only dependencies between events in this body. External
    # predecessor timing belongs to the enclosing scope.
    event_ids = {item.node_id for item in events}
    dependencies = {
        event_id: tuple(
            dependency_id
            for dependency_id in body_graph.predecessors(event_id)
            if dependency_id in event_ids
        )
        for event_id in event_ids
    }
    # Step 4: Storage nodes already carry their anchored residency specs.
    # Capacity folding resolves those anchors against each candidate placement.
    storage_specs = tuple(
        spec
        for _, data in body_graph.nodes(data=True)
        if data.get("kind") == "storage"
        for spec in data.get("anchors", ())
    )
    # Recurrence lowering is not implemented yet; the FX caller currently has
    # no loop-carried producer/consumer edges to supply.
    recurrences: tuple[tuple[str, str], ...] = ()
    # Step 5: A resource cannot be reused faster than its total busy time in
    # one body, so the busiest resource gives a lower bound on the interval.
    demand: dict[str, float] = defaultdict(float)
    for item in events:
        for resource_id in item.resource_ids:
            demand[resource_id] += item.duration_seconds
    lower_quanta = max(
        1, ceil(max(demand.values(), default=0.0) / _TIME_QUANTUM_SECONDS)
    )
    # Step 6: Find the smallest interval that satisfies timing alone. This is
    # the optimum if capacity is unconstrained, and the comparison point for
    # deciding whether capacity forced a larger interval later.
    timing_quanta, timing_starts = _search_periodic_interval(
        events, dependencies, recurrences, lower_quanta
    )
    if timing_starts is None:
        raise ValueError("FX modulo event placement did not converge")
    # Step 7: Probe one full body. Once the interval is at least one whole body,
    # successive iterations no longer overlap. If a single buffer still exceeds
    # capacity there, no interval can help and the failure is structural.
    if storage_specs and memory_capacities:
        # Step 7a: Find the full-body ceiling. The timing-only interval may be
        # shorter than the body, so place again at the body finish if needed.
        timing_finish = max(
            timing_starts[event.node_id] + event.duration_seconds
            for event in events
        )
        ceiling = max(
            timing_quanta * _TIME_QUANTUM_SECONDS,
            timing_finish,
        )
        ceiling_starts = timing_starts
        if ceiling > timing_quanta * _TIME_QUANTUM_SECONDS + _EPSILON:
            probe = _schedule_periodic(
                events, dependencies, recurrences, ceiling
            )
            if probe is None:
                raise ValueError("FX modulo event placement did not converge")
            ceiling_starts = probe
        # Step 7b: Resolve each anchored storage window against that placement.
        event_times = {
            item.node_id: (
                ceiling_starts[item.node_id],
                ceiling_starts[item.node_id] + item.duration_seconds,
            )
            for item in events
        }
        resolved = resolve_storage_windows(storage_specs, event_times, ceiling)
        windows = tuple(
            (memory_id, size_bytes, reserve, release)
            for _, memory_id, size_bytes, reserve, release in resolved
        )
        # Step 7c: Only a buffer larger than the whole memory is structural.
        # If every individual buffer fits, the interval search may still
        # separate overlapping windows across iterations.
        for memory_id, peak, capacity in capacity_violations(
            periodic_capacity_peak(windows), memory_capacities
        ):
            if not any(
                size_bytes > capacity
                for _, memory, size_bytes, *_ in resolved
                if memory == memory_id
            ):
                continue
            contributors = tuple(
                sorted(
                    (
                        (value_id, size_bytes)
                        for value_id, memory, size_bytes, *_ in resolved
                        if memory == memory_id
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            raise MemoryCapacityError(
                f"tile scope {scope_path!r} needs {peak} bytes on memory "
                f"{memory_id!r} but its capacity is {capacity}; reduce the tile size",
                scope_path=scope_path,
                memory_id=memory_id,
                peak_bytes=peak,
                capacity_bytes=capacity,
                required_scale=peak / capacity,
                contributors=contributors,
                kind="structural",
            )
    # Step 8: Search again, now folding storage residency onto each candidate
    # interval. A smaller interval overlaps more iterations and may exceed
    # capacity even though the timing-only interval was feasible.
    capacity_quanta, starts = _search_periodic_interval(
        events,
        dependencies,
        recurrences,
        lower_quanta,
        storage_specs,
        memory_capacities,
    )
    # Step 9: No interval satisfied timing and capacity together. Re-run
    # timing-only at the largest tried interval to determine whether the body
    # itself is over capacity or the search simply did not converge.
    if starts is None:
        # Step 9a: Re-place timing-only at the largest interval the capacity
        # search tried, so capacity can be measured without folding.
        probe = _schedule_periodic(
            events,
            dependencies,
            recurrences,
            capacity_quanta * _TIME_QUANTUM_SECONDS,
        )
        if probe is None:
            raise ValueError("FX modulo event placement did not converge")
        event_times = {
            item.node_id: (
                probe[item.node_id],
                probe[item.node_id] + item.duration_seconds,
            )
            for item in events
        }
        # Step 9b: Resolve residency against the timing-only placement.
        body_finish = max(finish for _, finish in event_times.values())
        resolved = resolve_storage_windows(storage_specs, event_times, body_finish)
        windows = tuple(
            (memory_id, size_bytes, reserve, release)
            for _, memory_id, size_bytes, reserve, release in resolved
        )
        # Step 9c: Classify the failure and report the largest contributors.
        for memory_id, peak, capacity in capacity_violations(
            periodic_capacity_peak(windows), memory_capacities
        ):
            contributors = tuple(
                sorted(
                    (
                        (value_id, size_bytes)
                        for value_id, memory, size_bytes, *_ in resolved
                        if memory == memory_id
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            # A single oversized buffer is structural. Otherwise the body
            # itself overlaps even when iterations are fully separated, so more
            # II cannot help; mapping must reorder or re-tile the body.
            kind = (
                "structural"
                if any(
                    size_bytes > capacity
                    for _, memory, size_bytes, *_ in resolved
                    if memory == memory_id
                )
                else "intra_iteration"
            )
            raise MemoryCapacityError(
                f"tile scope {scope_path!r} needs {peak} bytes on memory "
                f"{memory_id!r} but its capacity is {capacity}; reduce the tile size",
                scope_path=scope_path,
                memory_id=memory_id,
                peak_bytes=peak,
                capacity_bytes=capacity,
                required_scale=peak / capacity,
                contributors=contributors,
                kind=kind,
            )
        raise ValueError("FX modulo event placement did not converge")
    # Step 10: Build the placement. Capacity is binding when the first feasible
    # interval is larger than the timing-only interval.
    capacity_bound = False
    if storage_specs and memory_capacities:
        capacity_bound = capacity_quanta > timing_quanta
    initiation_interval = capacity_quanta * _TIME_QUANTUM_SECONDS
    body_finish = max(
        (starts[event.node_id] + event.duration_seconds for event in events),
        default=0.0,
    )
    # A body that already fits within one interval can use its finish time as
    # the interval; every residency window ends by body_finish, so this cannot
    # create new cross-iteration overlap.
    if body_finish <= initiation_interval + _EPSILON:
        initiation_interval = body_finish
    return EventModuloPlacement(
        scope_path,
        initiation_interval,
        body_finish,
        starts,
        capacity_bound=capacity_bound,
    )


def _schedule_periodic(
    events: tuple[_PeriodicEvent, ...],
    dependencies: Mapping[str, tuple[str, ...]],
    recurrences: tuple[tuple[str, str], ...],
    initiation_interval: float,
    storage_specs: Iterable[
        tuple[
            str,
            str,
            int,
            tuple[str | None, str],
            tuple[str | None, str],
        ]
    ] = (),
    memory_capacities: Mapping[str, int] = MappingProxyType({}),
) -> dict[str, float] | None:
    """Try to place one body at a fixed initiation interval.

    Steps:
        1. Reject the interval if any event is longer than it.
        2. Greedily place each event once all of its dependencies are placed.
        3. Check loop-carried recurrences across the interval boundary.
        4. Check folded storage residency when specs and capacities are given.
        5. Return event starts when all checks pass, else ``None``.

    Args:
        events: One iteration's events with their resources and durations.
        dependencies: Intra-iteration ``event -> predecessors`` edges.
        recurrences: Loop-carried ``(producer, consumer)`` edges across
            iterations. The current FX caller supplies none; the check remains
            here for future recurrence lowering.
        initiation_interval: Candidate interval in seconds.
        storage_specs: Anchored residency specs for this body, resolved against
            the candidate placement; empty disables the capacity check.
        memory_capacities: Finite bytes per memory id; empty disables the check.

    Returns:
        ``event_id -> start_seconds`` when feasible, else ``None``.

    Example:
        A timing-only call ignores capacity::

            _schedule_periodic(events, {}, (), 1.0)
    """
    # Step 1: An event that does not fit in one interval cannot be placed.
    if any(item.duration_seconds > initiation_interval + _EPSILON for item in events):
        return None
    by_id = {item.node_id: item for item in events}
    starts: dict[str, float] = {}
    calendars: dict[str, list[EventInterval]] = defaultdict(list)
    # Step 2: Greedy list placement until every event has a start time. An
    # event is ready only when all of its intra-body predecessors are placed.
    while len(starts) < len(events):
        ready = [
            item
            for item in events
            if item.node_id not in starts
            and all(dep in starts for dep in dependencies[item.node_id])
        ]
        if not ready:
            return None
        for item in ready:
            # Earliest start respects predecessor finishes, then resource slots.
            earliest = max(
                (
                    starts[dep] + by_id[dep].duration_seconds
                    for dep in dependencies[item.node_id]
                ),
                default=0.0,
            )
            start = _first_periodic_slot(
                calendars,
                item.resource_ids,
                earliest,
                item.duration_seconds,
                initiation_interval,
            )
            if start is None:
                return None
            starts[item.node_id] = start
    # Step 3: Check loop-carried recurrences. The current caller passes none,
    # but the check keeps the interface ready for recurrence lowering.
    if any(
        starts[destination] + initiation_interval
        < starts[source] + by_id[source].duration_seconds - _EPSILON
        for source, destination in recurrences
    ):
        return None
    # Step 4: Reject when steady-state residency (windows folded on the
    # interval) overflows any finite memory capacity.
    if storage_specs and memory_capacities:
        event_times = {
            item.node_id: (
                starts[item.node_id],
                starts[item.node_id] + item.duration_seconds,
            )
            for item in events
        }
        body_finish = max(finish for _, finish in event_times.values())
        resolved = resolve_storage_windows(storage_specs, event_times, body_finish)
        windows = tuple(
            (memory_id, size_bytes, reserve, release)
            for _, memory_id, size_bytes, reserve, release in resolved
        )
        violations = capacity_violations(
            periodic_capacity_peak(windows, initiation_interval), memory_capacities
        )
        if violations:
            return None
    # Step 5: Timing, recurrences, and capacity all passed.
    return starts


def _first_periodic_slot(
    calendars: Mapping[str, list[EventInterval]],
    resources: tuple[str, ...],
    start: float,
    duration: float,
    initiation_interval: float,
) -> float | None:
    """Find the earliest free periodic slot for one event.

    Steps:
        1. Start at the earliest time allowed by dependencies.
        2. Check every existing reservation on every required resource,
           replicated one period before and after the candidate so wrap-around
           conflicts are visible.
        3. On conflict, jump to the latest conflicting finish and retry.
        4. When no conflict remains, reserve the slot on every resource.
        5. Give up after a bounded number of retries.

    Args:
        calendars: Reserved intervals per resource, extended in place.
        resources: Resources the event needs simultaneously.
        start: Earliest candidate start time.
        duration: Event length in seconds.
        initiation_interval: Modulo period in seconds.

    Returns:
        The earliest start whose window is free across the wrap-around and
        reserves the interval, or ``None`` when the bounded search finds no slot.

    Example:
        >>> _first_periodic_slot(calendars, ("r",), 0.0, 1.0, 2.0)
        0.0
    """
    # Step 1: Begin at the dependency-constrained earliest start.
    candidate = start
    for _ in range(10_000):
        # Step 2: Build the conflicts visible in adjacent periods.
        conflicts: list[EventInterval] = []
        for resource_id in resources:
            for interval in calendars[resource_id]:
                # Period shift nearest the candidate, checked one either side so
                # iterations before and after the current one are also covered.
                shift = round(
                    (candidate - interval.start_seconds) / initiation_interval
                )
                for offset in (shift - 1, shift, shift + 1):
                    shifted_start = max(
                        0.0,
                        interval.start_seconds + offset * initiation_interval,
                    )
                    shifted_finish = max(
                        shifted_start,
                        interval.finish_seconds + offset * initiation_interval,
                    )
                    if (
                        candidate < shifted_finish - _EPSILON
                        and candidate + duration > shifted_start + _EPSILON
                    ):
                        conflicts.append(
                            EventInterval(
                                shifted_start, max(shifted_start, shifted_finish)
                            )
                        )
        if not conflicts:
            # Step 4: The slot is free; reserve it on every required resource.
            interval = EventInterval(candidate, candidate + duration)
            for resource_id in resources:
                calendars[resource_id].append(interval)
            return candidate
        # Step 3: Move to the end of the latest conflict and try again.
        candidate = max(item.finish_seconds for item in conflicts)
    # Step 5: The bounded search found no free slot.
    return None
