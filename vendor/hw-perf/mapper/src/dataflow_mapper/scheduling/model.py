"""Immutable interval records for Parabolic scheduling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor


class MemoryCapacityError(ValueError):
    """A schedule's live storage exceeds a finite hardware memory capacity.

    Optional attributes localize the violation so a caller can act on it (e.g. a
    re-tiling driver reads ``required_scale`` and ``scope_path``) instead of
    parsing the message.
    """

    def __init__(
        self,
        message: str,
        *,
        scope_path: ScopeKey = (),
        memory_id: str | None = None,
        peak_bytes: int | None = None,
        capacity_bytes: int | None = None,
        required_scale: float | None = None,
        contributors: tuple[tuple[str, int], ...] = (),
        kind: str = "capacity",
    ) -> None:
        """Record the message plus optional tile-local capacity diagnostics.

        Args:
            message: Human-readable description (unchanged for existing callers).
            scope_path: Tile scope path that overflowed, empty when not
                scope-local. May include wrapper names and ``TilingScope``
                coordinates; retiling extracts the last coordinate.
            memory_id: Offending memory id.
            peak_bytes: Peak concurrent residency on that memory.
            capacity_bytes: That memory's finite capacity.
            required_scale: ``peak_bytes / capacity_bytes``; how much smaller the
                tile's buffers must be for this body to fit.
            contributors: ``(value_id, size_bytes)`` buffer sizes on the memory,
                largest first, to hint which value drives the peak.
            kind: Failure class used to choose the repair: ``"structural"``
                (one buffer cannot fit), ``"intra_iteration"`` (the body itself
                overlaps), ``"inter_iteration"`` (successive iterations overlap),
                or ``"cross_scope"`` (parent and child reservations conflict);
                ``"capacity"`` is the unclassified default.

        Example:
            >>> try:
            ...     ...
            ... except MemoryCapacityError as error:
            ...     scale = error.required_scale   # e.g. 3.0
        """
        super().__init__(message)
        self.scope_path = scope_path
        self.memory_id = memory_id
        self.peak_bytes = peak_bytes
        self.capacity_bytes = capacity_bytes
        self.required_scale = required_scale
        self.contributors = contributors
        self.kind = kind


@dataclass(frozen=True, order=True)
class TilingScope:
    """Mapper-owned execution scope for one explicitly tiled axis split."""

    axis: str
    factor: int
    tile_index: int
    contracting: bool = False
    split: int = 0

    def __post_init__(self) -> None:
        """Validate a concrete tile coordinate within its selected scope."""
        if (
            not self.axis
            or isinstance(self.factor, bool)
            or not isinstance(self.factor, int)
            or self.factor < 2
            or isinstance(self.tile_index, bool)
            or not isinstance(self.tile_index, int)
            or not 0 <= self.tile_index < self.factor
            or not isinstance(self.contracting, bool)
            or isinstance(self.split, bool)
            or not isinstance(self.split, int)
            or self.split < 0
        ):
            raise ValueError("invalid tiling scope")


# Path-only identity of one mapper scope. A scope path alternates wrapper node
# names with the tile coordinate (``TilingScope``) of the branch that enters the
# wrapper. It never includes the module's own tiling axis.
ScopeKey = tuple[TilingScope | str, ...]


@dataclass(frozen=True, order=True)
class EventInterval:
    """Half-open execution interval `[start_seconds, finish_seconds)` for one event.

    Ordering is lexicographic on (start, finish), so sorting intervals groups them by
    start time and `min` selects the earliest-finishing interval.
    """

    start_seconds: float
    finish_seconds: float

    def __post_init__(self) -> None:
        """Reject negative starts and intervals that finish before they start."""
        if self.start_seconds < 0 or self.finish_seconds < self.start_seconds:
            raise ValueError("invalid event interval")

    @property
    def duration_seconds(self) -> float:
        """Elapsed time of the event in seconds."""
        return self.finish_seconds - self.start_seconds


def earliest_free_interval(
    calendars: Mapping[str, Sequence[EventInterval]],
    resource_ids: Iterable[str],
    earliest_start: float,
    duration: float,
) -> EventInterval:
    """First start time at which every listed resource is free for ``duration``.

    Args:
        calendars: Reserved intervals per resource id.
        resource_ids: Resources the event needs simultaneously; duplicates are
            collapsed.
        earliest_start: Data-dependent lower bound on the start time.
        duration: Length of the event in seconds.

    Returns:
        The earliest ``[start, start + duration)`` that overlaps no reservation
        on any listed resource.

    Example:
        Find a slot after an existing ``[0, 2)`` reservation::

            earliest_free_interval({"r": [EventInterval(0.0, 2.0)]}, ["r"], 0.0, 1.0)
            # -> EventInterval(2.0, 3.0)
    """
    resources = tuple(dict.fromkeys(resource_ids))
    start = earliest_start
    while True:
        finish = start + duration
        released = [
            occupied.finish_seconds
            for resource_id in resources
            for occupied in calendars.get(resource_id, ())
            if occupied.start_seconds < finish and start < occupied.finish_seconds
        ]
        if not released:
            return EventInterval(start, finish)
        start = max(released)


def periodic_capacity_peak(
    windows: Iterable[tuple[str, int, float, float]],
    period: float | None = None,
) -> dict[str, int]:
    """Peak concurrent storage bytes per memory over a set of live windows.

    Each window is a half-open ``[reserve_start, release)`` interval reserving
    ``size_bytes`` on ``memory_id``. When ``period`` is given the windows are
    folded into one period: every window is replicated at integer period shifts
    so overlap that wraps around the boundary is counted.

    Args:
        windows: ``(memory_id, size_bytes, reserve_start, release)`` intervals on
            a common timeline.
        period: Period to fold into, or ``None`` for a non-periodic timeline.

    Returns:
        Peak concurrent bytes per memory id (memories with no live bytes omitted).

    Example:
        Two overlapping buffers on one memory peak at their sum::

            windows = [("gpu", 100, 0.0, 2.0), ("gpu", 50, 1.0, 3.0)]
            periodic_capacity_peak(windows)
            # -> {"gpu": 150}
    """
    shards: dict[str, list[tuple[float, float, int]]] = {}
    for memory_id, size_bytes, reserve_start, release in windows:
        # Ignore empty or malformed windows.
        if size_bytes <= 0 or release <= reserve_start:
            continue
        shards.setdefault(memory_id, []).append((reserve_start, release, size_bytes))
    peak: dict[str, int] = {}
    for memory_id, intervals in shards.items():
        if period is None:
            copies = list(intervals)
        else:
            # Fold every window into the reference period [0, period). A window
            # may start several periods away from that range, so derive the
            # shifts per window instead of assuming a span around zero.
            copies = []
            for start, release, size in intervals:
                first_shift = floor(-release / period) + 1
                last_shift = ceil((period - start) / period) - 1
                copies.extend(
                    (
                        start + shift * period,
                        release + shift * period,
                        size,
                    )
                    for shift in range(first_shift, last_shift + 1)
                )
        times = sorted(
            {time for start, release, _ in copies for time in (start, release)}
        )
        # Only window boundaries change the live sum, so the peak is one of them.
        peak[memory_id] = max(
            (
                sum(size for start, release, size in copies if start <= time < release)
                for time in times
            ),
            default=0,
        )
    return peak


def capacity_violations(
    peak_bytes: Mapping[str, int],
    capacities: Mapping[str, int],
) -> tuple[tuple[str, int, int], ...]:
    """Memories whose peak concurrent residency exceeds a finite capacity.

    Args:
        peak_bytes: Peak concurrent bytes per memory id, as returned by
            :func:`periodic_capacity_peak`; absent memories count as zero.
        capacities: Finite capacity in bytes per memory id; absent memories are
            unconstrained and never reported.

    Returns:
        One ``(memory_id, peak_bytes, capacity_bytes)`` tuple per violating
        memory, in ``capacities`` order.

    Example:
        >>> capacity_violations({"gpu": 150}, {"gpu": 100, "hbm": 1000})
        (('gpu', 150, 100),)
    """
    return tuple(
        (memory_id, peak_bytes.get(memory_id, 0), capacity)
        for memory_id, capacity in capacities.items()
        if peak_bytes.get(memory_id, 0) > capacity
    )


def resolve_storage_windows(
    specs: Iterable[
        tuple[
            str,
            str,
            int,
            tuple[str | None, str],
            tuple[str | None, str],
        ]
    ],
    event_times: Mapping[str, tuple[float, float]],
    default_finish: float,
) -> tuple[tuple[str, str, int, float, float], ...]:
    """Resolve anchored residency specs into concrete windows at given event times.

    Each spec is ``(value_id, memory_id, size_bytes, reserve, release)`` where a
    boundary is ``(event_id, "start" | "finish")``. The spec builder anchors an
    external scoped input at its first use: a value with no producer and no
    inbound transfer in the scope is born when its first consumer starts, so
    recursive child iteration offsets survive instead of collapsing to parent
    time zero. A boundary whose event is absent from ``event_times`` (or has
    ``None`` id) means the scope never uses the value; it falls back to ``0.0``
    for reserve and to ``default_finish`` for release. Specs sharing
    ``(value_id, memory_id)`` describe one physical buffer, so their windows
    collapse to the earliest reserve, latest release, and largest size.

    Args:
        specs: Anchored residency specs, all on one (scope) timeline.
        event_times: ``event_id -> (start_seconds, finish_seconds)``.
        default_finish: Release time for a window whose release event is absent.

    Returns:
        ``(value_id, memory_id, size_bytes, reserve_start, release)`` tuples, one
        per ``(value_id, memory_id)`` group, sorted by that key.

    Example:
        >>> specs = [("x", "gpu", 100, ("p", "finish"), ("c", "start"))]
        >>> resolve_storage_windows(specs, {"p": (0.0, 1.0), "c": (2.0, 3.0)}, 5.0)
        (('x', 'gpu', 100, 1.0, 2.0),)
    """

    def boundary(anchor: tuple[str | None, str], fallback: float) -> float:
        """Time of one anchored boundary, or ``fallback`` when it is absent.

        Args:
            anchor: ``(event_id, "start" | "finish")``; ``None`` means no event.
            fallback: Time to use when the event is unknown.

        Returns:
            The anchored event time, else ``fallback``.

        Example:
            >>> boundary(("p", "finish"), 0.0)  # with p's window [0, 1)
            1.0
        """
        event_id, edge = anchor
        times = event_times.get(event_id) if event_id is not None else None
        if times is None:
            return fallback
        return times[0] if edge == "start" else times[1]

    grouped: dict[tuple[str, str], tuple[float, float, int]] = {}
    # Collapse all specs that describe the same buffer to one union window.
    for value_id, memory_id, size_bytes, reserve, release in specs:
        # A zero-byte alias carries no residency; its boundary events may be
        # scheduled outside the owning tile and need not span a valid window.
        if size_bytes <= 0:
            continue
        reserve_time = boundary(reserve, 0.0)
        release_time = boundary(release, default_finish)
        if release_time < reserve_time:
            raise ValueError(f"invalid storage lifetime for value {value_id!r}")
        existing = grouped.get((value_id, memory_id))
        if existing is None:
            grouped[(value_id, memory_id)] = (reserve_time, release_time, size_bytes)
            continue
        grouped[(value_id, memory_id)] = (
            min(existing[0], reserve_time),
            max(existing[1], release_time),
            max(existing[2], size_bytes),
        )
    return tuple(
        (value_id, memory_id, size_bytes, reserve, release)
        for (value_id, memory_id), (reserve, release, size_bytes) in sorted(
            grouped.items()
        )
    )
