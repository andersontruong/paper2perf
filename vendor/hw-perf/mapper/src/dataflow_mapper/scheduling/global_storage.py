"""Storage-liveness composition across nested scheduling scopes.

Each mapping scope composes its own subtree before returning to its parent.
Windows stay keyed by scope, so parent and child reservations on the same memory
cannot jointly overflow. Wrapper events nest a child scope inside its parent:

* a **shared external input** is counted once, no matter how many scopes read it;
* a child's **output** residency extends to the parent's first consumption, so a
  value's production and downstream consumption are one lifetime; and
* the per-memory **peak residency** is checked across the mapping subtree, not
  just within one scope.

View aliases are resolved across scopes to the physical buffer they describe;
materialized views keep their own transfer birth instead of inheriting the outer
base's lifetime.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from dataflow_mapper.scheduling.model import (
    ScopeKey,
    TilingScope,
    capacity_violations,
    periodic_capacity_peak,
)

# (value_id, memory_id, size_bytes, reserve, release, route_scope, movement),
# local time. ``route_scope`` stays tile-coordinate-only; the ledger key is the
# full path-only scope identity.
StorageSpec = tuple[str, str, int, float, float, tuple[TilingScope, ...], str | None]


@dataclass
class _ScopeRecord:
    """One scope's local residency plus its external input/output value names."""

    windows: list[StorageSpec] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    sinks: set[str] = field(default_factory=set)


@dataclass
class _WrapperRecord:
    """A parent's aggregate event wrapping one child scope, local to the parent."""

    start: float
    consumer_finish: float
    producer_start: float
    has_consumer: bool


@dataclass(frozen=True)
class GlobalStorageResolution:
    """A resolved mapping subtree: chained windows, peak bytes, and violations."""

    windows_by_scope: dict[ScopeKey, tuple[StorageSpec, ...]]
    peak_bytes: dict[str, int]
    violations: tuple[tuple[str, int, int], ...]
    scope_by_memory: dict[str, ScopeKey]


class StorageLedger:
    """Collects per-scope residency and resolves mapping subtrees."""

    def __init__(self) -> None:
        self._scopes: dict[ScopeKey, _ScopeRecord] = {}
        self._children: dict[ScopeKey, dict[str, ScopeKey]] = defaultdict(dict)
        self._wrappers: dict[tuple[ScopeKey, str], _WrapperRecord] = {}
        # (parent scope, alias sink value) -> (child scope, child sink value).
        self._aliases: dict[tuple[ScopeKey, str], tuple[ScopeKey, str]] = {}
        # (scope, view value) -> physical base (scope, value) it aliases.
        self._view_aliases: dict[tuple[ScopeKey, str], tuple[ScopeKey, str]] = {}

    def record_scope(
        self,
        key: ScopeKey,
        windows: list[StorageSpec],
        sources: set[str],
        sinks: set[str],
        aliases: dict[str, tuple[ScopeKey, str]] | None = None,
        view_aliases: dict[str, str | tuple[ScopeKey, str]] | None = None,
    ) -> None:
        """Record one scope's local residency windows and boundary value names."""
        record = self._scopes.setdefault(key, _ScopeRecord())
        record.windows = list(windows)
        record.sources = set(sources)
        record.sinks = set(sinks)
        for value_id, target in (aliases or {}).items():
            self._aliases[(key, value_id)] = target
        for value_id, base_value in (view_aliases or {}).items():
            if isinstance(base_value, str):
                self._view_aliases[(key, value_id)] = (key, base_value)
            else:
                self._view_aliases[(key, value_id)] = base_value

    def record_view_alias(
        self,
        scope_key: ScopeKey,
        view_value: str,
        base_scope: ScopeKey,
        base_value: str,
    ) -> None:
        """Record that ``view_value`` in one scope aliases another scope's value.

        Args:
            scope_key: Scope that owns the view.
            view_value: View value name in ``scope_key``.
            base_scope: Scope that owns the base value.
            base_value: Base value name in ``base_scope``.
        """
        self._view_aliases[(scope_key, view_value)] = (base_scope, base_value)

    def record_child(
        self, parent_key: ScopeKey, node_name: str, child_key: ScopeKey
    ) -> None:
        """Record that ``node_name`` in the parent wraps ``child_key``."""
        self._children[parent_key][node_name] = child_key

    def record_wrapper(
        self,
        scope_key: ScopeKey,
        node_name: str,
        start: float,
        consumer_finish: float,
        producer_start: float,
        has_consumer: bool,
    ) -> None:
        """Record a compute event's timing so nested children can be offset."""
        self._wrappers[(scope_key, node_name)] = _WrapperRecord(
            start, consumer_finish, producer_start, has_consumer
        )

    def _offsets(self, root_key: ScopeKey) -> dict[ScopeKey, float]:
        """Global start time of each scope by composing nested wrapper events."""
        offsets: dict[ScopeKey, float] = {root_key: 0.0}
        stack = [root_key]
        while stack:
            parent = stack.pop()
            for node_name, child in self._children.get(parent, {}).items():
                wrapper = self._wrappers.get((parent, node_name))
                start = wrapper.start if wrapper is not None else 0.0
                offsets[child] = offsets.get(parent, 0.0) + start
                stack.append(child)
        return offsets

    def resolve(
        self,
        root_key: ScopeKey,
        memory_capacities: dict[str, int],
    ) -> GlobalStorageResolution:
        """Resolve one mapping subtree and check its composed capacity.

        A child's output window is extended to the parent's first consumption and a
        child's input window to the parent's production, then shared external inputs
        are merged so each physical buffer is counted once. Per-memory peak bytes are
        measured over the subtree and compared against ``memory_capacities``.

        Args:
            root_key: Scope key of the mapping subtree root.
            memory_capacities: Finite capacity in bytes per memory id.

        Returns:
            Chained per-scope windows (local time), composed peak bytes per memory,
            and one ``(memory, peak, capacity)`` tuple per violating memory.
        """
        offsets = self._offsets(root_key)
        # Resolve only the subtree owned by this mapping scope. A parent sees
        # itself plus its descendants; a child never sees unrelated siblings.
        subtree = set(offsets)
        # (scope, value, memory, route, movement, size, birth, death) subtree time.
        entries: list[list[Any]] = []
        for scope_key, record in self._scopes.items():
            if scope_key not in subtree:
                continue
            offset = offsets.get(scope_key, 0.0)
            for (
                value_id,
                memory_id,
                size_bytes,
                reserve,
                release,
                route,
                movement,
            ) in record.windows:
                entries.append(
                    [
                        scope_key,
                        value_id,
                        memory_id,
                        route,
                        movement,
                        size_bytes,
                        offset + reserve,
                        offset + release,
                    ]
                )
        # Map each child scope to its immediate wrapper in the parent.
        parent_of = {
            child: (parent, node_name)
            for parent, children in self._children.items()
            for node_name, child in children.items()
            if child in subtree and parent in subtree
        }
        # Extend a child's input residency to the parent that produces it.
        for child, (parent, node_name) in parent_of.items():
            wrapper = self._wrappers.get((parent, node_name))
            child_record = self._scopes.get(child)
            if wrapper is None or child_record is None:
                continue
            producer_birth = offsets.get(parent, 0.0) + wrapper.producer_start
            for entry in entries:
                if entry[0] != child or entry[1] not in child_record.sources:
                    continue
                # A materialized view is its own buffer: it appears when its
                # transfer writes it, not when the outer base was produced.
                if (child, entry[1]) in self._view_aliases:
                    continue
                entry[6] = min(entry[6], producer_birth)
        # Extend a child's output residency through every ancestor wrapper. This
        # also covers a grandchild sink when the intermediate wrapper's own sink
        # was aliased away.
        for entry in entries:
            record = self._scopes.get(entry[0])
            if record is None or entry[1] not in record.sinks:
                continue
            current = entry[0]
            while current in parent_of:
                parent, node_name = parent_of[current]
                wrapper = self._wrappers.get((parent, node_name))
                if wrapper is None:
                    break
                entry[7] = max(
                    entry[7],
                    offsets.get(parent, 0.0) + wrapper.consumer_finish,
                )
                # A consumed wrapper ends the passthrough chain. A sink in the
                # parent also terminates it; otherwise the output is forwarded
                # to the next enclosing scope and its lifetime continues.
                parent_record = self._scopes.get(parent)
                if wrapper.has_consumer or (
                    parent_record is not None and node_name in parent_record.sinks
                ):
                    break
                current = parent
        # Follow view aliases across scopes to the physical buffer they all
        # describe. The terminal name is only an identity: each memory keeps its
        # own window, so an L2 slice and its outer-memory base do not merge.
        def view_base(scope_key: ScopeKey, value_id: str) -> tuple[ScopeKey, str]:
            seen: set[tuple[ScopeKey, str]] = set()
            while (scope_key, value_id) in self._view_aliases:
                if (scope_key, value_id) in seen:
                    break
                seen.add((scope_key, value_id))
                scope_key, value_id = self._view_aliases[(scope_key, value_id)]
            return scope_key, value_id

        # Merge to one window per physical buffer: shared external inputs and
        # aliased views use the shared ("*") owner; everything else stays local
        # to its own scope.
        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        alias_entries: list[list[Any]] = []
        for entry in entries:
            original_scope, original_value = entry[0], entry[1]
            if (original_scope, original_value) in self._aliases:
                alias_entries.append(entry)
                continue
            scope_key, value_id = view_base(original_scope, original_value)
            is_alias = (scope_key, value_id) != (original_scope, original_value)
            (
                scope_key,
                value_id,
                memory_id,
                route,
                movement,
                size_bytes,
                birth,
                death,
            ) = [scope_key, value_id, *entry[2:]]
            record = self._scopes.get(scope_key)
            owner = (
                "*"
                if is_alias or (record is not None and value_id in record.sources)
                else scope_key
            )
            group_key = (
                ("*", value_id, memory_id, None, None)
                if is_alias or owner == "*"
                else (owner, value_id, memory_id, route, movement)
            )
            group = groups.setdefault(
                group_key,
                {
                    "value": value_id,
                    "memory": memory_id,
                    "route": route,
                    "movement": movement,
                    "size": 0,
                    "birth": float("inf"),
                    "death": float("-inf"),
                    "scopes": set(),
                },
            )
            group["size"] = max(group["size"], size_bytes)
            group["birth"] = min(group["birth"], birth)
            group["death"] = max(group["death"], death)
            group["scopes"].add(original_scope)
        # Alias sinks share their canonical child buffer. Extend the canonical
        # window and remember which parent scopes must display the alias.
        alias_windows: dict[ScopeKey, list[tuple[Any, ...]]] = defaultdict(list)
        for entry in alias_entries:
            parent_scope, alias_value = entry[0], entry[1]
            targets = self._aliases[(parent_scope, alias_value)]
            if (
                isinstance(targets, tuple)
                and targets
                and not isinstance(targets[0], tuple)
            ):
                targets = (targets,)
            for target in targets:
                seen = set()
                while target in self._aliases and target not in seen:
                    seen.add(target)
                    target = self._aliases[target]
                    if isinstance(target, tuple) and target and isinstance(
                        target[0], tuple
                    ):
                        target = target[0]
                child_scope, child_value = target
                group = next(
                    (
                        candidate
                        for candidate in groups.values()
                        if candidate["value"] == child_value
                        and child_scope in candidate["scopes"]
                    ),
                    None,
                )
                if group is None:
                    continue
                group["birth"] = min(group["birth"], entry[6])
                group["death"] = max(group["death"], entry[7])
                alias_windows[parent_scope].append(
                    (alias_value, group, entry[3], entry[4])
                )
        # Per-scope local windows for display, and the subtree peak for capacity.
        windows_by_scope: dict[ScopeKey, list[StorageSpec]] = defaultdict(list)
        subtree_windows: list[tuple[str, int, float, float]] = []
        scope_by_memory: dict[str, ScopeKey] = {}
        largest: dict[str, int] = defaultdict(int)
        for group in groups.values():
            value_id = group["value"]
            for scope_key in group["scopes"]:
                offset = offsets.get(scope_key, 0.0)
                windows_by_scope[scope_key].append(
                    (
                        value_id,
                        group["memory"],
                        group["size"],
                        group["birth"] - offset,
                        group["death"] - offset,
                        group["route"],
                        group["movement"],
                    )
                )
            subtree_windows.append(
                (group["memory"], group["size"], group["birth"], group["death"])
            )
            if group["size"] > largest[group["memory"]]:
                largest[group["memory"]] = group["size"]
                scope_by_memory[group["memory"]] = next(iter(group["scopes"]))
        for parent_scope, aliases in alias_windows.items():
            parent_offset = offsets.get(parent_scope, 0.0)
            for alias_value, group, route, movement in aliases:
                windows_by_scope[parent_scope].append(
                    (
                        alias_value,
                        group["memory"],
                        group["size"],
                        group["birth"] - parent_offset,
                        group["death"] - parent_offset,
                        route,
                        movement,
                    )
                )
        peak = periodic_capacity_peak(subtree_windows)
        return GlobalStorageResolution(
            windows_by_scope={
                scope_key: tuple(windows)
                for scope_key, windows in windows_by_scope.items()
            },
            peak_bytes=peak,
            violations=capacity_violations(peak, memory_capacities),
            scope_by_memory=scope_by_memory,
        )


__all__ = ["GlobalStorageResolution", "ScopeKey", "StorageLedger", "StorageSpec"]
