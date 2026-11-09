"""FX event lowering: placed task graph -> physical event graph + schedule.

Builds the physical event graph (compute and transfer events with their
intervals), anchors storage-residency windows to the events that hand values
off, and hands the graph to the canonical hierarchical scheduler for timing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

import networkx as nx

from dataflow_mapper.routing import movement_resource, task_key
from dataflow_mapper.scheduling.hierarchical import (
    FXHierarchicalEventSchedule,
    build_fx_event_scope_tree,
    schedule_hierarchical_fx_events,
)
from dataflow_mapper.scheduling.model import (
    EventInterval,
    MemoryCapacityError,
    ScopeKey,
    TilingScope,
    capacity_violations,
    earliest_free_interval,
    periodic_capacity_peak,
    resolve_storage_windows,
)
from dataflow_mapper.scheduling.modulo import schedule_modulo_event_body
from hw_model.model import HardwareGraph

if TYPE_CHECKING:
    from dataflow_mapper.scheduling.global_storage import StorageLedger
    from dataflow_mapper.viz.trace import MappingTrace


def schedule_fx_graph(
    graph: nx.MultiDiGraph,
    hardware: HardwareGraph,
    *,
    validate_memory_capacity: bool = True,
    trace: MappingTrace | None = None,
    scope_path: ScopeKey = (),
    module_axis: str | None = None,
    ledger: StorageLedger | None = None,
    scope_key: ScopeKey | None = None,
) -> tuple[nx.DiGraph, FXHierarchicalEventSchedule]:
    """Lower a placed/routed task graph to a timed physical event graph + schedule.

    The lowering proceeds in these steps:

        1. Read tiling provenance off the task graph into per-node scope paths.
        2. Lower each placed compute node to a compute event carrying its resource
           set and duration.
        3. Lower each routed movement to a chain of per-link transfer events,
           reusing one chain for parent-owned movements shared by sibling tiles.
        4. Greedily assign every event a start interval, reserving all resources
           it uses.
        5. Anchor a storage-residency window per memory hop of each movement to
           the producer/consumer events, so residency can be re-resolved per
           placement.
        6. Build the tile-scope tree, place every leaf scope with modulo
           scheduling, and compose the hierarchy into a makespan.
        7. Resolve each pass-through storage node's anchored lifetime against
           the composed timeline, then enforce per-memory capacity.

    Args:
        graph: Placed/routed task graph whose compute nodes carry a resource,
            duration, an optional structural scope path, and whose edges carry an
            ordered ``route`` of link ids.
        hardware: Hardware graph supplying link latencies, memory nodes, and
            memory capacities.
        validate_memory_capacity: When true, raise ``MemoryCapacityError`` if the
            flat peak storage exceeds any memory budget.
        trace: Optional collector for the event graph, schedule, storage windows,
            and modulo placements.
        scope_path: Ancestor path key for this scope.
        module_axis: The scope module's tiling axis, or ``None`` for root.

    Returns:
        ``(event_graph, hierarchical_schedule)``. The event graph holds compute
        and transfer events with their intervals; the hierarchical schedule
        composes the nested tile scopes into a makespan.

    Example:
        Time an already placed and routed Linear task graph::

            graph = place_fx_graph(graph_module, hardware, {"M": 32, "K": 128, "N": 128})
            select_routes(graph, hardware)
            event_graph, schedule = schedule_fx_graph(graph, hardware)
            schedule.makespan_seconds  # total modeled latency in seconds

    Scope paths and intervals are carried on the event graph; storage residency
    is derived from the anchored specs; timing comes from the hierarchical
    scheduler.
    """
    # Structural scope paths read off the task graph: task id -> scope path, the
    # first-seen axis order, and each axis's tile factor.
    node_scopes, axis_order, repetitions = _scope_paths(graph)
    # Physical event graph: compute and transfer events plus data edges.
    event_graph = nx.DiGraph()
    # Scope path that owns each routed edge's transfer (set in the transfer pass).
    route_scopes: dict[str, tuple[TilingScope, ...]] = {}
    # Transfer chain (ordered event ids) produced for each routed edge.
    route_transfer_events: dict[str, tuple[str, ...]] = {}
    # Cache of transfer chains reused by sibling tiles under a shared parent scope.
    shared_transfers: dict[tuple[object, ...], tuple[str, ...]] = {}

    # Compute events come from the placed compute nodes in the task graph.
    for node_id, node_data in graph.nodes(data=True):
        if node_data["kind"] != "compute":
            continue
        # Boundary tasks are skipped: they contribute residency, not work.
        # `resource_id` is the single placed choice; `resources` is the full set
        # reserved by an aggregate (call_module) task, else just that one choice.
        event_graph.add_node(
            f"compute:{node_id}",
            kind="compute",
            source_node_name=node_id,
            source_edge_id=None,
            resource_id=node_data["resource"],
            duration=float(node_data["seconds"]),
            scopes=node_scopes.get(node_id, ()),
            resources=tuple(node_data.get("resources") or (node_data["resource"],)),
            branch_index=node_data.get("branch_index"),
        )

    # Transfer events: one per link hop of each routed edge. A movement owned by
    # a shared parent scope reuses one transfer chain across its sibling tiles.
    links_by_id = hardware.links_by_id()
    for source_task, destination_task, edge_id, data in graph.edges(
        keys=True, data=True
    ):
        source_event = f"compute:{task_key(source_task)}"
        destination_event = f"compute:{task_key(destination_task)}"
        # `predecessor` is the event the value flows from; it is threaded through
        # each hop so the hop events chain producer -> ... -> consumer.
        predecessor = source_event if event_graph.has_node(source_event) else None
        link_ids = data["route"]
        # Endpoint scopes: boundary task ids carry a ``source:``/``sink:``
        # prefix and may now carry their own scope path. Look up the full id
        # first, then fall back to the stripped value name for older graphs.
        source_scopes = node_scopes.get(source_task) or node_scopes.get(
            source_task.removeprefix("source:").removeprefix("sink:"), ()
        )
        destination_scopes = node_scopes.get(destination_task) or node_scopes.get(
            destination_task.removeprefix("sink:").removeprefix("source:"), ()
        )
        # Producer owns the movement when its scope equals or contains the
        # consumer's, so sibling tiles share one parent-owned transfer.
        if not source_scopes:
            scope = ()
        elif destination_scopes[: len(source_scopes)] == source_scopes:
            scope = source_scopes
        elif source_scopes[: len(destination_scopes)] == destination_scopes:
            # The destination is the ancestor: the parent scope owns this
            # movement, not the producing child.
            scope = destination_scopes
        else:
            scope = destination_scopes
        route_scopes[edge_id] = scope
        if not link_ids:
            # Same-resource movement: no links, so just a direct data edge.
            route_transfer_events[edge_id] = ()
            if predecessor and event_graph.has_node(destination_event):
                event_graph.add_edge(predecessor, destination_event)
            continue

        # A movement is parent-owned when the owning scope is a strict ancestor of
        # the consumer's scope, i.e. sibling tiles consume the same transfer.
        is_parent_owned = (
            len(scope) < len(destination_scopes)
            and destination_scopes[: len(scope)] == scope
        )
        # Identity of a transfer: same value, endpoints, route, and byte counts.
        shared_key = (
            data["value_id"],
            movement_resource(graph, source_task, data["value_id"]),
            movement_resource(graph, destination_task, data["value_id"]),
            tuple(link_ids),
            tuple(data["transfer_bytes"]),
        )
        # Reuse an identical parent-owned chain instead of re-emitting it per tile.
        shared = shared_transfers.get(shared_key) if is_parent_owned else None
        if shared is not None:
            # Splice into the cached chain: producer -> first hop, last hop onward.
            if predecessor:
                event_graph.add_edge(predecessor, shared[0])
            predecessor = shared[-1]
            route_transfer_events[edge_id] = shared
        else:
            # Lower each link hop into its own transfer event over that link.
            transfer_ids = []
            for hop_index, (link_id, byte_count) in enumerate(
                zip(link_ids, data["transfer_bytes"], strict=True)
            ):
                link = links_by_id[link_id]
                event_id = f"transfer:{edge_id}:{hop_index}"
                event_graph.add_node(
                    event_id,
                    kind="transfer",
                    source_node_name=source_task,
                    source_edge_id=edge_id,
                    resource_id=link_id,
                    duration=hardware.link_seconds(link, byte_count),
                    scopes=scope,
                )
                if predecessor:
                    event_graph.add_edge(predecessor, event_id)
                transfer_ids.append(event_id)
                predecessor = event_id
            # Remember the chain so sibling tiles can share it.
            transfer_chain = tuple(transfer_ids)
            route_transfer_events[edge_id] = transfer_chain
            if is_parent_owned:
                shared_transfers[shared_key] = transfer_chain
        # Close the chain onto the consumer when it exists in this scope.
        if event_graph.has_node(destination_event) and predecessor:
            event_graph.add_edge(predecessor, destination_event)

    # Mapping-phase ordering for intra-iteration materialization overlap. If
    # sibling views of one base land in the same memory and together exceed its
    # capacity, the next view must wait until the previous view's consumers are
    # done. This is a body-ordering decision, not something modulo scheduling
    # should repair by shifting an event after the fact.
    node_order = {name: index for index, name in enumerate(graph.nodes)}
    # Each group is one capacity-triggered set of sibling views, in graph order:
    # (view task, its transfer chain, its bytes).
    ordering_groups: list[list[tuple[str, tuple[str, ...], int]]] = []

    materializations: dict[
        tuple[str, str], list[tuple[str, tuple[str, ...], int]]
    ] = {}
    for source_task, destination_task, edge_id, data in graph.edges(
        keys=True, data=True
    ):
        if not source_task.startswith("source:") or not destination_task.startswith(
            "source:"
        ):
            continue
        transfer_chain = route_transfer_events.get(edge_id, ())
        view_memory = graph.nodes[destination_task].get("resource")
        if not transfer_chain or view_memory is None:
            continue
        materializations.setdefault((source_task, str(view_memory)), []).append(
            (destination_task, transfer_chain, int(data["size_bytes"]))
        )
    for (_base_task, memory_id), views in materializations.items():
        capacity = hardware.memory_capacity(memory_id)
        if capacity is None or sum(view[2] for view in views) <= capacity:
            continue
        ordering_groups.append(views)

    # A view may be placed with its base and then transferred into the consumer
    # memory. Order those sibling view transfers too when their combined
    # footprint exceeds the destination memory.
    view_transfers: dict[
        tuple[str, str], dict[str, tuple[str, tuple[str, ...], int]]
    ] = {}
    for source_task, destination_task, edge_id, data in graph.edges(
        keys=True, data=True
    ):
        if not source_task.startswith("source:"):
            continue
        if graph.nodes[destination_task].get("kind") != "compute":
            continue
        base_value = graph.nodes[source_task].get("alias_value")
        transfer_chain = route_transfer_events.get(edge_id, ())
        if base_value is None or not transfer_chain:
            continue
        memory_id = movement_resource(graph, destination_task, data["value_id"])
        view_transfers.setdefault((str(base_value), memory_id), {}).setdefault(
            source_task,
            (destination_task, transfer_chain, int(data["size_bytes"])),
        )
    for (_base_value, memory_id), views in view_transfers.items():
        capacity = hardware.memory_capacity(memory_id)
        if capacity is None or sum(view[2] for view in views.values()) <= capacity:
            continue
        ordering_groups.append(
            [
                (source_task, transfer_chain, size_bytes)
                for source_task, (_, transfer_chain, size_bytes) in views.items()
            ]
        )

    for views in ordering_groups:
        views.sort(key=lambda view: node_order[view[0]])
        for (previous_view, _, _), (_, next_chain, _) in pairwise(views):
            next_transfer = next_chain[0]
            for consumer in graph.successors(previous_view):
                if graph.nodes[consumer].get("kind") != "compute":
                    continue
                consumer_event = f"compute:{consumer}"
                if not event_graph.has_node(consumer_event):
                    continue
                if nx.has_path(event_graph, next_transfer, consumer_event):
                    continue
                event_graph.add_edge(consumer_event, next_transfer)

    # Storage residency: one anchored spec per memory hop of each route. A value
    # is born when its producer starts, or at its first use when it enters this
    # scope as an external input without a scoped producer or inbound transfer.
    # It dies when its last consumer finishes (or, for a terminal value, when its
    # last outbound transfer completes). Anchoring external inputs at first use
    # preserves the child iteration offsets chosen by recursive modulo
    # scheduling. Each memory hop holds the value only while it physically
    # resides there, so the window is projected per memory. The specs also drive
    # the modulo body's capacity check.
    storage_specs: dict[
        tuple[tuple[TilingScope, ...], str, str, str | None], list[tuple]
    ] = {}
    # (scope, value, memory, movement) -> (predecessor ids, successor ids), used
    # to wire each pass-through storage node into the event-graph dataflow.
    storage_handoffs: dict[
        tuple[tuple[TilingScope, ...], str, str, str | None],
        tuple[set[str], set[str]],
    ] = {}
    for source_task, destination_task, edge_id, data in graph.edges(
        keys=True, data=True
    ):
        # Every resource the value visits, endpoints included. Walk the ordered
        # links, checking each attaches to the previous endpoint and lands on the
        # declared destination. Only memory entries reserve storage below.
        path = [movement_resource(graph, source_task, data["value_id"])]
        for link_id in data["route"]:
            link = links_by_id[link_id]
            if path[-1] not in (link[0], link[1]):
                raise ValueError(
                    f"route {data['route']!r} contains a discontinuous path"
                )
            path.append(hardware.link_other_endpoint(link, path[-1]))
        if path[-1] != movement_resource(graph, destination_task, data["value_id"]):
            raise ValueError("route ends at the wrong resource")
        path_resources = tuple(path)
        transfer_events = route_transfer_events[edge_id]
        source_event = f"compute:{task_key(source_task)}"
        destination_event = f"compute:{task_key(destination_task)}"
        source_present = event_graph.has_node(source_event)
        destination_present = event_graph.has_node(destination_event)
        # Liveness anchors resolve against any timeline (the greedy pass now, a
        # modulo placement later). An external input is born at its first inbound
        # transfer; if it has no scoped transfer either, it is born at its first
        # use so a child iteration's offset is not collapsed to parent time zero.
        # A terminal value dies at its last outbound transfer.
        birth = (
            (source_event, "start")
            if source_present
            else (transfer_events[0], "start")
            if transfer_events
            else (destination_event, "start")
            if destination_present
            else (None, "start")
        )
        death = (
            (destination_event, "finish")
            if destination_present
            else (
                (transfer_events[-1], "finish") if transfer_events else (None, "finish")
            )
        )
        scopes = route_scopes.get(edge_id, ())
        size_bytes = data["size_bytes"]
        value_id = data["value_id"]
        # Per-memory projection: the first memory holds the value from birth to
        # its first outbound hop, the last from its final inbound hop to death,
        # and intermediate memories only between adjacent hops.
        memory_indices = [
            index
            for index, resource_id in enumerate(path_resources)
            if hardware.is_memory(resource_id)
        ]
        if not memory_indices:
            continue
        first_memory = memory_indices[0]
        last_memory = memory_indices[-1]
        for index in memory_indices:
            resource_id = path_resources[index]
            if index == first_memory and index == last_memory:
                reserve, release = birth, death
            elif index == first_memory:
                reserve, release = birth, (transfer_events[index], "finish")
            elif index == last_memory:
                reserve, release = (transfer_events[index - 1], "start"), death
            else:
                reserve, release = (
                    (transfer_events[index - 1], "start"),
                    (transfer_events[index], "finish"),
                )
            # A buffer is owned by the movement that writes it: the birth buffer
            # (first memory) belongs to the value, so every reader shares it,
            # while a later buffer belongs to its inbound transfer, so
            # independent movements through one memory keep distinct buffers
            # instead of joining on a shared node.
            movement = None if index == first_memory else transfer_events[index - 1]
            buffer_key = (scopes, value_id, resource_id, movement)
            storage_specs.setdefault(buffer_key, []).append(
                (value_id, resource_id, size_bytes, reserve, release)
            )
            # Wire the store between the events that hand the value off: the
            # producer (or inbound hop) writes it, the outbound hop (or consumer)
            # reads it. A resolved window unions hops that share one buffer.
            predecessors, successors = storage_handoffs.setdefault(
                buffer_key, (set(), set())
            )
            if index == 0:
                if source_present:
                    predecessors.add(source_event)
            else:
                predecessors.add(transfer_events[index - 1])
            if index < len(path_resources) - 1:
                successors.add(transfer_events[index])
            elif destination_present:
                successors.add(destination_event)
    # Attach one pass-through storage node per physical buffer before any
    # scheduling. The node carries its anchored residency specs; its interval is
    # resolved after the hierarchy is composed. Removing the direct hand-off
    # edges makes the value flow producer -> storage -> consumer.
    scoped_value_memories = {
        (value_id, memory_id)
        for (scopes, value_id, memory_id, _), specs in storage_specs.items()
        if scopes
        for _, _, _, _, _ in specs
    }
    for (scopes, value_id, memory_id, movement), specs in storage_specs.items():
        if not scopes and (value_id, memory_id) in scoped_value_memories:
            continue
        size_bytes = max(spec[2] for spec in specs)
        if size_bytes <= 0:
            continue
        node_id = (
            f"storage:{_storage_scope_key(scopes)}:{value_id}@{memory_id}"
            f"{_storage_movement_suffix(movement)}"
        )
        event_graph.add_node(
            node_id,
            kind="storage",
            source_node_name=value_id,
            source_edge_id=None,
            resource_id=memory_id,
            duration=0.0,
            start_seconds=0.0,
            finish_seconds=0.0,
            size_bytes=size_bytes,
            scopes=scopes,
            movement=movement,
            anchors=tuple(specs),
        )
        predecessors, successors = storage_handoffs.get(
            (scopes, value_id, memory_id, movement), ((), ())
        )
        present_predecessors = [
            item for item in predecessors if event_graph.has_node(item)
        ]
        present_successors = [
            item for item in successors if event_graph.has_node(item)
        ]
        for predecessor in present_predecessors:
            for successor in present_successors:
                if event_graph.has_edge(predecessor, successor):
                    event_graph.remove_edge(predecessor, successor)
        for predecessor in present_predecessors:
            event_graph.add_edge(predecessor, node_id)
        for successor in present_successors:
            event_graph.add_edge(node_id, successor)

    # Greedy interval assignment: repeatedly start the earliest ready event on
    # its resources, then record the reserved interval on the event node.
    # `calendars` holds the reserved intervals per resource; `finishes` the
    # finish time of each already-placed event.
    calendars: dict[str, list[EventInterval]] = {}
    finishes: dict[str, float] = {}
    unscheduled = set(event_graph.nodes)
    while unscheduled:
        # Ready = every predecessor has been placed (sorted for determinism).
        ready = sorted(
            event_id
            for event_id in unscheduled
            if all(dep in finishes for dep in event_graph.predecessors(event_id))
        )
        if not ready:
            raise ValueError(
                "FX scheduling dependencies contain a cycle or missing activity"
            )
        event_id = ready[0]
        task = event_graph.nodes[event_id]
        # Storage is a passive hand-off: it orders the value's producer and
        # consumer but reserves no hardware resource. Its residency interval is
        # resolved from anchors after the hierarchy is composed.
        if task.get("kind") == "storage":
            resource_ids = ()
            duration = 0.0
        else:
            # Reserve every resource the task uses (an aggregate may use several).
            resource_ids = tuple(
                task.get("resources") or (str(task["resource_id"]),)
            )
            duration = float(task["duration"])
        predecessors = tuple(event_graph.predecessors(event_id))
        # Data-dependent earliest start is the latest predecessor finish.
        start = max((finishes[item] for item in predecessors), default=0.0)
        # Push later if any resource is busy, then reserve the chosen interval.
        interval = earliest_free_interval(calendars, resource_ids, start, duration)
        for resource_id in resource_ids:
            calendars.setdefault(resource_id, []).append(interval)
        finishes[event_id] = interval.finish_seconds
        task["start_seconds"] = interval.start_seconds
        task["finish_seconds"] = interval.finish_seconds
        unscheduled.remove(event_id)

    # Distinct scope paths present in the graph, ordered by axis order then tile
    # index so the tree is built deterministically.
    scope_paths = tuple(
        sorted(
            {
                data["scopes"]
                for _, data in event_graph.nodes(data=True)
                if data.get("scopes")
            },
            key=lambda path: tuple(
                (axis_order.index(scope.axis), scope.tile_index) for scope in path
            ),
        )
    )
    # Capacity of every memory that declares one. The modulo body folds periodic
    # residency onto the initiation interval against these; the whole-graph check
    # below enforces them on the composed timeline.
    memory_capacities = {
        resource_id: hardware.memory_capacity(resource_id)
        for resource_id in hardware.nodes_by_id()
        if hardware.is_memory(resource_id)
        and hardware.memory_capacity(resource_id) is not None
    }
    # Prepare and modulo-place every scope body. The hierarchy composes nested
    # scopes and uses each leaf's capacity-bounded placement, including
    # contracting leaves. Storage nodes are part of the body so the scheduler
    # sees the full pass-through chain.
    modulo_placements = {}
    for body_scope in scope_paths:
        body_graph = event_graph.subgraph(
            event_id
            for event_id, data in event_graph.nodes(data=True)
            if tuple(data["scopes"]) == body_scope
        )
        modulo_placements[body_scope] = schedule_modulo_event_body(
            body_graph,
            memory_capacities,
        )
    # Restrict the tree to axes that actually appear, dropping annotations that
    # only described an enclosing wrapper.
    represented_axes = {scope.axis for path in scope_paths for scope in path}
    represented_order = tuple(axis for axis in axis_order if axis in represented_axes)
    represented_repetitions = {axis: repetitions[axis] for axis in represented_order}
    event_scope_tree = build_fx_event_scope_tree(
        event_graph, represented_order, represented_repetitions
    )
    # Compose the scopes. Capacity is enforced by schedule_modulo_event_body for
    # periodic residency and by the resolved whole-graph check below, so the
    # hierarchy only orders events.
    hierarchical_schedule = schedule_hierarchical_fx_events(
        event_graph,
        event_scope_tree,
        modulo_placements,
    )
    if ledger is not None and scope_key is not None:
        ledger_key = scope_key
        # Record each compute event's timing so a nested child scope can be placed
        # on the global timeline and its residency extended to this scope's use.
        for event_id, data in event_graph.nodes(data=True):
            if data.get("kind") != "compute":
                continue
            successors = [
                event_graph.nodes[s] for s in event_graph.successors(event_id)
            ]
            predecessors = [
                event_graph.nodes[p] for p in event_graph.predecessors(event_id)
            ]
            finish = float(data["finish_seconds"])
            start = float(data["start_seconds"])
            ledger.record_wrapper(
                ledger_key,
                str(data["source_node_name"]),
                start,
                max((float(d["finish_seconds"]) for d in successors), default=finish),
                min((float(d["start_seconds"]) for d in predecessors), default=start),
                bool(successors),
            )
    # Resolve each pass-through storage node's anchored residency against the
    # composed absolute timeline. The node already exists in the dataflow; only
    # its lifetime and byte count are filled in here.
    resolved_event_times = {
        event_id: (float(data["start_seconds"]), float(data["finish_seconds"]))
        for event_id, data in event_graph.nodes(data=True)
    }
    resolved_storage: list[
        tuple[str, str, int, float, float, tuple[TilingScope, ...], str | None]
    ] = []
    for node_id, data in event_graph.nodes(data=True):
        if data.get("kind") != "storage":
            continue
        resolved = resolve_storage_windows(
            data.get("anchors", ()),
            resolved_event_times,
            hierarchical_schedule.makespan_seconds,
        )
        if not resolved:
            continue
        value_id, memory_id, size_bytes, reserve, release = resolved[0]
        data["start_seconds"] = reserve
        data["finish_seconds"] = release
        data["duration"] = release - reserve
        data["size_bytes"] = size_bytes
        resolved_storage.append(
            (
                value_id,
                memory_id,
                size_bytes,
                reserve,
                release,
                tuple(data.get("scopes", ())),
                data.get("movement"),
            )
        )
    capacity_storage = list(resolved_storage)
    # Global peak per memory, measured from the resolved storage nodes.
    peak_memory = periodic_capacity_peak(
        (memory_id, size_bytes, reserve, release)
        for _, memory_id, size_bytes, reserve, release, _, _ in capacity_storage
    )
    if ledger is not None and scope_key is not None:
        # Record this scope's residency for the outermost call to resolve globally.
        aliases: dict[str, object] = {}
        for sink_id, data in graph.nodes(data=True):
            if not sink_id.startswith("sink:") or not data.get("alias"):
                continue
            inputs = data.get("alias_inputs")
            if inputs:
                target_scope = data.get("alias_scope", scope_key)
                aliases[sink_id.removeprefix("sink:")] = tuple(
                    (target_scope, value) for value in inputs
                )
                continue
            if (
                data.get("alias_scope") is None
                or data.get("alias_value") is None
            ):
                continue
            aliases[sink_id.removeprefix("sink:")] = (
                data["alias_scope"],
                data["alias_value"],
            )
        view_aliases = {
            source_id.removeprefix("source:"): data["alias_value"]
            for source_id, data in graph.nodes(data=True)
            if source_id.startswith("source:")
            and data.get("alias_value") is not None
        }
        ledger.record_scope(
            scope_key,
            list(resolved_storage),
            {n.split(":", 1)[1] for n in graph.nodes if n.startswith("source:")},
            {n.split(":", 1)[1] for n in graph.nodes if n.startswith("sink:")},
            aliases=aliases,
            view_aliases=view_aliases,
        )
    # Report a zero peak for memories with no residency, matching the check.
    for resource_id in hardware.nodes_by_id():
        if hardware.is_memory(resource_id):
            peak_memory.setdefault(resource_id, 0)
    hierarchical_schedule = replace(
        hierarchical_schedule, peak_memory_bytes=peak_memory
    )
    if trace is not None:
        # Surface the resolved storage lifetimes and modulo placements that the
        # schedule otherwise keeps private, for downstream visualization.
        trace.record_scheduled(
            scope_path,
            module_axis,
            event_graph,
            hierarchical_schedule,
            modulo_placements,
        )
    if validate_memory_capacity:
        for resource_id, peak, capacity in capacity_violations(
            peak_memory, memory_capacities
        ):
            sizes: dict[str, int] = {}
            for value_id, memory_id, size_bytes, *_ in capacity_storage:
                if memory_id == resource_id:
                    sizes[value_id] = max(sizes.get(value_id, 0), size_bytes)
            contributors = tuple(
                sorted(sizes.items(), key=lambda item: item[1], reverse=True)
            )
            raise MemoryCapacityError(
                f"memory {resource_id!r} capacity exceeded: {peak} > {capacity}",
                memory_id=resource_id,
                peak_bytes=peak,
                capacity_bytes=capacity,
                required_scale=peak / capacity,
                contributors=contributors,
                kind="intra_iteration",
            )
    return event_graph, hierarchical_schedule


def _scope_paths(
    graph: nx.MultiDiGraph,
) -> tuple[dict[str, tuple[TilingScope, ...]], tuple[str, ...], Mapping[str, int]]:
    """Collect the structural scope paths carried by the placed task graph.

    Task nodes carry ``scope_path``: an ordered tuple of :class:`TilingScope`
    built by the mapper from the enclosing module structure and each branch's
    local tile index. This reads them back and derives the global axis order and
    each axis's tile factor.

    Returns:
        ``(scopes_by_node, axis_order, repetitions)`` where ``scopes_by_node``
        maps a task id to its ordered ``TilingScope`` path, ``axis_order`` is the
        first-seen axis order, and ``repetitions`` maps each axis to its tile
        factor. Raises ``ValueError`` on inconsistent factors for one axis.
    """
    # Task id -> its ordered scope path, plus the global axis order and
    # per-axis tile factors discovered while scanning.
    scopes_by_node: dict[str, tuple[TilingScope, ...]] = {}
    axis_order: list[str] = []
    repetitions: dict[str, int] = {}
    for node_id, data in graph.nodes(data=True):
        scopes = tuple(data.get("scope_path", ()))
        # Nodes with no structural scope (e.g. wrapper aggregates) are omitted.
        if not scopes:
            continue
        scopes_by_node[node_id] = scopes
        for scope in scopes:
            # First sighting fixes this axis's position in the global order.
            if scope.axis not in axis_order:
                axis_order.append(scope.axis)
            # Every node on one axis must agree on the tile factor.
            existing = repetitions.setdefault(scope.axis, scope.factor)
            if existing != scope.factor:
                raise ValueError(f"inconsistent tiling factors for axis {scope.axis!r}")
    return scopes_by_node, tuple(axis_order), MappingProxyType(repetitions)


def _storage_scope_key(scopes: tuple[TilingScope, ...]) -> str:
    """Compact, deterministic key for a storage node's owning scope.

    The owning scope is part of every storage node id, so two tile clones that
    produce the same value on the same memory keep distinct buffers.
    """
    if not scopes:
        return "root"
    return "|".join(f"{scope.axis}{scope.tile_index}" for scope in scopes)


def _storage_movement_suffix(movement: str | None) -> str:
    """Node-id suffix naming the movement that owns a non-birth buffer.

    The birth buffer (``None``) keeps the bare id because every reader shares it;
    each later buffer is named by the transfer event that writes it.
    """
    if movement is None:
        return ""
    return "#" + movement.replace(":", ".")
