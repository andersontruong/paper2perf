"""HEFT list scheduling over a heterogeneous processor graph.

Ports the Heterogeneous Earliest Finish Time heuristic (Topcuoglu et al.) that
the mapper uses as its placement solver. Costs are seconds. Communication is
supplied as explicit per-pair transfer times, which routing computes from the
hardware graph before placement runs.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
import scipy


class Mapper(ABC):
    """Placement-engine contract: schedule a task DAG onto processor slots.

    Implementations return an ``(tasks, resources)`` start-time matrix holding
    each task's start time on each resource, or ``inf`` where the task is not
    placed. Communication cost is a scheduling input rather than an optional
    post-pass: the heuristic needs it while choosing each task's resource.
    """

    @abstractmethod
    def map(
        self,
        dependencies: scipy.sparse.spmatrix,
        computation_cost: npt.NDArray,
        communication_costs: Mapping[tuple[int, int, int, int], float],
    ) -> npt.NDArray:
        """Place every task and return its `(tasks, resources)` start times."""
        raise NotImplementedError


class HEFTMapper(Mapper):
    """Heterogeneous Earliest Finish Time list scheduler.

    Schedules a task DAG over processors with explicit directed communication
    times.
    """

    def map(
        self,
        dependencies: scipy.sparse.spmatrix,
        computation_cost: npt.NDArray,
        communication_costs: Mapping[tuple[int, int, int, int], float],
    ) -> npt.NDArray:
        """Schedule a DAG using explicit directed communication times.

        ``communication_costs`` maps ``(source_task, destination_task,
        source_resource, destination_resource)`` to seconds. Missing entries
        are unreachable placement pairs.
        """
        costs = np.asarray(computation_cost, dtype=float).copy()
        _validate_explicit_costs(dependencies, costs, communication_costs)
        return _schedule_heft(
            dependencies,
            costs,
            lambda task1, task2, proc1, proc2: communication_costs.get(
                (task1, task2, proc1, proc2), np.inf
            ),
        )


def _schedule_heft(
    dependencies: scipy.sparse.spmatrix,
    computation_cost: npt.NDArray,
    communication_cost: Callable[[int, int, int, int], float],
) -> npt.NDArray:
    """List-schedule a DAG with classic HEFT.

    Tasks are ordered by decreasing upward rank (average computation cost plus the
    average communication cost of each outgoing edge plus its successor's rank),
    with a topological order breaking ties. Each task is then assigned to the
    resource that minimizes its finish time, including communication from
    already-placed predecessors. Returns a `(tasks, resources)` start-time matrix
    with `inf` for unmapped pairs. Raises ValueError for cyclic DAGs, tasks with no
    finite cost, or unreachable resources.

    Only the chosen resource travels onward: placement reads which entry of the
    returned matrix is finite and discards the start times. HEFT's insertion
    (gap-filling) refinement therefore buys nothing here -- it only improves the
    times that nobody consumes -- so it is deliberately not implemented.
    """
    dependency_matrix = dependencies.toarray().astype(bool)
    costs = np.asarray(computation_cost, dtype=float).copy()
    task_count, resource_count = costs.shape
    if np.any(np.all(np.isinf(costs), axis=1)):
        raise ValueError("every task must have at least one finite computation cost")

    normalized_costs = np.where(np.isinf(costs), np.nan, costs)
    average_cost = np.nanmean(normalized_costs, axis=1)
    predecessors = {
        task: tuple(dependency_matrix[:, task].nonzero()[0])
        for task in range(task_count)
    }
    successors = {
        task: tuple(dependency_matrix[task, :].nonzero()[0])
        for task in range(task_count)
    }
    average_communication = _average_communication(
        successors, communication_cost, resource_count
    )
    ranks = _upward_ranks(average_cost, successors, average_communication)
    topological_index = {
        task: index
        for index, task in enumerate(_topological_order(predecessors, successors))
    }
    task_order = sorted(
        range(task_count), key=lambda task: (-ranks[task], topological_index[task])
    )
    starts = np.full((task_count, resource_count), np.inf, dtype=float)
    available = np.zeros(resource_count, dtype=float)
    finish = np.zeros((task_count, resource_count), dtype=float)
    assigned_resource = np.full(task_count, -1, dtype=int)
    for task in task_order:
        best: tuple[float, int] | None = None
        for resource in range(resource_count):
            if np.isnan(normalized_costs[task, resource]):
                continue
            ready = 0.0
            for predecessor in predecessors[task]:
                predecessor_resource = assigned_resource[predecessor]
                if predecessor_resource < 0:
                    raise ValueError("HEFT predecessor was not assigned exactly once")
                ready = max(
                    ready,
                    finish[predecessor, predecessor_resource]
                    + communication_cost(
                        predecessor,
                        task,
                        predecessor_resource,
                        resource,
                    ),
                )
            finish_time = max(available[resource], ready) + costs[task, resource]
            candidate = (finish_time, resource)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise ValueError(f"task {task} has no schedulable resource")
        finish_time, resource = best
        if not np.isfinite(finish_time):
            raise ValueError(f"task {task} has no reachable resource")
        starts[task, resource] = finish_time - costs[task, resource]
        finish[task, resource] = finish_time
        available[resource] = finish_time
        assigned_resource[task] = resource
    return starts


def _validate_explicit_costs(
    dependencies: scipy.sparse.spmatrix,
    computation_cost: npt.NDArray,
    communication_costs: Mapping[tuple[int, int, int, int], float],
) -> None:
    """Reject explicit costs with malformed keys or non-finite/negative values."""
    if dependencies.shape[0] != dependencies.shape[1]:
        raise ValueError("dependencies must be square")
    if computation_cost.ndim != 2 or computation_cost.shape[0] != dependencies.shape[0]:
        raise ValueError("computation_cost must have one row per task")
    task_count, resource_count = computation_cost.shape
    if any(
        len(key) != 4
        or key[0] < 0
        or key[0] >= task_count
        or key[1] < 0
        or key[1] >= task_count
        or key[2] < 0
        or key[2] >= resource_count
        or key[3] < 0
        or key[3] >= resource_count
        or seconds < 0
        or not np.isfinite(seconds)
        for key, seconds in communication_costs.items()
    ):
        raise ValueError(
            "communication costs must have valid indices and finite nonnegative values"
        )


def _average_communication(
    successors: dict[int, tuple[int, ...]],
    communication_cost: Callable[[int, int, int, int], float],
    resource_count: int,
) -> dict[tuple[int, int], float]:
    """Average transfer time per dependency, over the reachable resource pairs.

    HEFT's upward rank weights each successor by the *mean* communication time of
    the edge, not by zero. Explicit cost tables only define reachable
    (source task, destination task, source resource, destination resource)
    quadruples, so the mean is taken over the pairs the table defines; an edge
    with no reachable pair averages to 0, matching HEFT's same-resource case.
    """
    averages: dict[tuple[int, int], float] = {}
    for task, children in successors.items():
        for child in children:
            total = 0.0
            defined = 0
            for source in range(resource_count):
                for destination in range(resource_count):
                    seconds = communication_cost(task, child, source, destination)
                    if np.isfinite(seconds):
                        total += seconds
                        defined += 1
            averages[(task, child)] = total / defined if defined else 0.0
    return averages


def _upward_ranks(
    average_cost: npt.NDArray,
    successors: dict[int, tuple[int, ...]],
    average_communication: Mapping[tuple[int, int], float],
) -> npt.NDArray:
    """HEFT upward rank: ``wbar_i + max_j(cbar_ij + rank_u(j))`` over successors.

    Computed by peeling successors in reverse topological order, so it doubles as a
    cycle check (raises ValueError when the DAG has a cycle).
    """
    ranks = np.zeros(len(average_cost), dtype=float)
    pending = {task: len(children) for task, children in successors.items()}
    ready = [task for task, count in pending.items() if count == 0]
    predecessors: dict[int, list[int]] = {task: [] for task in successors}
    for task, children in successors.items():
        for child in children:
            predecessors[child].append(task)
    while ready:
        task = ready.pop()
        ranks[task] = average_cost[task] + max(
            (
                average_communication[(task, child)] + ranks[child]
                for child in successors[task]
            ),
            default=0.0,
        )
        for predecessor in predecessors[task]:
            pending[predecessor] -= 1
            if pending[predecessor] == 0:
                ready.append(predecessor)
    if any(pending.values()):
        raise ValueError("HEFT requires an acyclic task graph")
    return ranks


def _topological_order(
    predecessors: dict[int, tuple[int, ...]],
    successors: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """Kahn topological order with ascending task-id tie-breaks.

    A deterministic order keeps HEFT rankings reproducible across runs. Raises
    ValueError when the graph has a cycle.
    """
    pending = {task: len(parents) for task, parents in predecessors.items()}
    ready = sorted(task for task, count in pending.items() if count == 0)
    order: list[int] = []
    while ready:
        task = ready.pop(0)
        order.append(task)
        for child in successors[task]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
        ready.sort()
    if len(order) != len(predecessors):
        raise ValueError("HEFT requires an acyclic task graph")
    return tuple(order)
