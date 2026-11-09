# dataflow_mapper

`dataflow_mapper` maps an application dataflow onto a hardware graph and
estimates performance. It consumes two canonical models and owns the mapping
stages:

```text
Parabolic ParallelModel / FX graph (application semantics and transformations)
hw_model                             (compute, memories, links, capacities)
                 \                  /
                   dataflow_mapper
               placement -> routing -> scheduling
                           |
                      MappingResult
```

The mapper does not define application or hardware classes; it imports them
from `parabolic` and `hw_model`.

## Stages

- **placement** (`placement.py`) — *where* FX compute operations and boundary
  values go. `place_fx_graph(graph_module, hardware, bindings, options=...)`
  derives the task graph from FX with `fx_task_graph`, costs each task per
  resource from its FLOP requirements (gated on hardware `flop_capability`), and
  assigns resources with the HEFT engine. `call_module` wrappers are aggregate
  tasks costed by their mapped child scope's makespan.
- **routing** (`routing.py`) — *how* data moves. `select_routes(graph, hardware,
  options=...)` annotates every task-graph edge with its chosen hardware path:
  `route` (link ids), `transfer_bytes`, and `seconds`. A
  movement's endpoints are the true source and destination resources — for an
  aggregate `call_module`, the memory its child reads or writes the value at,
  not a representative compute resource.
- **scheduling** (`scheduling/`) — *when* compute, transfer and storage events
  occur. `schedule_fx_graph(graph, hardware)` lowers the placed/routed task graph
  to a physical event graph (`fx.py`), greedily assigns intervals (`model.py`),
  places leaf scopes with modulo scheduling (`modulo.py`), and composes the
  scope tree into a makespan (`hierarchical.py`). Each mapping scope composes
  its own subtree with `global_storage.py`: windows stay keyed by scope, so a
  child's reservation contributes to its parent's footprint. Composed view
  aliases resolve to one physical buffer.
- **orchestration** (`execution.py`) — `map_fx_graph` composes placement, routing
  and scheduling for one scope, recursing into nested `call_module` scopes and
  returning `MappingResult`. `map_with_capacity_retiling` consumes a
  factorless `apply_tiling_plan` template, starts each split at factor 2
  subject to a per-axis `min_tile_size`, and retunes the factors on
  `MemoryCapacityError`. A global axis sequence may repeat an axis; each
  occurrence is a recursive split. Sibling `call_module` sites are tiled
  independently and are never merged into one tiling scope. Factors are shared
  across sibling scopes with the same `(axis, split)` key.
  Tiled regions are body-only: a plain outer join scope wraps a tiled scope
  whose entry slices feed one representative body call. The call target is a
  leaf body, or the next outer scope when an intermediate forwarding body was
  collapsed. Concretization replicates that call and rebuilds the join.
  Displayed scope IDs use `outer-`, `tiled-`, and `body-` role prefixes.
- **scope keys** — scopes are identified by a path-only `tuple[TilingScope |
  str, ...]`: entering a `call_module` appends its wrapper name and entering a
  tile replica appends the tile coordinate. The coordinate includes the split
  index, so repeated splits of the same axis remain distinct. The module's own
  tiling axis is not part of the key, so plain outer/body scopes remain distinct.
- **HEFT engine** (`heft.py`) — the `Mapper` interface and `HEFTMapper`
  list-scheduler used by `placement.py`. Placement consumes only the chosen
  resource from HEFT's start-time matrix.
- **boundary pins** — the root scope infers the outermost memory from hardware
  topology (memories with no compute neighbor, else the memories farthest from
  compute) and pins the model's inputs and outputs there; nested scopes thread
  their parent's pins through placement and routing internally. A non-root
  single-output wrapper keeps its boundary sink but aliases its child's storage:
  the sink emits no transfer and capacity counts the buffer once.
- **storage lifetimes** — storage is a pass-through node in the event graph, not
  a post-scheduling annotation. It orders the producer/transfer/consumer chain
  but reserves no resource. Its anchored lifetime is resolved after the
  hierarchy is composed. A produced value is born at its producer's start; an
  external value with an in-scope transfer is born at that transfer's start; an
  external value with neither is born at its first consumer's start. A value
  dies at its last consumer finish or last outbound transfer. First-use
  anchoring preserves child-iteration offsets chosen by modulo scheduling.
- **view materialization** — a proper view of one external base gets a
  base-to-view materialization edge. The view's byte count crosses into the
  memory that holds the slice; the base remains at its inherited boundary
  memory. An unscoped boundary window that duplicates a scoped tile window is
  an alias, not another allocation. Cross-scope view aliases collapse to the
  physical buffer they describe, while a materialized slice keeps its own
  transfer birth rather than inheriting the outer base's lifetime.
- **capacity escalation** — mapping construction owns boundary pins,
  materialization, alias normalization, and intra-iteration ordering. Mapping
  validates local residency and composes each scope's subtree with its children.
  Only three repairs exist: **remap** (mapping reorders or promotes and retries),
  **II growth** (modulo separates successive iterations), and **retiling**
  (orchestration grows parfactors). `MemoryCapacityError.kind` records
  `structural`, `intra_iteration`, `inter_iteration`, or `cross_scope` so a
  caller can tell which repair can help. II cannot repair same-body overlap;
  retiling is the final fallback.
- **visualization** (`viz/`) — renders a self-contained HTML report from the
  artifacts a `MappingTrace` collects. `adapters.py` converts FX, task, and event
  graphs into the display schema (`schema.py`); `graphviz.py` and `gantt.py`
  render scope graphs and Gantt charts; `html.py` assembles the report shell.

## Public API

```python
from dataflow_mapper.execution import map_with_capacity_retiling

tiled = parallel_model.apply_tiling_plan(("M", "M", "N", "K"))
concrete, parfactors, result = map_with_capacity_retiling(
    tiled, hardware, bindings, min_tile_size={"M": 4, "N": 4, "K": 4}
)
print(parfactors, result.latency)  # e.g. {"M@0": 2, "M@1": 2, ...}
```

`map_fx_graph` maps a model whose tiled scopes already have selected factors.

`place_fx_graph`, `select_routes`, and `fx_task_graph` are available for
stage-level inspection. `map_with_capacity_retiling` (in
`dataflow_mapper.execution`) consumes a factorless `apply_tiling_plan` template,
chooses initial factors per split (shared across sibling scopes), maps fresh
copies while retuning them, and returns the concretized model, the selected
`{axis@split: factor}` map, and the `MappingResult`. The template remains
factorless; the tiling plan itself is hardware-agnostic, and the per-axis
minimum tile size is supplied by the mapper.

## Inputs and outputs

- Canonical input: Parabolic's `ParallelModel` FX graph, an
  `hw_model.model.HardwareGraph`, and concrete symbol bindings.
- Output: `MappingResult`, containing the placed and routed task graph, the
  hierarchical schedule makespan as `latency`, and the reserved resources.

## Visualization

`MappingTrace` records each scope's placed graph, routed graph, event graph,
schedule, and storage windows while mapping. The `viz` package turns that into a
`Report` of `Section`s — hardware, per-phase dataflow graphs, event graphs, and
Gantt charts — and `write_report` emits one self-contained HTML file. See
`examples/linear.py` for the end-to-end flow.

## Tests

```bash
python -m pytest tests/mapper
```

Tests live in `tests/mapper/`: `test_heft.py`,
`test_capability_costs.py`, `test_capacity_scheduling.py`,
`test_global_storage.py`, `test_storage_liveness.py`, and
`test_linear_example.py`. The Linear suite pins the body-only outer/tiled/body
scope structure, independent sibling sites, recursive repeated-axis splits,
shared split factors, minimum tile sizes, path-only keys with split-aware
coordinates, multi-output boundary metadata, and retuning.
