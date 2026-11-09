# Parabolic: PyTorch Compiler for Symbolic Parallelism

## Install
Only CPU version is needed for modeling:
```
pip install . --extra-index-url=https://download.pytorch.org/whl/cpu
```

## Discovering Parallelizable Regions

Before manually inserting slice/cat pairs and hoisting, you can **discover** all subgraphs where a parallelization factor (parfactor) can propagate. This uses the same dimension-propagation rules as `hoist_parallel_slices` (upstream) and `lower_parallel_cats` (downstream) but only analyzes — it does not rewrite the graph.

A **parallelizable region** is a maximal connected subgraph where a parfactor on a given dimension can propagate through every op. Each node in the region tracks which local dimension carries the parfactor (since dimension indices change through transpose, permute, etc.).

- **Tight region**: The parfactor propagates only through non-contracting operations (elementwise, transpose, permute, view, addmm M/N, bmm batch/M/N). No extra cost.
- **Extended region**: The parfactor also crosses contracting boundaries (matmul K-dimension), which requires an all-reduce after each contracting op. The extended region is a superset of the tight region and carries a `contracting_volume` cost.

### Example: Discovering Regions in a Transformer

```python
import sympy as sp
from parabolic.parallel_model import ParallelModel

# 1. Build and annotate the model
dim_per_head = sp.symbols('dim_per_head', positive=True, integer=True)
num_heads    = sp.symbols('num_heads',    positive=True, integer=True)
embed_dim    = num_heads * dim_per_head
batch_size   = sp.symbols('batch_size',   positive=True, integer=True)
sequence     = sp.symbols('sequence',     positive=True, integer=True)

model = ParallelModel(
    Transformer,
    symbols={'dim_per_head': dim_per_head, 'num_heads': num_heads, 'embed_dim': embed_dim},
    batch_shape={'batch_size': batch_size, 'sequence': sequence}
)
gm = model.annotate(('sequence', 'batch_size', 'embed_dim'))
gm = model.propagate_attr_metadata_pass()
gm = model.fuse_unsqueeze_transpose_squeeze()
gm = model.fuse_transpose_chains()
gm = model.fuse_parallel_sharding_to_slices()
gm = model.prune_sym_size_chains()

# 2. Discover all regions, organized by parfactor symbol
regions_by_sym = model.find_all_regions()
for sym_name, regions in regions_by_sym.items():
    print(f'{sym_name}: {len(regions)} regions')
    for r in regions:
        print(f'  {r.region_id}: {len(r.dim_labels)} nodes')

# 3. Or target a specific symbol (e.g. which ops can be parallelized along num_heads?)
nh_regions = model.find_regions_for_symbol('num_heads')
for r in nh_regions:
    print(f'{r.region_id}: dims = {r.dim_labels}')

# 4. Visualize regions per symbol
from parabolic.viz import plot_regions, plot_all_regions
for sym_name, regions in regions_by_sym.items():
    plot_regions(gm, regions, f'figures/regions_{sym_name}.png')

# 5. Combined visualization of all symbols on one graph
plot_all_regions(gm, regions_by_sym, 'figures/regions_all.png')

# 6. Annotate per-region cost metrics (compute / memory / communication)
from parabolic.region_costs import annotate_all_region_costs
parfactor_for = lambda name: sp.Symbol(f'{name}_par0', positive=True, integer=True)
costs = annotate_all_region_costs(gm, regions_by_sym, parfactor_for, dtype=torch.float16)

# 7. Visualizations with cost legend / per-box compute summary
for sym_name, regions in regions_by_sym.items():
    plot_regions(
        gm, regions, f'figures/regions_{sym_name}_costs.png',
        show_costs_in_legend=True, show_costs_in_box=True,
    )
plot_all_regions(
    gm, regions_by_sym, 'figures/regions_all_costs.png',
    show_costs_in_legend=True,
)
```

### API

| Method | Description |
|--------|-------------|
| `find_all_regions()` | Discover all free symbols in node shapes and return `dict[str, list[ParallelRegion]]` mapping symbol name to its regions. |
| `find_regions_for_symbol(name)` | Seed from all nodes whose output shape contains the named symbol. Returns disjoint regions where the parfactor can propagate. |
| `annotate_region_costs(gm, region, parfactor, dtype=fp16)` | Populate the region's `compute_speedup`, `memory_bytes`, `comm_bytes`, and `parfactor` fields with SymPy expressions. Returns a `RegionCosts` instance. |
| `annotate_all_region_costs(gm, regions_by_sym, parfactor_for, dtype=fp16)` | Convenience: annotate every region. `parfactor_for(name) -> sp.Expr` chooses the parfactor expression per symbol. |
| `plot_regions(gm, regions, file, show_costs_in_legend=False, show_costs_in_box=False)` | Visualize one symbol's regions. The cost flags are no-ops when costs haven't been annotated. |
| `plot_all_regions(gm, regions_by_sym, file, show_costs_in_legend=False, show_costs_in_box=False)` | Combined visualization of all symbols' regions on one graph, color-coded per symbol. |

### `ParallelRegion` Fields

| Field | Type | Description |
|-------|------|-------------|
| `region_id` | `str` | Human-readable ID including the parfactor, e.g. `"R0(batch_size)"`, `"R3(dim_per_head*num_heads)_ext"` |
| `dim_labels` | `dict[str, int]` | Maps each FX node name to the local dimension carrying the parfactor |
| `nodes` | `set[str]` | Convenience property: set of node names in the region |
| `contracting_ops` | `list[str]` | Node names where an all-reduce is needed (empty for tight regions) |
| `contracting_volume` | `int` | Total elements needing all-reduce (0 for tight regions) |
| `seed_node_name` | `str` | Name of the BFS seed node |
| `seed_dim` | `int` | Dimension index at the seed node |
| `seed_dim_expr` | `str` | Symbolic expression at that dimension (e.g. `"batch_size"`, `"dim_per_head*num_heads"`) |
| `compute_speedup` | `list[sp.Expr]` or `None` | `[alu, exp, mac]` SymPy vector — aggregated `flop_req` divided by `parfactor`, with `(N-1) * out_numel` added to alu for each contracting op. `None` until annotated. |
| `memory_bytes` | `sp.Expr` or `None` | Max over ops of `(input + output) bytes`. `None` until annotated. |
| `comm_bytes` | `sp.Expr` or `None` | Max over ops of `max(fan-in, fan-out)` bytes. **This is a fan-in/fan-out bandwidth proxy, not all-reduce traffic** — see `annotate_collective_latency` for that. `None` until annotated. |
| `parfactor` | `sp.Expr` or `None` | The parfactor expression `N` used as the denominator for `compute_speedup`. |

### Region Costs

`annotate_region_costs` (and the `annotate_all_region_costs` convenience wrapper) attaches three cost metrics to each `ParallelRegion`. All metrics stay symbolic (SymPy expressions) so consumers can substitute concrete values when they have a hardware mapping.

| Metric | What it represents |
|--------|--------------------|
| `compute_speedup` | Per-worker FLOP work after parallelizing by `N`. A vector `[alu, exp, mac]` matching `flop_req` axes. For contracting (split-K) regions, the `alu` axis includes a `(N-1) * output_numel` term per contracting op to model the implicit reduction. |
| `memory_bytes` | The largest per-op `(input + output) bytes` in the region. Default dtype is FP16 (2 bytes/element); pass `dtype=torch.float32` etc. to override. |
| `comm_bytes` | The largest per-op `max(fan-in bytes, fan-out bytes)` in the region. Reflects bandwidth pressure at the busiest op, not all-reduce traffic. |

### Propagation Rules

Starting from a seed `(node, dim)`, a bidirectional BFS expands the region:

| Op | Upstream (hoist) | Downstream (lower) |
|----|-----------------|-------------------|
| clone, add, mul, gelu | Propagates (right-aligned broadcasting) | Propagates |
| transpose, permute, t | Dimension remapped through permutation | Inverse permutation |
| view, reshape | Propagates if matching dimension found by size, with symbol-aware fallback when the parfactor appears in exactly one dim | Same |
| slice.Tensor | Propagates on non-sliced dims; symbol-aware fallback allows propagation on the sliced dim when the parfactor symbol is preserved on both sides | Same |
| addmm (M/N) | Propagates to A or B on the non-contracting dim | Propagates |
| addmm (K) | Blocks | Column-parallel switch: tight region passes through to output's N dim (no `contracting_ops`); a duplicated extended region also emits with the matmul in `contracting_ops` |
| bmm (batch/M/N) | Propagates | Propagates |
| bmm (K) | Blocks | Same column-parallel switch + duplicated extended region |
| softmax | Blocks on reduction dim; propagates on others | Same |
| native_layer_norm | Blocks on normalized dims; propagates on batch/seq | Same |

`find_regions_for_symbol` returns both the tight super-region (no `contracting_ops`) and the duplicated extended region (`contracting_ops` populated with K-contracting matmul names) for any region whose BFS crossed a K boundary. The tight region corresponds to column-parallel routing (no reduction); the extended region corresponds to row-parallel routing (split K + all-reduce). Cost annotation handles each appropriately.

See `tests/transformer_region/transformer.py` for the full end-to-end example.

## Running Tests

```bash
# All tests
pytest tests/ -s --disable-warnings

# Slice/cat identity insertion test
pytest tests/test_slice_cat.py -s --disable-warnings

# Slice hoisting tests (one per supported op: clone, add, mul, transpose, permute, view, addmm, bmm)
pytest tests/test_slice_hoist.py -s --disable-warnings

# Cat lowering tests (including contracting mm/addmm/bmm)
pytest tests/test_concat_lower.py -s --disable-warnings

# Parallelizable region finding
pytest tests/test_regions.py -s --disable-warnings

# Region cost annotation
pytest tests/test_region_costs.py -s --disable-warnings
```

```bash
# End-to-end transformer region discovery (run from tests/transformer_region/)
cd tests/transformer_region && python -m pytest transformer.py -s --disable-warnings
```
