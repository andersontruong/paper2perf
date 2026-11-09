"""Public API for the dataflow mapper.

The mapper consumes Parabolic's FX/``ParallelModel`` application graph and the
hardware model (``hw_model``), and owns placement, routing, and scheduling.
Exports are resolved lazily so importing a single module does not drag in the
rest.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "MappingResult": "dataflow_mapper.execution",
    "fx_task_graph": "dataflow_mapper.placement",
    "map_fx_graph": "dataflow_mapper.execution",
    "place_fx_graph": "dataflow_mapper.placement",
    "select_routes": "dataflow_mapper.routing",
}

__all__ = [
    "MappingResult",
    "fx_task_graph",
    "map_fx_graph",
    "place_fx_graph",
    "select_routes",
]


def __getattr__(name: str):
    """Import an exported symbol from its owning module on first access."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
