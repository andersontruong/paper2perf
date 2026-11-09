"""NetworkX-backed hardware topology shared by the software model and mapper.

A hardware topology is an attribute ``networkx`` graph wrapped by
:class:`HardwareGraph`. There are no resource or link classes; nodes and edges
are plain attribute mappings documented in :mod:`hw_model.model`.
"""

from hw_model.model import (
    BANDWIDTH_KEY,
    CAPABILITY_KEY,
    CAPACITY_KEY,
    COMPUTE_KIND,
    KIND_KEY,
    LATENCY_KEY,
    MEMORY_KIND,
    NAME_KEY,
    THROUGHPUT_KEY,
    HardwareEdge,
    HardwareGraph,
    link_name,
)

__all__ = [
    "BANDWIDTH_KEY",
    "CAPABILITY_KEY",
    "CAPACITY_KEY",
    "COMPUTE_KIND",
    "KIND_KEY",
    "LATENCY_KEY",
    "MEMORY_KIND",
    "NAME_KEY",
    "THROUGHPUT_KEY",
    "HardwareEdge",
    "HardwareGraph",
    "link_name",
]
