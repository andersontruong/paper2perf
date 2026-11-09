"""Untrusted model responses are parsed at every graph boundary."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Citation(StrictModel):
    chunk_id: str
    quote: str = Field(min_length=8, max_length=1600)


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    citations: list[Citation] = Field(min_length=1, max_length=6)


class Answer(StrictModel):
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=8)


class Intent(StrictModel):
    action: Literal["question", "simulate", "inspect"]
    query: str = Field(min_length=1, max_length=1200)


class Fact(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    parameter: str
    value: float = Field(gt=0)
    unit: str
    scope: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    precision: Literal["fp16", "fp32", "unspecified"] = "unspecified"
    mode: Literal["dense", "sparse", "unspecified"] = "unspecified"
    aggregation: Literal["device", "component", "unspecified"] = "unspecified"
    citations: list[Citation] = Field(min_length=1, max_length=6)


class Relationship(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    source: str
    target: str
    kind: Literal["connects", "contains", "shares_resource"]
    variant: str
    citations: list[Citation] = Field(min_length=1, max_length=6)


class Extraction(StrictModel):
    facts: list[Fact] = Field(default_factory=list, max_length=60)
    relationships: list[Relationship] = Field(default_factory=list, max_length=40)
    missing: list[str] = Field(default_factory=list, max_length=20)
    conflicts: list[str] = Field(default_factory=list, max_length=20)


class Quantity(StrictModel):
    value: float = Field(gt=0)
    unit: Literal["byte", "byte/s", "FLOP/s"]
    basis: Literal["documented", "derived", "assumed"]
    fact_ids: list[str] = Field(default_factory=list)
    scale: float = Field(default=1, gt=0)
    rationale: str = Field(min_length=1)
    assumption_id: str | None = None
    derivation: Literal["unit_conversion", "aggregate_capacity", "per_clock_rate"] = "unit_conversion"


class Memory(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    capacity: Quantity


class Compute(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    kind: Literal["matrix", "vector"]
    throughput: Quantity


class Link(StrictModel):
    source: str
    target: str
    bandwidth: Quantity
    relationship_id: str
    relationship_ids: list[str] = Field(default_factory=list, max_length=24)
    topology_basis: Literal["documented", "containment_access", "containment_projection", "shared_memory_projection"] = "documented"


class ResourceBinding(StrictModel):
    source_resource: str
    memory_resource: str
    containment_id: str


class PhysicalMemoryPool(StrictModel):
    memory_id: str
    member_resource_ids: list[str] = Field(min_length=2, max_length=8)
    relationship_ids: list[str] = Field(min_length=1, max_length=12)
    capacity_fact_id: str


class Hardware(StrictModel):
    name: str
    variant: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1, max_length=12)
    memories: list[Memory] = Field(min_length=1, max_length=12)
    computes: list[Compute] = Field(min_length=1, max_length=8)
    links: list[Link] = Field(min_length=1, max_length=24)
    resource_bindings: list[ResourceBinding] = Field(default_factory=list, max_length=12)
    memory_pools: list[PhysicalMemoryPool] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def topology(self):
        ids = [x.id for x in [*self.memories, *self.computes]]
        if len(set(ids)) != len(ids):
            raise ValueError("hardware node IDs must be unique")
        reached = {self.memories[0].id}
        pairs = set()
        for link in self.links:
            if link.source not in ids or link.target not in ids or link.source == link.target:
                raise ValueError(f"invalid hardware link endpoints: {link.source!r} -> {link.target!r}; "
                                 f"unknown endpoints: {sorted({link.source, link.target} - set(ids))}; "
                                 f"available nodes: {ids}; use source resource IDs consistently")
            pair = frozenset((link.source, link.target))
            if pair in pairs:
                raise ValueError("duplicate undirected hardware link; specify each connection once")
            pairs.add(pair)
        for _ in ids:
            for link in self.links:
                if link.source in reached:
                    reached.add(link.target)
                if link.target in reached:
                    reached.add(link.source)
        if set(ids) != reached:
            raise ValueError("all nodes must be reachable from the first memory through undirected links; "
                             f"unreachable nodes: {sorted(set(ids) - reached)}")
        memory_ids = {m.id for m in self.memories}
        memory_reached = {self.memories[0].id}
        for _ in memory_ids:
            for link in self.links:
                if link.source in memory_ids and link.target in memory_ids:
                    if link.source in memory_reached:
                        memory_reached.add(link.target)
                    if link.target in memory_reached:
                        memory_reached.add(link.source)
        if memory_reached != memory_ids:
            raise ValueError("all memories must be reachable from external memory without intermediate "
                             "compute nodes; missing memory-only paths: "
                             + str(sorted(memory_ids - memory_reached)))
        return self


class Gemm(StrictModel):
    operator: Literal["gemm"]
    M: int = Field(gt=0, le=65536, strict=True)
    N: int = Field(gt=0, le=65536, strict=True)
    K: int = Field(gt=0, le=65536, strict=True)
    dtype: Literal["fp16", "fp32"]
    bias: Literal[False] = False
    description: str
    basis: Literal["user_request", "document"] = "user_request"
    citations: list[Citation] = Field(default_factory=list, max_length=6)


class Proposal(StrictModel):
    hardware: Hardware | None = None
    workload: Gemm | None = None
    missing: list[str] = Field(default_factory=list, max_length=20)
    rationale: str
    limitations: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def withhold_incomplete_hardware(cls, data):
        # A model may list missing requirements yet include a partial graph.
        # Such a graph is never executable. Preserve the gaps so retrieval and
        # reporting can address them rather than repairing its disconnectedness.
        # The original candidate remains in the saved LLM response artifact.
        if isinstance(data, dict) and isinstance(data.get("missing"), list) and data["missing"]:
            return {**data, "hardware": None}
        return data


class Assumption(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: float = Field(gt=0)
    unit: Literal["byte", "byte/s", "FLOP/s"]
    rationale: str = Field(min_length=8, max_length=1000)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=12000)
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    mode: Literal["auto", "question", "simulate", "inspect"] = "auto"
    assumptions: list[Assumption] = Field(default_factory=list, max_length=20)
    construction: Literal["grounded", "legacy"] = "grounded"
    bandwidth_policy: Literal["strict", "hbm-proxy"] = "hbm-proxy"
    evidence_run_id: str | None = None
    blueprint: dict | None = None
    deterministic_analysis: bool = True
    workload_contract: Gemm | None = None


class Finding(StrictModel):
    kind: Literal["observed", "hypothesis", "limitation"]
    text: str
    metric: str | None = None
    value: float | None = None


class Analysis(StrictModel):
    findings: list[Finding] = Field(max_length=12)


class RetrievalPlan(StrictModel):
    queries: list[str] = Field(default_factory=list, max_length=3)
    reason: str


class QuantityReference(StrictModel):
    fact_id: str | None = None
    count_fact_id: str | None = None
    clock_fact_id: str | None = None
    assumption_id: str | None = None


class MemoryReference(StrictModel):
    id: str
    capacity: QuantityReference


class ComputeReference(StrictModel):
    id: str
    kind: Literal["matrix", "vector"]
    throughput: QuantityReference


class LinkReference(StrictModel):
    source: str
    target: str
    relationship_id: str
    relationship_ids: list[str] = Field(default_factory=list, max_length=24)
    bandwidth: QuantityReference
    topology_basis: Literal["documented", "containment_access", "containment_projection", "shared_memory_projection"] = "documented"


class DesignBlocker(StrictModel):
    category: Literal["workload", "capacity", "throughput", "topology", "source_scope", "source_conflict"]
    reason: str = Field(min_length=8, max_length=1000)
    reference_id: str | None = None
    conflicting_fact_ids: list[str] = Field(default_factory=list, max_length=8)


class HardwareBlueprint(StrictModel):
    """Model selects evidence; trusted code, not the model, supplies all values."""
    name: str
    variant: str
    memories: list[MemoryReference] = Field(default_factory=list, max_length=12)
    computes: list[ComputeReference] = Field(default_factory=list, max_length=8)
    links: list[LinkReference] = Field(default_factory=list, max_length=24)
    workload: Gemm | None = None
    proxy_bandwidth_fact_id: str | None = None
    missing: list[str] = Field(default_factory=list, max_length=20)
    blockers: list[DesignBlocker] = Field(default_factory=list, max_length=20)
    resource_bindings: list[ResourceBinding] = Field(default_factory=list, max_length=12)
    memory_pools: list[PhysicalMemoryPool] = Field(default_factory=list, max_length=12)
    rationale: str
    limitations: list[str] = Field(default_factory=list, max_length=12)
