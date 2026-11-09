"""Compile a reference-only blueprint into executable, auditable hw-perf inputs."""
import hashlib
import math

from .calculations import derive
from .schemas import Assumption, HardwareBlueprint, Proposal, Quantity


LIMITATIONS = [
    "Single-device aggregate model: per-SM local capacities are pooled, not independently scheduled SMs.",
    "KB/MB/GB use decimal SI units; binary units are used only when the source explicitly says KiB/MiB/GiB.",
    "A contains-to-access link represents modeled access to local memory, not a measured physical interconnect.",
    "Undirected shared link bandwidth, zero fixed transfer latency; no cache-hit, occupancy, instruction, or clock-variation model.",
    "Documented peak throughput and any boost-clock conversion describe an idealized peak-rate scenario, not sustained performance.",
    "Logical memory views share capacity only with a cited combined-capacity fact and an explicit single-memory-block proof; the capacity is counted once.",
    "Unmodeled source containers may project connections onto a documented contained local-memory boundary; bindings and source connections are retained in topology-audit.json.",
]


def compile_blueprint(blueprint, extraction, assumptions, bandwidth_policy="strict", *, topology_audit=None):
    blueprint = HardwareBlueprint.model_validate(blueprint)
    if blueprint.missing:
        raise ValueError(
            "Unclassified design concerns require design repair, not another source search. "
            "Reclassify every legacy missing entry: adapter-unmodeled details and policy-resolved "
            "bandwidth gaps belong in limitations; a genuine unresolved workload, capacity, "
            "throughput, topology, source-scope, or source-conflict concern belongs in typed blockers "
            "with its reference_id. Leave missing empty. Do not discard source uncertainty. "
            "Concerns: " + "; ".join(blueprint.missing))
    facts = {f.id: f for f in extraction.facts}
    for blocker in blueprint.blockers:
        if blocker.category != "source_conflict":
            continue
        ids = list(dict.fromkeys(blocker.conflicting_fact_ids))
        if len(ids) < 2 or any(fid not in facts for fid in ids):
            raise ValueError("A source_conflict blocker must reference at least two verified "
                             "conflicting_fact_ids. A selected fact plus an already resolved "
                             "variant/footnote comparison is not a conflict. Record the selection "
                             "in limitations, or retain genuine unresolved uncertainty as a "
                             "source_scope blocker. Concern: " + blocker.reason)
        compared = [facts[fid] for fid in ids]
        scopes = {(f.parameter, f.variant, f.precision, f.mode, f.aggregation, f.unit) for f in compared}
        if len(scopes) != 1 or compared[0].variant != blueprint.variant:
            raise ValueError("Conflict facts differ in quantity, variant, precision, mode, "
                             "aggregation or units. Resolve the scope selection explicitly in "
                             "limitations; if applicability remains uncertain, retain a source_scope "
                             "blocker rather than claiming a same-scope source_conflict. Facts: " + str(ids))
        if all(math.isclose(f.value, compared[0].value, rel_tol=1e-9) for f in compared):
            raise ValueError("Conflict facts have matching values. Record the resolved comparison "
                             "in limitations rather than an unresolved source_conflict: " + str(ids))
    from .topology import normalize_topology
    assumptions = [Assumption.model_validate(a) for a in assumptions]
    if not blueprint.blockers:
        blueprint = normalize_topology(blueprint, extraction, topology_audit, assumptions)
    elif topology_audit is not None:
        topology_audit.update({"deferred": True, "reason": "Unresolved typed design blockers"})
    allowed = {a.id: a for a in assumptions}
    generated = []
    gaps = [f"{b.category}: {b.reason}" + (f" [{b.reference_id}]" if b.reference_id else "")
            for b in blueprint.blockers]
    audit = []
    limitations = list(dict.fromkeys([*LIMITATIONS, *blueprint.limitations]))
    used_citations = {}

    def quantity(ref, unit, field, allow_proxy=False):
        try:
            if ref.assumption_id:
                if ref.fact_id or ref.count_fact_id or ref.clock_fact_id:
                    raise ValueError("An assumed quantity cannot also claim documented fact lineage")
                a = allowed.get(ref.assumption_id)
                if a is None or a.unit != unit:
                    raise ValueError("No matching explicit assumption: " + ref.assumption_id)
                q = Quantity(value=a.value, unit=unit, basis="assumed",
                             assumption_id=a.id, rationale=a.rationale)
            else:
                q = derive(ref, facts, unit)
                for fid in q.fact_ids:
                    if facts[fid].variant != blueprint.variant:
                        raise ValueError("Quantity source does not match selected hardware variant")
                    for cite in facts[fid].citations:
                        used_citations[(cite.chunk_id, cite.quote)] = cite
            audit.append({"field": field, **q.model_dump()})
            return q
        except ValueError as exc:
            # Explicit bad references never turn into assumptions. Proxy policy
            # covers only an omitted bandwidth or an unconverted per-clock rate.
            missing_rate = ref.fact_id is None and not any(
                [ref.assumption_id, ref.count_fact_id, ref.clock_fact_id])
            unconverted = (ref.fact_id in facts and facts[ref.fact_id].unit in
                           {"Bytes/clk", "bytes/clk", "B/clk", "byte/clock"}
                           and facts[ref.fact_id].variant == blueprint.variant
                           and not ref.clock_fact_id and not ref.assumption_id
                           and not ref.count_fact_id)
            if allow_proxy and bandwidth_policy == "hbm-proxy" and (missing_rate or unconverted):
                proxy_id = blueprint.proxy_bandwidth_fact_id
                proxy = facts.get(proxy_id)
                if (proxy is None or proxy.parameter != "memory_bandwidth"
                        or proxy.aggregation != "device" or proxy.variant != blueprint.variant):
                    gaps.append(field + ": proxy policy needs a selected device-memory bandwidth fact")
                    return None
                from .schemas import QuantityReference
                try:
                    reference = derive(QuantityReference(fact_id=proxy_id), facts, unit)
                except ValueError as error:
                    gaps.append(field + ": " + str(error))
                    return None
                aid = "proxy_" + hashlib.sha256(field.encode()).hexdigest()[:16]
                rationale = ("Explicit hbm-proxy experiment: use the documented device-memory rate "
                             f"from {proxy_id} for {field}. This is an assumed internal-link rate, "
                             "not a specification or a validated performance bound.")
                a = Assumption(id=aid, value=reference.value, unit=unit, rationale=rationale)
                generated.append(a)
                q = Quantity(value=a.value, unit=unit, basis="assumed", assumption_id=aid,
                             rationale=rationale)
                audit.append({"field": field, **q.model_dump(), "proxy_source_fact_id": proxy_id,
                              "unresolved_source_fact_id": ref.fact_id,
                              "missing_reason": str(exc)})
                limitations.append(rationale)
                for cite in proxy.citations:
                    used_citations[(cite.chunk_id, cite.quote)] = cite
                return q
            gaps.append(field + ": " + str(exc))
            return None

    memories, computes, links = [], [], []
    for memory in blueprint.memories:
        q = quantity(memory.capacity, "byte", memory.id + ".capacity")
        if q:
            if not math.isclose(q.value, round(q.value), rel_tol=0, abs_tol=1e-6):
                gaps.append(memory.id + ".capacity must be integral bytes")
            memories.append({"id": memory.id, "capacity": q})
    for compute in blueprint.computes:
        q = quantity(compute.throughput, "FLOP/s", compute.id + ".throughput")
        if q:
            computes.append({"id": compute.id, "kind": compute.kind, "throughput": q})
    by_relation = {r.id: r for r in extraction.relationships}
    for pool in blueprint.memory_pools:
        for rid in pool.relationship_ids:
            relationship = by_relation.get(rid)
            if relationship is not None:
                for cite in relationship.citations:
                    used_citations[(cite.chunk_id, cite.quote)] = cite
    for binding in blueprint.resource_bindings:
        relationship = by_relation.get(binding.containment_id)
        if relationship is not None:
            for cite in relationship.citations:
                used_citations[(cite.chunk_id, cite.quote)] = cite
    for link in blueprint.links:
        relationship = by_relation.get(link.relationship_id)
        if relationship is not None:
            for cite in relationship.citations:
                used_citations[(cite.chunk_id, cite.quote)] = cite
        q = quantity(link.bandwidth, "byte/s", link.source + "->" + link.target + ".bandwidth", True)
        if q:
            links.append({"source": link.source, "target": link.target, "bandwidth": q,
                          "relationship_id": link.relationship_id,
                          "relationship_ids": link.relationship_ids,
                          "topology_basis": link.topology_basis})
    if not memories:
        gaps.append("No evidence-backed memory resource was selected")
    if not computes:
        gaps.append("No evidence-backed compute resource was selected")
    if not links:
        gaps.append("No complete memory-to-compute connection was selected")
    if blueprint.workload is None:
        gaps.append("An explicit dense GEMM workload is required")
    # Missing source paths are evidence gaps, not something a design retry can
    # fix by inventing an edge. Return all gaps together, including capacities.
    if memories and computes:
        reached = {blueprint.memories[0].id}
        for _ in range(len(memories) + len(computes)):
            for link in blueprint.links:
                if link.source in reached:
                    reached.add(link.target)
                if link.target in reached:
                    reached.add(link.source)
        unreachable = {n["id"] for n in [*memories, *computes]} - reached
        if unreachable:
            gaps.append("Source topology is disconnected: retrieve source-backed connectivity "
                        "between external/shared cache and compute/local memory. Unreachable: "
                        + ", ".join(sorted(unreachable)))
        else:
            memory_ids = {m.id for m in blueprint.memories}
            memory_reached = {blueprint.memories[0].id}
            for _ in memory_ids:
                for link in blueprint.links:
                    if link.source in memory_ids and link.target in memory_ids:
                        if link.source in memory_reached: memory_reached.add(link.target)
                        if link.target in memory_reached: memory_reached.add(link.source)
            missing_routes = memory_ids - memory_reached
            if missing_routes:
                gaps.append("Source memory hierarchy needs memory-only connectivity from external memory "
                            "to " + ", ".join(sorted(missing_routes)) + ". The backend cannot route "
                            "memory transfers through intermediate compute nodes. Retrieve documented "
                            "cache/global-to-local memory data paths.")
    gaps = list(dict.fromkeys(gaps))
    hardware = None if gaps else {
        "name": blueprint.name, "variant": blueprint.variant,
        "citations": list(used_citations.values())[:12],
        "memories": memories, "computes": computes, "links": links,
        "resource_bindings": blueprint.resource_bindings,
        "memory_pools": blueprint.memory_pools,
    }
    proposal = Proposal(hardware=hardware, workload=blueprint.workload,
                        missing=gaps[:20], rationale=blueprint.rationale,
                        limitations=limitations[:12])
    effective = {a.id: a for a in [*assumptions, *generated]}
    return proposal, list(effective.values()), audit, gaps
