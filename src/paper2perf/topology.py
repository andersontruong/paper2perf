"""Project documented grouping containers onto a modeled local-memory boundary.

A source container is not equated to a compute node. Its containment proof names
an explicit aggregate memory boundary. No names, bandwidths, or capacities are guessed.
"""
import math

from .calculations import derive
from .schemas import LinkReference, QuantityReference, ResourceBinding


def check_bindings(bindings, relationships, memory_ids, node_ids, variant, memory_aliases=None):
    memory_aliases = memory_aliases or {}
    by_id = {r.id: r for r in relationships}
    mapping = {}
    for binding in bindings:
        r = by_id.get(binding.containment_id)
        if (r is None or r.kind != 'contains' or r.variant != variant
                or r.source != binding.source_resource or r.target != binding.memory_resource):
            raise ValueError(
                f'Resource binding {binding.source_resource!r} -> {binding.memory_resource!r} '
                f'lacks matching selected-variant containment evidence for '
                f'{binding.containment_id!r}')
        if (binding.source_resource in node_ids or binding.source_resource in memory_aliases
                or memory_aliases.get(binding.memory_resource, binding.memory_resource) not in memory_ids):
            raise ValueError('Only an unmodeled container may bind directly to a modeled memory boundary')
        if binding.source_resource in mapping:
            raise ValueError('Duplicate or ambiguous resource binding')
        mapping[binding.source_resource] = memory_aliases.get(binding.memory_resource, binding.memory_resource)
    return mapping


def normalize_topology(blueprint, extraction, audit=None, assumptions=None):
    blueprint = blueprint.model_copy(deep=True)
    memory_ids = {m.id for m in blueprint.memories}
    compute_ids = {c.id for c in blueprint.computes}
    node_ids = memory_ids | compute_ids
    if len(node_ids) != len(blueprint.memories) + len(blueprint.computes):
        raise ValueError('hardware node IDs must be unique')
    relations = {r.id: r for r in extraction.relationships if r.variant == blueprint.variant}
    facts = {f.id: f for f in extraction.facts}
    by_assumption = {a.id: a for a in (assumptions or [])}
    from .memory_pools import infer_memory_pools, check_memory_pools
    blueprint.memory_pools = infer_memory_pools(blueprint, extraction)
    aliases, pool_relationships = check_memory_pools(
        blueprint.memory_pools, blueprint.memories, blueprint.computes, extraction,
        blueprint.variant, references=True)
    bindings = []
    events = [{"action": "represent_shared_physical_memory", "pool": p.model_dump()}
              for p in blueprint.memory_pools]
    memory_by_id = {m.id: m for m in blueprint.memories}
    annotations = []
    for r in relations.values():
        if r.id in pool_relationships and r.source == r.target:
            target = aliases.get(r.source, r.source)
            annotation = {"source_resource": r.source, "memory_id": target,
                          "relationship_id": r.id,
                          "capacity_fact_id": memory_by_id[target].capacity.fact_id}
            annotations.append(annotation)
            events.append({"action": "represent_already_combined_memory", **annotation})
    for binding in blueprint.resource_bindings:
        if aliases.get(binding.source_resource) == binding.memory_resource:
            events.append({'action': 'remove_redundant_pool_binding',
                           'binding': binding.model_dump(),
                           'mapped_memory': binding.memory_resource})
            continue
        if binding.source_resource in compute_ids and binding.memory_resource in memory_ids:
            proof = relations.get(binding.containment_id)
            if (proof is None or proof.kind != 'contains'
                    or proof.source != binding.source_resource or proof.target != binding.memory_resource):
                raise ValueError(
                    f'Resource binding {binding.source_resource!r} -> {binding.memory_resource!r} '
                    f'lacks matching selected-variant containment evidence for '
                    f'{binding.containment_id!r}')
            access_edges = [link for link in blueprint.links
                            if {link.source, link.target} == {proof.source, proof.target}
                            and proof.id in {link.relationship_id, *link.relationship_ids}]
            if not access_edges:
                raise ValueError(f'Modeled compute {binding.source_resource!r} must keep its identity: '
                                 'replace its container binding with a source-backed containment_access '
                                 'edge and an explicit bandwidth reference or unresolved rate')
            # Both resources already exist and the proven access is present.
            # Removing only this redundant mapping leaves nodes, edges and
            # quantities intact; check_bindings remains strict on final IR.
            events.append({'action': 'remove_redundant_compute_binding',
                           'binding': binding.model_dump(),
                           'preserved_access_relationship_id': proof.id})
            continue
        bindings.append(binding)
    mapping = {**aliases, **check_bindings(bindings, extraction.relationships, memory_ids, node_ids, blueprint.variant, aliases)}
    # Consider actual connectivity endpoints and selected edges, not arbitrary
    # names such as 'sm'/'sm_compute' or every grouping parent in the extraction.
    needed = {x for r in relations.values() if r.kind in {'connects', 'shares_resource'} for x in (r.source, r.target)}
    needed |= {x for link in blueprint.links for x in (link.source, link.target)}
    for source in sorted(needed - node_ids - set(mapping)):
        candidates = [r for r in relations.values()
                      if r.kind == 'contains' and r.source == source
                      and (r.target in memory_ids or r.target in aliases)]
        if len(candidates) != 1:
            raise ValueError(f'Unmodeled source resource {source!r}: expected one documented contained '
                             f'memory boundary, found {[r.target for r in candidates]}. '
                             'Do not guess a compute alias; preserve distinct source resources or '
                             'supply an evidence-backed resource binding.')
        r = candidates[0]
        bindings.append(ResourceBinding(source_resource=source, memory_resource=r.target,
                                        containment_id=r.id))
        mapping[source] = aliases.get(r.target, r.target)
    check_bindings(bindings, extraction.relationships, memory_ids, node_ids, blueprint.variant, aliases)
    normalized = []
    by_pair = {}
    covered = set(pool_relationships)

    def add(link, recovered=False):
        ids = list(dict.fromkeys([link.relationship_id, *link.relationship_ids]))
        primary = relations.get(link.relationship_id)
        if primary is None:
            raise ValueError(f'Unknown or wrong-variant source relationship {link.relationship_id!r}')
        original = {primary.source, primary.target}
        projected = {mapping.get(primary.source, primary.source), mapping.get(primary.target, primary.target)}
        if {link.source, link.target} not in (original, projected):
            raise ValueError(f'Link {link.relationship_id!r} endpoints disagree with its source: '
                             f'expected {sorted(original)} or documented projection {sorted(projected)}')
        source = mapping.get(primary.source, primary.source)
        target = mapping.get(primary.target, primary.target)
        if source == target:
            if primary.id in pool_relationships:
                events.append({'action': 'represent_sharing_as_pool', 'relationship_id': primary.id})
                return
            if primary.kind == 'contains' and any(b.containment_id == primary.id for b in bindings):
                events.append({'action': 'represent_containment_as_binding', 'relationship_id': primary.id})
                return
            raise ValueError('Source connectivity collapses into a self-link; cannot project it safely')
        if source not in node_ids or target not in node_ids:
            raise ValueError(f'Unresolved hardware endpoints for {primary.id}: {source}, {target}')
        for rid in ids:
            r = relations.get(rid)
            if r is None:
                raise ValueError('Additional link provenance references an unknown relationship')
            endpoints = {mapping.get(r.source, r.source), mapping.get(r.target, r.target)}
            if endpoints != {source, target}:
                raise ValueError('Merged relationship does not project to the same physical edge')
        link.source, link.target = source, target
        link.relationship_ids = ids[1:]
        if original != {source, target}:
            link.topology_basis = ('shared_memory_projection' if primary.source in aliases
                                   or primary.target in aliases else 'containment_projection')
            events.append({'action': 'project_container_connection', 'relationship_id': primary.id,
                           'source_endpoints': [primary.source, primary.target],
                           'hardware_endpoints': [source, target]})
        elif primary.kind == 'contains':
            link.topology_basis = 'containment_access'
        pair = frozenset((source, target))
        if pair in by_pair:
            old = by_pair[pair]
            old_ids = {old.relationship_id, *old.relationship_ids}
            # Duplicate model references to the same connection are not extra
            # bandwidth channels. Differing explicit rates remain an error.
            empty = QuantityReference()
            if old.bandwidth != link.bandwidth and old.bandwidth != empty and link.bandwidth != empty:
                resolved = []
                for reference in (old.bandwidth, link.bandwidth):
                    if reference.assumption_id:
                        assumption = by_assumption.get(reference.assumption_id)
                        resolved.append(None if assumption is None
                                        else (assumption.value, assumption.unit))
                    else:
                        try:
                            quantity = derive(reference, facts, 'byte/s')
                            resolved.append((quantity.value, quantity.unit))
                        except ValueError:
                            resolved.append(None)
                equivalent = (resolved[0] is not None and resolved[1] is not None
                              and resolved[0][1] == resolved[1][1]
                              and math.isclose(resolved[0][0], resolved[1][0], rel_tol=1e-12))
                if not equivalent:
                    raise ValueError('Projected duplicate connections specify conflicting '
                                     'bandwidth references')
            if old.bandwidth == empty:
                old.bandwidth = link.bandwidth
            old.relationship_ids = list(dict.fromkeys([*old.relationship_ids, *[i for i in ids if i not in old_ids]]))
            if link.topology_basis == 'shared_memory_projection':
                old.topology_basis = 'shared_memory_projection'
            elif link.topology_basis == 'containment_projection' and old.topology_basis != 'shared_memory_projection':
                old.topology_basis = 'containment_projection'
            elif link.topology_basis == 'containment_access' and old.topology_basis == 'documented':
                old.topology_basis = 'containment_access'
            events.append({'action': 'merge_projected_connection', 'relationship_ids': ids,
                           'bandwidth_references': [old.bandwidth.model_dump(),
                                                   link.bandwidth.model_dump()]})
        else:
            by_pair[pair] = link
            normalized.append(link)
        covered.update(ids)
        if recovered:
            events.append({'action': 'restore_source_connection', 'relationship_id': primary.id})

    for link in blueprint.links:
        add(link)
    # Restore only actual selected-variant source connectivity. An unknown rate
    # stays null and is handled by strict mode or the explicitly selected policy.
    for r in relations.values():
        if r.kind == 'connects' and r.id not in covered:
            add(LinkReference(source=r.source, target=r.target, relationship_id=r.id,
                              bandwidth=QuantityReference()), recovered=True)
    if len(normalized) > 24:
        raise ValueError('Projected topology exceeds the supported link limit')
    blueprint.links = normalized
    blueprint.resource_bindings = bindings
    if audit is not None:
        audit.update({'memory_pools': [p.model_dump() for p in blueprint.memory_pools],
                      'already_combined_memories': annotations,
                      'resource_bindings': [b.model_dump() for b in bindings], 'events': events,
                      'resolved_links': [link.model_dump() for link in normalized]})
    return blueprint
