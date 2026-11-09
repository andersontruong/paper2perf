"""One physical capacity for source-documented logical views of a single block."""
import re

from .schemas import PhysicalMemoryPool


def single_block(relationship):
    text = ' '.join(' '.join(c.quote for c in relationship.citations).lower().split())
    return bool(re.search(r'\b(?:single|same)\s+(?:physical\s+)?memory\s+block\b', text))


def combined_capacity(fact):
    return (fact is not None and fact.parameter in {'cache_capacity', 'memory_capacity'}
            and 'combined' in (fact.scope + ' ' + ' '.join(c.quote for c in fact.citations)).lower())


def check_memory_pools(pools, memories, computes, extraction, variant, *, references=False):
    facts = {f.id: f for f in extraction.facts}
    relations = {r.id: r for r in extraction.relationships}
    mem = {m.id: m for m in memories}
    node_ids = set(mem) | {c.id for c in computes}
    aliases, internal, seen_pools = {}, set(), set()
    for pool in pools:
        members = set(pool.member_resource_ids)
        if (pool.memory_id not in mem or len(members) != len(pool.member_resource_ids)
                or pool.memory_id in seen_pools):
            raise ValueError('Physical memory pool needs a unique modeled capacity as its boundary')
        seen_pools.add(pool.memory_id)
        if (members - {pool.memory_id}) & node_ids:
            raise ValueError('Shared physical capacity must not duplicate modeled memories or merge compute nodes')
        fact = facts.get(pool.capacity_fact_id)
        capacity = mem[pool.memory_id].capacity
        fid = capacity.fact_id if references else (capacity.fact_ids[0] if capacity.fact_ids else None)
        if (not combined_capacity(fact) or fact.variant != variant or fid != pool.capacity_fact_id
                or (not references and capacity.basis == 'assumed')):
            raise ValueError('Physical memory pooling requires the selected cited combined-capacity fact')
        reached = {next(iter(members))}
        edges = []
        proof = any(relations.get(rid) is not None and relations[rid].kind == 'shares_resource'
                    and single_block(relations[rid]) for rid in pool.relationship_ids)
        for rid in pool.relationship_ids:
            r = relations.get(rid)
            if (r is None or r.variant != variant or r.kind not in {'connects', 'shares_resource'}
                    or not {r.source, r.target}.issubset(members) or r.source == r.target
                    or (not single_block(r) and not (r.kind == 'connects' and proof))):
                raise ValueError('Physical memory pooling lacks an explicit single-memory-block source proof')
            edges.append((r.source, r.target))
            internal.add(rid)
        for _ in members:
            for a, b in edges:
                if a in reached: reached.add(b)
                if b in reached: reached.add(a)
        if reached != members:
            raise ValueError('Every logical memory member must be connected by a physical sharing proof')
        for member in members - {pool.memory_id}:
            if member in aliases:
                raise ValueError('Logical memory resource belongs to multiple physical pools')
            aliases[member] = pool.memory_id
    for r in extraction.relationships:
        if r.variant != variant:
            continue
        if r.source == r.target and r.kind in {'connects', 'shares_resource'} and single_block(r):
            # Extraction can describe a block that is already combined using a
            # reflexive sharing annotation. It is not a two-view pool or a bus.
            text = ' '.join(c.quote for c in r.citations).lower()
            if r.kind == 'connects' and not re.search(r'\bcombin(?:ed|ing)\b', text):
                raise ValueError('A self-transfer is not an already-combined memory annotation')
            if r.source in node_ids and r.source not in mem:
                raise ValueError('An already-combined memory annotation cannot replace a compute node')
            target = aliases.get(r.source, r.source)
            if target in mem:
                candidates = [mem[target]]
            else:
                candidates = []
                for memory in memories:
                    q = memory.capacity
                    fid = q.fact_id if references else (q.fact_ids[0] if q.fact_ids else None)
                    if combined_capacity(facts.get(fid)):
                        candidates.append(memory)
            if len(candidates) != 1:
                raise ValueError('Already-combined memory needs one unambiguous modeled combined-capacity resource')
            memory = candidates[0]
            q = memory.capacity
            fid = q.fact_id if references else (q.fact_ids[0] if q.fact_ids else None)
            fact = facts.get(fid)
            if (not combined_capacity(fact) or fact.variant != variant
                    or not references and q.basis == 'assumed'):
                raise ValueError('Already-combined memory requires its cited combined-capacity fact')
            if r.source != memory.id:
                if r.source in aliases and aliases[r.source] != memory.id:
                    raise ValueError('Conflicting already-combined memory representations')
                aliases[r.source] = memory.id
            internal.add(r.id)
        endpoints = {aliases.get(r.source, r.source), aliases.get(r.target, r.target)}
        if (r.kind == 'shares_resource' or
                r.kind == 'connects' and single_block(r) and endpoints.issubset(mem)):
            if r.id not in internal:
                raise ValueError('Source-documented shared memory must be represented as one proved physical pool')
    return aliases, internal


def infer_memory_pools(blueprint, extraction):
    pools = [p.model_copy(deep=True) for p in blueprint.memory_pools]
    memory_ids = {m.id for m in blueprint.memories}
    compute_ids = {c.id for c in blueprint.computes}
    facts = {f.id: f for f in extraction.facts}
    by_memory = {p.memory_id: p for p in pools}
    for r in extraction.relationships:
        if r.variant != blueprint.variant or r.kind not in {'connects', 'shares_resource'} or not single_block(r):
            continue
        if r.source == r.target:
            # Validated as an already-combined annotation in check_memory_pools;
            # never manufacture a PhysicalMemoryPool with just one unique ID.
            continue
        endpoints = {r.source, r.target}
        known = endpoints & memory_ids
        if endpoints & compute_ids:
            continue
        if len(known) > 1:
            raise ValueError('Source-documented physical pool is duplicated as multiple modeled memories; '
                             'select one combined-capacity node and retain logical views in memory_pools')
        if known:
            target = next(iter(known))
        else:
            candidates = [m for m in blueprint.memories if combined_capacity(facts.get(m.capacity.fact_id))]
            if len(candidates) != 1:
                continue
            target = candidates[0].id
        memory = next(m for m in blueprint.memories if m.id == target)
        fact = facts.get(memory.capacity.fact_id)
        if not combined_capacity(fact):
            continue
        if target in by_memory:
            pool = by_memory[target]
            pool.member_resource_ids = sorted(set(pool.member_resource_ids) | endpoints)
            pool.relationship_ids = list(dict.fromkeys([*pool.relationship_ids, r.id]))
        else:
            pool = PhysicalMemoryPool(memory_id=target, member_resource_ids=sorted(endpoints),
                                      relationship_ids=[r.id], capacity_fact_id=fact.id)
            pools.append(pool)
            by_memory[target] = pool
    for r in extraction.relationships:
        if r.variant != blueprint.variant or r.kind != 'connects':
            continue
        for pool in pools:
            if {r.source, r.target}.issubset(set(pool.member_resource_ids)):
                pool.relationship_ids = list(dict.fromkeys([*pool.relationship_ids, r.id]))
                break
    check_memory_pools(pools, blueprint.memories, blueprint.computes, extraction,
                       blueprint.variant, references=True)
    return pools
