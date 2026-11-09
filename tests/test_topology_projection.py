"""Regression for the user-reported graph shape, not a semantic paper evaluation."""
import unittest

from paper2perf.compiler import compile_blueprint
from paper2perf.evidence import EvidenceError, verify_proposal
from paper2perf.schemas import LinkReference, QuantityReference, Relationship, ResourceBinding
from test_grounded_compiler import inputs


def reported_case():
    ex, chunks, bp = inputs()
    refs = {r.id: r for r in ex.relationships}
    # User-reported resource/relationship IDs; reuse fixture citations only to
    # exercise mechanical validation. This is not a new production source model.
    shape = [
        ('l2_to_memory', 'l2_cache', 'memory', 'connects', 'l2_connects_hbm2'),
        ('memory_contains_l2', 'memory', 'l2_cache', 'contains', 'l2_connects_hbm2'),
        ('gpu_contains_sms', 'gpu', 'sm', 'contains', 'sm_contains_l1_smem'),
        ('sm_contains_l1_smem', 'sm', 'l1_smem', 'contains', 'sm_contains_l1_smem'),
        ('sm_compute_to_l1_smem', 'sm_compute', 'l1_smem', 'connects', 'sm_contains_l1_smem'),
        ('sm_to_l2', 'sm', 'l2_cache', 'connects', 'sm_connects_l2'),
    ]
    ex.relationships = [Relationship(id=rid, source=src, target=dst, kind=kind,
                                    variant=bp.variant, citations=refs[cite].citations)
                        for rid, src, dst, kind, cite in shape]
    for node, rid in zip(bp.memories, ['memory', 'l2_cache', 'l1_smem']):
        node.id = rid
    bp.computes[0].id = 'sm_compute'
    bp.links = [
        LinkReference(source='sm', target='l1_smem', relationship_id='sm_contains_l1_smem', bandwidth=QuantityReference()),
        LinkReference(source='sm', target='l2_cache', relationship_id='sm_to_l2', bandwidth=QuantityReference(fact_id='a100_l2_read_bandwidth')),
        LinkReference(source='l2_cache', target='memory', relationship_id='l2_to_memory', bandwidth=QuantityReference(fact_id='a100_sxm_mem_bandwidth')),
    ]
    return ex, chunks, bp


class ProjectionTests(unittest.TestCase):
    def test_reported_endpoints_compile_without_renaming_or_duplicating_compute(self):
        ex, chunks, bp = reported_case()
        audit = {}
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy', topology_audit=audit)
        self.assertFalse(gaps)
        verify_proposal(p, ex, assumptions, chunks)
        self.assertEqual([c.id for c in p.hardware.computes], ['sm_compute'])
        self.assertEqual(len(p.hardware.memories), 3)
        self.assertEqual(len(p.hardware.links), 3)
        pairs = {frozenset((l.source, l.target)) for l in p.hardware.links}
        self.assertEqual(pairs, {frozenset(('memory', 'l2_cache')),
                                 frozenset(('l2_cache', 'l1_smem')),
                                 frozenset(('l1_smem', 'sm_compute'))})
        self.assertEqual(len(p.hardware.resource_bindings), 1)
        binding = p.hardware.resource_bindings[0]
        self.assertEqual((binding.source_resource, binding.memory_resource), ('sm', 'l1_smem'))
        self.assertTrue(any(e.get('relationship_id') == 'sm_compute_to_l1_smem'
                            for e in audit['events']))
        self.assertEqual(len(assumptions), 2)

    def test_restored_connection_still_needs_bandwidth_in_strict_mode(self):
        ex, _, bp = reported_case()
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [])
        self.assertIsNone(p.hardware)
        self.assertFalse(assumptions)
        self.assertTrue(any('sm_compute' in gap for gap in gaps))

    def test_no_containment_proof_means_no_inferred_mapping(self):
        ex, _, bp = reported_case()
        ex.relationships = [r for r in ex.relationships if r.id != 'sm_contains_l1_smem']
        bp.links = bp.links[1:]
        with self.assertRaisesRegex(ValueError, 'Do not guess a compute alias'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_ambiguous_container_boundary_is_not_guessed(self):
        ex, _, bp = reported_case()
        other = ex.relationships[3].model_copy(deep=True)
        other.id, other.target = 'sm_contains_other_memory', 'l2_cache'
        ex.relationships.append(other)
        with self.assertRaisesRegex(ValueError, 'expected one documented'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_explicit_binding_is_checked_against_containment(self):
        ex, _, bp = reported_case()
        bp.resource_bindings = [ResourceBinding(source_resource='sm', memory_resource='l2_cache',
                                               containment_id='sm_contains_l1_smem')]
        with self.assertRaisesRegex(ValueError, 'matching.*containment'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_tampered_binding_fails_independent_validation(self):
        ex, chunks, bp = reported_case()
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        p.hardware.resource_bindings[0].memory_resource = 'l2_cache'
        with self.assertRaisesRegex(EvidenceError, 'containment'):
            verify_proposal(p, ex, assumptions, chunks)

    def test_projection_deduplicates_edges_and_keeps_both_relationship_proofs(self):
        ex, chunks, bp = reported_case()
        extra = ex.relationships[-1].model_copy(deep=True)
        extra.id, extra.source, extra.target = 'local_to_l2', 'l1_smem', 'l2_cache'
        ex.relationships.append(extra)
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        verify_proposal(p, ex, assumptions, chunks)
        self.assertEqual(len(p.hardware.links), 3)
        projected = next(l for l in p.hardware.links if l.relationship_id == 'sm_to_l2')
        self.assertIn('local_to_l2', projected.relationship_ids)
        projected.relationship_ids = []
        with self.assertRaisesRegex(EvidenceError, 'omits documented'):
            verify_proposal(p, ex, assumptions, chunks)

    def test_wrong_link_endpoints_are_not_fuzzily_replaced(self):
        ex, _, bp = reported_case()
        bp.links[1].target = 'memory'
        with self.assertRaisesRegex(ValueError, 'endpoints disagree'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')


from test_grounded_compiler import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Projected graph execution requires installed simulator dependencies')
class RealProjectedSimulationTests(unittest.TestCase):
    def test_reported_graph_runs_through_real_capture_mapping_and_schedule(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from paper2perf.hwperf_worker import execute
        ex, chunks, bp = reported_case()
        proposal, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy')
        verify_proposal(proposal, ex, assumptions, chunks)
        with TemporaryDirectory() as folder:
            metrics = execute(proposal, folder)
            self.assertGreater(metrics['makespan_seconds'], 0)
            self.assertEqual(metrics['backend'], 'hw-perf')
            graph = json.loads((Path(folder) / 'hardware-graph.json').read_text())
            self.assertEqual(len(graph['nodes']), 4)
            self.assertEqual(len(graph.get('edges', graph.get('links', []))), 3)


def modeled_compute_binding_case():
    ex, chunks, bp = inputs()
    names = {'a100_hbm2': 'hbm2_device_memory', 'a100_l1_smem': 'l1_shared_memory',
             'a100_l2_cache': 'l2_cache', 'a100_sm': 'a100_sm'}
    for memory in bp.memories:
        memory.id = names[memory.id]
    ex.relationships = [r for r in ex.relationships if r.id != 'l1_smem_connects_l2']
    for relationship in ex.relationships:
        relationship.source = names[relationship.source]
        relationship.target = names[relationship.target]
    # Supply the memory-to-memory route required by the backend separately
    # from the reported redundant binding; an actor cannot act as a memory bus.
    route = ex.relationships[-1].model_copy(deep=True)
    route.id, route.source, route.target = 'local_memory_route', 'l2_cache', 'l1_shared_memory'
    route.kind = 'connects'
    ex.relationships.append(route)
    bp.links = [LinkReference(source=r.source, target=r.target, relationship_id=r.id,
                              bandwidth=QuantityReference(),
                              topology_basis='containment_access' if r.kind == 'contains' else 'documented')
                for r in ex.relationships]
    bp.resource_bindings = [ResourceBinding(source_resource='a100_sm', memory_resource='l1_shared_memory',
                                           containment_id='sm_contains_l1_smem')]
    return ex, chunks, bp


class RedundantComputeBindingTests(unittest.TestCase):
    def test_reported_compute_and_access_edge_survive_redundant_binding_removal(self):
        ex, chunks, bp = modeled_compute_binding_case()
        audit = {}
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy', topology_audit=audit)
        verify_proposal(p, ex, assumptions, chunks)
        self.assertFalse(gaps)
        self.assertEqual(p.hardware.resource_bindings, [])
        self.assertEqual([c.id for c in p.hardware.computes], ['a100_sm'])
        self.assertEqual(p.hardware.computes[0].throughput.value, 312e12)
        self.assertEqual(len(p.hardware.links), 4)
        access = next(l for l in p.hardware.links if l.relationship_id == 'sm_contains_l1_smem')
        self.assertEqual(access.topology_basis, 'containment_access')
        self.assertEqual({access.source, access.target}, {'a100_sm', 'l1_shared_memory'})
        self.assertTrue(any(e['action'] == 'remove_redundant_compute_binding' for e in audit['events']))
        self.assertEqual(len(bp.resource_bindings), 1)  # Input plan is preserved for audit.

    def test_forged_containment_is_not_silently_discarded(self):
        ex, _, bp = modeled_compute_binding_case()
        bp.resource_bindings[0].containment_id = 'invented_proof'
        with self.assertRaisesRegex(ValueError, 'containment evidence'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_without_access_edge_binding_cannot_erase_compute(self):
        ex, _, bp = modeled_compute_binding_case()
        bp.links = [l for l in bp.links if l.relationship_id != 'sm_contains_l1_smem']
        with self.assertRaisesRegex(ValueError, 'must keep its identity'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_strict_bandwidth_gaps_remain_after_normalization(self):
        ex, _, bp = modeled_compute_binding_case()
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [])
        self.assertIsNone(p.hardware)
        self.assertFalse(assumptions)
        self.assertEqual(sum('bandwidth' in gap for gap in gaps), 4)

    def test_final_ir_validator_still_rejects_compute_collapse(self):
        ex, chunks, bp = modeled_compute_binding_case()
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        p.hardware.resource_bindings = bp.resource_bindings
        with self.assertRaisesRegex(EvidenceError, 'unmodeled container'):
            verify_proposal(p, ex, assumptions, chunks)
