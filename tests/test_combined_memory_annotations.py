"""A single already-combined block is not a multi-resource alias group."""
import unittest

from paper2perf.compiler import compile_blueprint
from paper2perf.evidence import verify_proposal
from paper2perf.schemas import HardwareBlueprint, PhysicalMemoryPool
from test_failed_chat_regression import repaired_inputs


def case(kind='shares_resource', source_name='shared_memory'):
    ex, chunks, bp = repaired_inputs()
    r = next(r for r in ex.relationships if r.id == 'l1_to_smem_combined')
    r.source = r.target = source_name
    r.kind = kind
    return ex, chunks, bp


class CombinedAnnotationTests(unittest.TestCase):
    def test_single_combined_resource_does_not_create_singleton_pool(self):
        ex, chunks, bp = case()
        audit = {}
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy', topology_audit=audit)
        verify_proposal(p, ex, assumptions, chunks)
        self.assertFalse(gaps)
        self.assertEqual(p.hardware.memory_pools, [])
        self.assertEqual(len(p.hardware.memories), 3)
        self.assertEqual(p.hardware.memories[-1].capacity.value, 192000 * 108)
        self.assertEqual(len(p.hardware.links), 4)
        self.assertEqual(len(audit['already_combined_memories']), 1)
        self.assertEqual(audit['already_combined_memories'][0]['memory_id'], 'shared_memory')

    def test_historical_connects_annotation_is_accepted_only_with_combining_proof(self):
        ex, chunks, bp = case('connects')
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        verify_proposal(p, ex, assumptions, chunks)
        self.assertFalse(p.hardware.memory_pools)

    def test_single_logical_name_can_map_to_unique_combined_physical_capacity(self):
        ex, chunks, bp = case(source_name='source_combined_block')
        audit = {}
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy', topology_audit=audit)
        verify_proposal(p, ex, assumptions, chunks)
        self.assertEqual(p.hardware.memory_pools, [])
        self.assertEqual(audit['already_combined_memories'][0]['source_resource'], 'source_combined_block')
        self.assertEqual(audit['already_combined_memories'][0]['memory_id'], 'shared_memory')

    def test_ambiguous_physical_capacity_is_not_selected(self):
        ex, _, bp = case(source_name='source_combined_block')
        bp.memories[1].capacity = bp.memories[-1].capacity.model_copy(deep=True)
        with self.assertRaisesRegex(ValueError, 'unambiguous modeled'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_real_self_transfer_does_not_become_free_sharing(self):
        ex, _, bp = case('connects')
        relation = next(r for r in ex.relationships if r.id == 'l1_to_smem_combined')
        relation.citations[0].quote = 'Transfer data within a single memory block using an internal copy engine.'
        with self.assertRaisesRegex(ValueError, 'self-transfer'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_partial_shared_capacity_is_not_combined_capacity(self):
        ex, _, bp = case()
        bp.memories[-1].capacity.fact_id = 'cache_capacity_shared_max'
        with self.assertRaisesRegex(ValueError, 'cited combined-capacity'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_compute_cannot_be_reinterpreted_as_combined_memory(self):
        ex, _, bp = case(source_name='a100_sm')
        with self.assertRaisesRegex(ValueError, 'cannot replace a compute'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_actual_alias_groups_still_require_two_unique_members(self):
        with self.assertRaises(ValueError):
            PhysicalMemoryPool(memory_id='m', member_resource_ids=['m'],
                               relationship_ids=['r'], capacity_fact_id='capacity')
        ex, _, bp = case()
        bp.memory_pools = [PhysicalMemoryPool(memory_id='shared_memory',
            member_resource_ids=['shared_memory', 'shared_memory'],
            relationship_ids=['l1_to_smem_combined'], capacity_fact_id='cache_capacity_l1_shared')]
        with self.assertRaises(ValueError):
            compile_blueprint(bp, ex, [], 'hbm-proxy')
