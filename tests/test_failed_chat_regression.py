"""Replay every design in the attached failed run; never a production fallback."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import unittest

from paper2perf.compiler import compile_blueprint
from paper2perf.evidence import verify_extraction, verify_proposal
from paper2perf.schemas import Extraction, Fact, HardwareBlueprint, Relationship
from test_grounded_compiler import AVAILABLE

FIXTURES = Path(__file__).parent / 'fixtures'


def load(name):
    return json.loads((FIXTURES / name).read_text())


def original_inputs():
    return (Extraction.model_validate(load('failed-chat-evidence.json')),
            load('failed-chat-chunks.json'),
            [HardwareBlueprint.model_validate(p) for p in load('failed-chat-blueprints.json')])


def repaired_inputs():
    """Expected extraction/design repair using original body passages, not invented specs."""
    ex, chunks, plans = original_inputs()
    body = load('failed-chat-body-chunks.json')
    chunk = next(c for c in body if c['page'] == 16 and 'the NVIDIA A100 GPU has' in c['text'])
    start = chunk['text'].index('the NVIDIA A100 GPU has')
    end = chunk['text'].index('HBM2 memory', start) + len('HBM2 memory')
    ex.facts.append(Fact(id='device_hbm_capacity', parameter='memory_capacity', value=40, unit='GB',
        scope='A100 SXM4 device HBM2 capacity', variant=plans[-1].variant,
        precision='unspecified', mode='unspecified', aggregation='device',
        citations=[{'chunk_id': chunk['id'], 'quote': chunk['text'][start:end]}]))
    # The original retrieved paragraph already states accessibility of L2 to SMs.
    source = next(r for r in ex.relationships if r.id == 'hbm_to_l2')
    ex.relationships.append(Relationship(id='sm_to_l2_recovered', source='a100_sm', target='l2_cache',
        kind='connects', variant=plans[-1].variant, citations=source.citations))
    local_source = next(r for r in ex.relationships if r.id == 'sm_to_shared_async')
    ex.relationships.append(Relationship(id='l2_to_shared_recovered', source='l2_cache', target='shared_memory',
        kind='connects', variant=plans[-1].variant, citations=local_source.citations))
    ex.missing = [m for m in ex.missing if not m.startswith('Total A100 device memory capacity')]
    bp = plans[-1]
    bp.memories[0].capacity.fact_id = 'device_hbm_capacity'
    bp.rationale = 'Test oracle after targeted source repair: external capacity and SM/cache access now have cited support.'
    bp.limitations = [m for m in bp.limitations if not m.startswith(('No SM-to-L2', 'Total A100 device memory'))]
    by_id = {c['id']: c for c in [*chunks, *body]}
    return ex, list(by_id.values()), bp


class FailedRunRegressionTests(unittest.TestCase):
    def test_actual_extraction_passes_against_its_original_retrieved_context(self):
        ex, chunks, _ = original_inputs()
        verify_extraction(ex, chunks)

    def test_all_three_failed_designs_report_true_gaps_without_l1_alias_failure(self):
        ex, _, plans = original_inputs()
        for plan in plans:
            with self.subTest(plan=plan.rationale[:60]):
                audit = {}
                p, _, _, gaps = compile_blueprint(plan, ex, [], 'hbm-proxy', topology_audit=audit)
                self.assertIsNone(p.hardware)
                self.assertEqual(len(gaps), 2)
                self.assertTrue(any('hbm2_device_memory.capacity' in gap for gap in gaps))
                self.assertTrue(any('topology is disconnected' in gap for gap in gaps))
                self.assertEqual(audit['memory_pools'][0]['member_resource_ids'], ['l1_cache', 'shared_memory'])

    def test_source_repair_reaches_valid_hardware_without_capacity_double_counting(self):
        ex, chunks, bp = repaired_inputs()
        verify_extraction(ex, chunks)
        audit = {}
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy', topology_audit=audit)
        verify_proposal(p, ex, assumptions, chunks)
        self.assertFalse(gaps)
        self.assertEqual(len(p.hardware.memories), 3)
        self.assertEqual(p.hardware.memories[0].capacity.value, 40e9)
        self.assertEqual(p.hardware.memories[2].capacity.value, 192000 * 108)
        self.assertEqual(len(p.hardware.memory_pools), 1)
        self.assertEqual(len(p.hardware.links), 4)
        self.assertEqual(len(assumptions), 3)
        self.assertNotIn('l1_cache', {n.id for n in p.hardware.memories})

    def test_pooling_is_refused_without_single_physical_block_proof(self):
        ex, _, plans = original_inputs()
        shared = next(r for r in ex.relationships if r.id == 'l1_to_smem_combined')
        shared.citations[0].quote = 'These are distinct physical caches connected by a bus.'
        with self.assertRaisesRegex(ValueError, 'Unmodeled source resource'):
            compile_blueprint(plans[-1], ex, [], 'hbm-proxy')

    def test_partial_shared_capacity_cannot_masquerade_as_combined_pool(self):
        ex, _, plans = original_inputs()
        plans[-1].memories[-1].capacity.fact_id = 'cache_capacity_shared_max'
        with self.assertRaisesRegex(ValueError, 'Unmodeled source resource'):
            compile_blueprint(plans[-1], ex, [], 'hbm-proxy')

    def test_pool_tampering_fails_independent_evidence_validation(self):
        ex, chunks, bp = repaired_inputs()
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        p.hardware.memory_pools[0].relationship_ids = ['hbm_to_l2']
        with self.assertRaisesRegex(ValueError, 'single-memory-block source proof'):
            verify_proposal(p, ex, assumptions, chunks)

    def test_strict_policy_does_not_fill_remaining_internal_rates(self):
        ex, _, bp = repaired_inputs()
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [])
        self.assertIsNone(p.hardware)
        self.assertFalse(assumptions)
        self.assertEqual(sum('bandwidth' in gap for gap in gaps), 3)


@unittest.skipUnless(AVAILABLE, 'Real failed-chat regression requires installed simulator dependencies')
class RealFailedChatBackendTests(unittest.TestCase):
    def test_repaired_source_graph_runs_through_actual_capture_mapping_and_schedule(self):
        from paper2perf.hwperf_worker import execute
        ex, chunks, bp = repaired_inputs()
        proposal, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy')
        verify_proposal(proposal, ex, assumptions, chunks)
        with TemporaryDirectory() as folder:
            metrics = execute(proposal, folder)
            self.assertGreater(metrics['makespan_seconds'], 0)
            self.assertEqual(metrics['backend'], 'hw-perf')
            self.assertTrue((Path(folder) / 'mapping-graph.json').is_file())


class SourceNavigationTests(unittest.TestCase):
    def test_table_of_contents_navigation_finds_capacity_body_without_numeric_defaults(self):
        from paper2perf.retrieval import follow_contents
        _, chunks, _ = original_inputs()
        body = load('failed-chat-body-chunks.json')
        followed = follow_contents(chunks, [*chunks, *body])
        self.assertTrue(any(c['page'] == 16 and 'the NVIDIA A100 GPU has 40 GB' in c['text']
                            for c in followed))

    def test_contract_queries_cover_capacity_and_disconnected_topology(self):
        from paper2perf.retrieval import contract_queries
        ex, _, plans = original_inputs()
        _, _, _, gaps = compile_blueprint(plans[-1], ex, [], 'hbm-proxy')
        queries = contract_queries(gaps)
        self.assertTrue(any('capacity' in q for q in queries))
        self.assertTrue(any('accessible' in q for q in queries))

    def test_actual_failed_run_can_repair_extraction_and_construct_with_model_calls_only_at_boundaries(self):
        from paper2perf.config import Settings
        from paper2perf.nodes import Nodes
        from paper2perf.store import Store
        original, original_chunks, plans = original_inputs()
        repaired, repaired_chunks, repaired_plan = repaired_inputs()
        calls = []
        extraction_count = 0
        design_count = 0
        def ask(task, payload, schema):
            nonlocal extraction_count, design_count
            calls.append((task, payload))
            if task == 'extract':
                extraction_count += 1
                return original if extraction_count == 1 else repaired
            if task == 'design':
                design_count += 1
                return plans[-1] if design_count == 1 else repaired_plan
            raise AssertionError('Unexpected planning/model request: ' + task)
        with TemporaryDirectory() as folder:
            store = Store(Path(folder));sid = store.new_session();rid = store.start_run(sid, {})
            client = Mock();client.ask.side_effect = ask;client.calls = 0
            nodes = Nodes(Settings(data_dir=Path(folder)), store, client, Mock(), rid)
            store.search = Mock(return_value=original_chunks)
            store.chunks = Mock(return_value=repaired_chunks)
            question = 'Simulate C=A@B dense FP16, no bias, M=256 N=256 K=256'
            state = {'run_id': rid, 'session_id': sid, 'mode': 'simulate', 'message': question,
                     'history': [], 'document_ids': ['original-source'], 'construction': 'grounded',
                     'bandwidth_policy': 'hbm-proxy', 'assumptions': [], 'deterministic_analysis': True}
            state.update(nodes.route(state));state.update(nodes.retrieve(state))
            state.update(nodes.extract(state));state.update(nodes.propose(state));state.update(nodes.validate(state))
            self.assertEqual(state['decision'], 'missing')
            gaps = list(state['validation_errors'])
            state.update(nodes.plan_retrieval(state))
            self.assertEqual(state['decision'], 'search')
            state.update(nodes.retrieve(state));state.update(nodes.extract(state))
            self.assertEqual(state['decision'], 'verified', str(state.get('extraction_errors')))
            state.update(nodes.propose(state));state.update(nodes.validate(state))
            self.assertEqual(state['decision'], 'ready', str(state.get('validation_errors')))
            second_extract = [payload for task, payload in calls if task == 'extract'][1]
            self.assertEqual(second_extract['modeling_requirements'], gaps)
            self.assertIsNotNone(second_extract['previous_verified_extraction'])
            self.assertEqual([task for task, _ in calls], ['extract', 'design', 'extract', 'design'])


class PhysicalPoolBoundaryTests(unittest.TestCase):
    def test_explicit_physical_pool_name_preserves_both_logical_views(self):
        ex, chunks, bp = repaired_inputs()
        bp.memories[-1].id = 'combined_local_pool'
        p, assumptions, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy')
        verify_proposal(p, ex, assumptions, chunks)
        self.assertFalse(gaps)
        self.assertEqual(p.hardware.memory_pools[0].memory_id, 'combined_local_pool')
        self.assertEqual(set(p.hardware.memory_pools[0].member_resource_ids), {'shared_memory', 'l1_cache'})
        self.assertEqual(p.hardware.memories[-1].capacity.value, 192000 * 108)
        self.assertEqual(len(p.hardware.memories), 3)

    def test_ambiguous_combined_pool_is_not_selected_by_name(self):
        ex, _, bp = repaired_inputs()
        bp.memories[-1].id = 'combined_local_pool'
        bp.memories[1].capacity = bp.memories[-1].capacity.model_copy(deep=True)
        with self.assertRaisesRegex(ValueError, 'Unmodeled source resource'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')


class MemoryPoolSafetyTests(unittest.TestCase):
    def test_logical_views_cannot_each_receive_the_full_combined_capacity(self):
        ex, _, bp = repaired_inputs()
        duplicate = bp.memories[-1].model_copy(deep=True)
        duplicate.id = 'l1_cache'
        bp.memories.append(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicated as multiple modeled memories'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_final_ir_cannot_erase_physical_sharing_provenance(self):
        ex, chunks, bp = repaired_inputs()
        p, assumptions, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        p.hardware.memory_pools = []
        with self.assertRaises(ValueError):
            verify_proposal(p, ex, assumptions, chunks)


class MemoryRouteContractTests(unittest.TestCase):
    def test_graph_connected_only_through_compute_is_a_source_retrieval_gap(self):
        ex, _, bp = repaired_inputs()
        ex.relationships = [r for r in ex.relationships if r.id != 'l2_to_shared_recovered']
        p, _, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertTrue(any('memory-only connectivity' in gap for gap in gaps))

    def test_typed_hardware_rejects_compute_as_memory_router(self):
        from paper2perf.schemas import Hardware
        ex, _, bp = repaired_inputs()
        p, _, _, _ = compile_blueprint(bp, ex, [], 'hbm-proxy')
        hardware = p.hardware.model_dump()
        hardware['links'] = [l for l in hardware['links'] if l['relationship_id'] != 'l2_to_shared_recovered']
        with self.assertRaisesRegex(ValueError, 'memory-only paths'):
            Hardware.model_validate(hardware)
