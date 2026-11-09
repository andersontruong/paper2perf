"""Regression coverage using the user's real, document-extracted A100 evidence.

These are test fixtures, never production fallbacks or simulator predictions.
"""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paper2perf.calculations import derive
from paper2perf.compiler import compile_blueprint
from paper2perf.evidence import EvidenceError, verify_extraction, verify_proposal
from paper2perf.schemas import Extraction, Gemm, HardwareBlueprint, QuantityReference

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    extraction = Extraction.model_validate(json.loads((ROOT / 'tests/fixtures/a100-evidence.json').read_text()))
    chunks = json.loads((ROOT / 'tests/fixtures/a100-chunks.json').read_text())
    blueprint = HardwareBlueprint.model_validate(json.loads((ROOT / 'examples/a100-whitepaper-blueprint.json').read_text()))
    blueprint.workload = Gemm(operator='gemm', M=256, N=256, K=256, dtype='fp16', description='Explicit replay GEMM')
    return extraction, chunks, blueprint


class CompilerTests(unittest.TestCase):
    def test_actual_extraction_matches_original_source_chunks(self):
        extraction, chunks, _ = inputs()
        verify_extraction(extraction, chunks)

    def test_strict_mode_reports_only_required_bandwidth_gaps(self):
        extraction, _, blueprint = inputs()
        p, assumptions, _, gaps = compile_blueprint(blueprint, extraction, [])
        self.assertIsNone(p.hardware)
        self.assertEqual(len(gaps), 3)
        self.assertTrue(all('bandwidth' in gap for gap in gaps))
        self.assertEqual(assumptions, [])
        self.assertFalse(any('occupancy' in gap for gap in gaps))

    def test_proxy_experiment_preserves_hierarchy_and_full_provenance(self):
        extraction, chunks, blueprint = inputs()
        p, assumptions, audit, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        verify_proposal(p, extraction, assumptions, chunks)
        self.assertEqual(gaps, [])
        self.assertEqual(len(p.hardware.memories), 3)
        self.assertEqual(len(p.hardware.links), 4)
        self.assertEqual(len(assumptions), 3)
        local = p.hardware.memories[2].capacity
        self.assertEqual(local.value, 192 * 1000 * 108)
        self.assertEqual(local.derivation, 'aggregate_capacity')
        self.assertEqual(len(local.fact_ids), 2)
        self.assertEqual(p.hardware.computes[0].id, 'a100_sm')
        self.assertEqual(p.hardware.computes[0].throughput.value, 312e12)
        for entry in audit:
            if 'proxy_source_fact_id' in entry:
                self.assertEqual(entry['basis'], 'assumed')
                self.assertEqual(entry['fact_ids'], [])
                self.assertIn('not a specification', entry['rationale'])

    def test_values_are_computed_from_evidence_not_a100_constants(self):
        extraction, _, blueprint = inputs()
        for fact in extraction.facts:
            if fact.id == 'a100_sxm_mem_bandwidth':
                fact.value = 1234
            if fact.id == 'a100_sm_count':
                fact.value = 96
        p, assumptions, _, _ = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertEqual(p.hardware.memories[2].capacity.value, 192000 * 96)
        self.assertEqual(p.hardware.links[0].bandwidth.value, 1234e9)
        self.assertTrue(all(a.value == 1234e9 for a in assumptions))

    def test_mixed_variant_capacity_count_cannot_be_multiplied(self):
        extraction, _, blueprint = inputs()
        for fact in extraction.facts:
            if fact.id == 'a100_sm_count':
                fact.variant = 'Another product'
        p, _, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertTrue(any('variant' in gap for gap in gaps))

    def test_proxy_does_not_cover_invalid_fact_reference(self):
        extraction, _, blueprint = inputs()
        blueprint.links[1].bandwidth.fact_id = 'fabricated_id'
        p, _, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertTrue(gaps)

    def test_proxy_source_must_be_documented_device_memory_bandwidth(self):
        extraction, _, blueprint = inputs()
        blueprint.proxy_bandwidth_fact_id = 'a100_fp16_tensor_dense_throughput'
        p, _, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertTrue(any('device-memory' in gap for gap in gaps))

    def test_tampered_derived_capacity_is_rejected(self):
        extraction, chunks, blueprint = inputs()
        p, assumptions, _, _ = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        p.hardware.memories[2].capacity.value *= 2
        with self.assertRaisesRegex(EvidenceError, 'disagrees'):
            verify_proposal(p, extraction, assumptions, chunks)

    def test_sparse_throughput_cannot_substitute_for_dense(self):
        extraction, chunks, blueprint = inputs()
        blueprint.computes[0].throughput.fact_id = 'a100_fp16_tensor_sparse_throughput'
        p, assumptions, _, _ = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        with self.assertRaisesRegex(EvidenceError, 'dense throughput'):
            verify_proposal(p, extraction, assumptions, chunks)

    def test_clock_conversion_uses_two_matching_source_facts(self):
        extraction, _, _ = inputs()
        from paper2perf.schemas import Fact
        context = json.loads((ROOT / 'tests/fixtures/a100-clock-context.json').read_text())
        clock = Fact.model_validate(context['fact'])
        clock.id = 'clock'
        _, chunks, _ = inputs()
        checked = extraction.model_copy(deep=True)
        checked.facts.append(clock)
        verify_extraction(checked, [*chunks, context['chunk']])
        facts = {f.id: f for f in [*extraction.facts, clock]}
        q = derive(QuantityReference(fact_id='a100_l2_read_bandwidth', clock_fact_id='clock'), facts, 'byte/s')
        self.assertEqual(q.value, 5120 * 1410e6)
        self.assertEqual(q.derivation, 'per_clock_rate')
        clock.variant = 'Different GPU'
        with self.assertRaisesRegex(ValueError, 'variants'):
            derive(QuantityReference(fact_id='a100_l2_read_bandwidth', clock_fact_id='clock'), facts, 'byte/s')

    def test_wrong_count_dimension_is_rejected(self):
        extraction, _, blueprint = inputs()
        blueprint.memories[2].capacity.count_fact_id = 'a100_tensor_cores_per_sm'
        p, _, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertTrue(any('SM count' in gap for gap in gaps))

    def test_containment_access_requires_actual_parent_compute(self):
        extraction, chunks, blueprint = inputs()
        p, assumptions, _, _ = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        p.hardware.links[-1].topology_basis = 'documented'
        with self.assertRaisesRegex(EvidenceError, 'connectivity'):
            verify_proposal(p, extraction, assumptions, chunks)

    def test_dropping_documented_connection_is_not_accepted(self):
        extraction, chunks, blueprint = inputs()
        p, assumptions, _, _ = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        p.hardware.links = [l for l in p.hardware.links if l.relationship_id != 'sm_connects_l2']
        with self.assertRaisesRegex(EvidenceError, 'omits documented'):
            verify_proposal(p, extraction, assumptions, chunks)

    def test_contained_resource_cannot_be_silently_deleted(self):
        extraction, chunks, blueprint = inputs()
        p, assumptions, _, _ = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        # Even if another extraction only records containment, preserve that level.
        extraction.relationships = [r for r in extraction.relationships if r.id != 'l1_smem_connects_l2']
        p.hardware.memories = p.hardware.memories[:2]
        p.hardware.links = [l for l in p.hardware.links if 'a100_l1_smem' not in {l.source, l.target}]
        with self.assertRaisesRegex(EvidenceError, 'inside the selected compute'):
            verify_proposal(p, extraction, assumptions, chunks)


AVAILABLE = all(importlib.util.find_spec(name) for name in [
    'torch', 'networkx', 'pydot', 'pysmt', 'z3', 'parabolic', 'hw_model', 'dataflow_mapper'])


@unittest.skipUnless(AVAILABLE, 'Real A100 capture/mapping requires installed simulator dependencies')
class RealA100SimulationTests(unittest.TestCase):
    def test_real_a100_hierarchy_capture_map_schedule(self):
        from paper2perf.hwperf_worker import execute
        extraction, chunks, blueprint = inputs()
        p, assumptions, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        verify_proposal(p, extraction, assumptions, chunks)
        with TemporaryDirectory() as folder:
            result = execute(p, folder)
            self.assertGreater(result['makespan_seconds'], 0)
            self.assertEqual(result['backend'], 'hw-perf')
            graph = json.loads((Path(folder) / 'hardware-graph.json').read_text())
            self.assertEqual(len(graph['nodes']), 4)
            self.assertEqual(len(graph.get('edges', graph.get('links', []))), 4)
            self.assertTrue((Path(folder) / 'mapping-graph.json').is_file())
            self.assertTrue((Path(folder) / 'events.json').is_file())


class GroundedNodeTests(unittest.TestCase):
    def setUp(self):
        from paper2perf.config import Settings
        from paper2perf.nodes import Nodes
        from paper2perf.store import Store
        from unittest.mock import Mock
        self.temp = TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.sid = self.store.new_session()
        self.rid = self.store.start_run(self.sid, {})
        self.client = Mock()
        self.client.calls = 0
        self.client.ask.side_effect = AssertionError('Unexpected model call in reference-only replay')
        self.nodes = Nodes(Settings(data_dir=Path(self.temp.name)), self.store,
                           self.client, Mock(), self.rid)
        extraction, chunks, blueprint = inputs()
        self.state = {'run_id': self.rid, 'session_id': self.sid,
                      'construction': 'grounded', 'bandwidth_policy': 'hbm-proxy',
                      'extraction': extraction.model_dump(), 'seed_extraction': extraction.model_dump(),
                      'chunks': chunks, 'blueprint': blueprint.model_dump(),
                      'assumptions': [], 'extraction_attempts': 0, 'extraction_repairs': 0,
                      'message': 'Simulate dense fp16 no bias M=256 N=256 K=256',
                      'query': 'Simulate dense fp16 no bias M=256 N=256 K=256',
                      'history': [], 'validation_errors': [], 'action': 'simulate',
                      'status': 'running', 'deterministic_analysis': True}

    def tearDown(self):
        self.temp.cleanup()

    def test_revalidate_construct_validate_without_model_requests(self):
        self.state.update(self.nodes.extract(self.state))
        self.state.update(self.nodes.propose(self.state))
        result = self.nodes.validate(self.state)
        self.assertEqual(result['decision'], 'ready')
        self.client.ask.assert_not_called()
        for artifact in ['blueprint.json', 'derivations.json', 'assumptions.json',
                         'hardware.json', 'workload.py', 'validation.json']:
            self.assertTrue((self.store.root / 'runs' / self.rid / artifact).is_file())

    def test_seed_evidence_is_revalidated_not_blindly_trusted(self):
        self.state['seed_extraction']['facts'][0]['value'] = 999999
        result = self.nodes.extract(self.state)
        self.assertNotEqual(result['decision'], 'verified')
        self.assertIsNone(result['extraction'])

    def test_policy_is_not_silently_enabled(self):
        self.state['bandwidth_policy'] = 'strict'
        self.state.update(self.nodes.propose(self.state))
        result = self.nodes.validate(self.state)
        self.assertEqual(result['decision'], 'missing')
        self.assertEqual(self.state['assumptions'], [])
        self.state.update(result)
        report = self.nodes.report(self.state)
        self.assertEqual(report['status'], 'needs_information')
        self.assertIn('No simulation was run', report['response'])

    def test_contract_shape_cannot_be_overridden_by_blueprint(self):
        self.state['workload_contract'] = {**self.state['blueprint']['workload'], 'M': 512}
        result = self.nodes.propose(self.state)
        self.assertIsNone(result['proposal'])
        self.assertIn('workload', result['validation_errors'][0])

    def test_proxy_assumptions_remain_unique_across_repairs(self):
        self.state.update(self.nodes.propose(self.state))
        self.state.update(self.nodes.propose(self.state))
        self.assertEqual(len(self.state['assumptions']), 3)


class EvidenceReplayTests(unittest.TestCase):
    def setUp(self):
        from paper2perf.config import Settings
        from paper2perf.runner import InvestigationRunner
        from paper2perf.store import Store
        self.temp = TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.sid = self.store.new_session()
        extraction, chunks, _ = inputs()
        pages = [c['text'] for c in chunks]
        self.did = self.store.add_document(self.sid, 'source-excerpts.txt', '.txt',
                                           '\n'.join(pages).encode(), pages)
        remap = {chunk['id']: f'{self.did}-p{i}-c0' for i, chunk in enumerate(chunks, 1)}
        for entry in [*extraction.facts, *extraction.relationships]:
            for citation in entry.citations:
                citation.chunk_id = remap[citation.chunk_id]
        self.old = self.store.start_run(self.sid, {'document_ids': [self.did]})
        self.store.artifact(self.old, 'evidence.json', extraction.model_dump())
        self.new = self.store.start_run(self.sid, {})
        self.request = {'evidence_run_id': self.old, 'document_ids': [self.did]}
        self.runner = InvestigationRunner(Settings(data_dir=self.store.root), self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_replay_checks_originals_and_records_evidence_hash(self):
        result = self.runner.replay_evidence(self.sid, self.new, self.request)
        self.assertEqual(len(result['facts']), 12)
        receipt = json.loads((self.store.root / 'runs' / self.new / 'evidence-reuse.json').read_text())
        self.assertTrue(receipt['revalidated'])
        self.assertEqual(len(receipt['sha256']), 64)
        self.assertEqual(receipt['source_documents'][0]['document_id'], self.did)

    def test_foreign_session_evidence_is_rejected(self):
        other = self.store.new_session()
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            self.runner.replay_evidence(other, self.new, self.request)

    def test_changed_original_document_is_rejected(self):
        (self.store.root / 'documents' / (self.did + '.txt')).write_text('Changed source')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.runner.replay_evidence(self.sid, self.new, self.request)

    def test_replay_cannot_drop_source_documents(self):
        with self.assertRaisesRegex(ValueError, 'all original evidence'):
            self.runner.replay_evidence(self.sid, self.new, {**self.request, 'document_ids': []})


@unittest.skipUnless(AVAILABLE and importlib.util.find_spec('langgraph'),
                     'Real replay requires LangGraph and installed hw-perf dependencies')
class RealReplayWorkflowTests(unittest.TestCase):
    setUp = EvidenceReplayTests.setUp
    tearDown = EvidenceReplayTests.tearDown

    def test_full_langgraph_replay_uses_real_backend_and_no_model(self):
        from unittest.mock import patch
        from paper2perf.schemas import ChatRequest
        _, _, blueprint = inputs()
        request = ChatRequest(
            message='Simulate dense fp16 GEMM C=A@B no bias M=256 N=256 K=256',
            mode='simulate', document_ids=[self.did], evidence_run_id=self.old,
            blueprint=blueprint.model_dump(), bandwidth_policy='hbm-proxy',
            workload_contract=blueprint.workload, deterministic_analysis=True).model_dump()
        with patch('paper2perf.llm.ModelClient.ask', side_effect=AssertionError('Unexpected model request')):
            result = self.runner.run(self.sid, request)
        folder = self.store.root / 'runs' / result['id']
        stderr = (folder / 'backend-stderr.txt').read_text() if (folder / 'backend-stderr.txt').exists() else ''
        self.assertEqual(result['status'], 'completed', str(result['result']) + stderr)
        self.assertGreater(result['result']['metrics']['makespan_seconds'], 0)
        self.assertEqual(result['result']['metrics']['backend'], 'hw-perf')
        self.assertEqual(list(folder.glob('llm-*-request.json')), [])
        self.assertTrue((folder / 'evidence-reuse.json').is_file())
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertTrue(manifest['simulation_executed'])
        self.assertEqual(manifest['budgets']['model_calls'], 0)
        self.assertIn('HBM-proxy', result['result']['response'])


class DesignContractTests(unittest.TestCase):
    def test_unclassified_missing_details_require_repair_not_silent_dismissal(self):
        extraction, _, blueprint = inputs()
        blueprint.missing = ['No documented register-file per-clock bandwidth.']
        with self.assertRaisesRegex(ValueError, 'design repair, not another source search'):
            compile_blueprint(blueprint, extraction, [], 'hbm-proxy')

    def test_real_scope_uncertainty_still_blocks_with_proxy_enabled(self):
        from paper2perf.schemas import DesignBlocker
        extraction, _, blueprint = inputs()
        blueprint.blockers = [DesignBlocker(category='source_scope',
            reason='Selected local-memory capacity cannot be assigned to this device variant.',
            reference_id=blueprint.memories[2].capacity.fact_id)]
        p, _, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertIn('source_scope', gaps[0])

    def test_fidelity_limitations_do_not_block_a_resolved_contract(self):
        extraction, chunks, blueprint = inputs()
        blueprint.limitations += ['Register-file timing and per-wave schedules are not modeled.']
        p, assumptions, _, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
        self.assertEqual(gaps, [])
        verify_proposal(p, extraction, assumptions, chunks)
        self.assertTrue(any('per-wave' in note for note in p.limitations))


class FreshDesignRepairTests(unittest.TestCase):
    setUp = GroundedNodeTests.setUp
    tearDown = GroundedNodeTests.tearDown

    def test_user_failure_enters_design_repair_then_validates_without_retrieval(self):
        from paper2perf.schemas import HardwareBlueprint
        self.state['blueprint'] = None
        self.state['repairs'] = 0
        _, _, original = inputs()
        rejected = original.model_copy(deep=True)
        concerns = [
            'No documented shared-memory bandwidth (bytes/clk per SM) for the L1/SMEM level.',
            'No documented L1 data-cache load/bandwidth per SM.',
            'No documented register-file per-clock bandwidth.',
            'No documented L2 or DRAM access latency values.',
            'No documented mapping/scheduling of the GEMM onto SM tiles or waves.',
            'Accumulator-type timing is not distinguished for this workload.',
            'Maximum usable SMEM scope is ambiguous relative to combined capacity.',
        ]
        rejected.missing = concerns
        corrected = original.model_copy(deep=True)
        corrected.limitations += ['Internal bandwidth is explicitly proxied; register-file, '
                                  'access latency, accumulator timing and SM waves are not modeled. '
                                  'The combined pool uses combined capacity rather than maximum usable SMEM.']
        payloads = []
        def design(task, payload, schema):
            self.assertEqual(task, 'design')
            self.assertIs(schema, HardwareBlueprint)
            payloads.append(payload)
            return rejected if len(payloads) == 1 else corrected
        self.client.ask.side_effect = design
        self.state.update(self.nodes.propose(self.state))
        first = self.nodes.validate(self.state)
        self.assertEqual(first['decision'], 'invalid')
        self.state.update(first)
        self.state.update(self.nodes.repair(self.state))
        self.state.update(self.nodes.propose(self.state))
        self.assertEqual(self.nodes.validate(self.state)['decision'], 'ready')
        self.assertEqual(len(payloads), 2)
        self.assertIn('register-file', payloads[1]['diagnostics'][0])
        self.assertEqual(payloads[1]['previous_blueprint']['missing'], concerns)
        self.assertIn('hbm-proxy', payloads[1]['adapter_contract']['bandwidth_resolution'])
        diagnostics = json.loads((self.store.root / 'runs' / self.rid / 'design-validation.json').read_text())
        self.assertTrue(diagnostics['passed'])
