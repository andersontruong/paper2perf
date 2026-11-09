import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from paper2perf.config import Settings
from paper2perf.graph import build_graph
from paper2perf.hwperf import BackendUnavailable, verify_bundled_source
from paper2perf.nodes import Nodes
from paper2perf.schemas import Analysis, Answer, RetrievalPlan
from paper2perf.store import Store
from fixtures import CHUNKS, CITE, extraction, proposal


class ScriptedModel:
    """Unit-test double; never reachable from the production application."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = 0

    def ask(self, task, payload, schema):
        self.calls += 1
        value = self.responses[task]
        return value(payload) if callable(value) else value


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.sid = self.store.new_session()
        self.rid = self.store.start_run(self.sid, {})
        self.settings = Settings(data_dir=Path(self.temp.name))
        self.client = ScriptedModel()
        self.backend = SimpleNamespace(simulate=lambda *args: (_ for _ in ()).throw(
            BackendUnavailable("Missing runtime dependencies")))
        self.nodes = Nodes(self.settings, self.store, self.client, self.backend, self.rid)
        self.state = {"run_id": self.rid, "session_id": self.sid, "mode": "simulate",
                      "message": "Simulate a 16 by 16 by 16 dense FP16 GEMM", "query": "TestChip",
                      "history": [], "document_ids": [], "assumptions": [], "chunks": CHUNKS,
                      "extraction": extraction().model_dump(), "proposal": proposal().model_dump(),
                      "action": "simulate", "status": "running", "repairs": 0,
                      "retrieval_round": 1}

    def tearDown(self):
        self.temp.cleanup()

    def test_validated_contract(self):
        self.assertEqual(self.nodes.validate(self.state)["decision"], "ready")

    def test_unsupported_shape_not_silently_invented(self):
        self.state["proposal"]["workload"]["M"] = 512
        result = self.nodes.validate(self.state)
        self.assertEqual(result["decision"], "invalid")
        self.assertIn("dimensions", result["validation_errors"][0])

    def test_missing_model_requests_evidence(self):
        self.state["proposal"]["hardware"] = None
        self.state["proposal"]["missing"] = ["Need L2 bandwidth"]
        self.assertEqual(self.nodes.validate(self.state)["decision"], "missing")

    def test_backend_failure_never_produces_metrics(self):
        result = self.nodes.simulate(self.state)
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("metrics", result)

    def test_missing_information_report_is_not_success(self):
        self.state["validation_errors"] = ["Missing source evidence"]
        result = self.nodes.report(self.state)
        self.assertEqual(result["status"], "needs_information")
        self.assertIn("No simulation was run", result["response"])

    def test_retrieval_budget_is_enforced_without_model_call(self):
        self.state["retrieval_round"] = self.settings.max_retrievals
        self.assertEqual(self.nodes.plan_retrieval(self.state)["decision"], "stop")
        self.assertEqual(self.client.calls, 0)

    def test_analysis_text_cannot_hide_wrong_number_behind_correct_reference(self):
        self.state["metrics"] = {"makespan_seconds": 0.001}
        self.client.responses["analyze"] = Analysis.model_validate({"findings": [{
            "kind": "observed", "text": "The makespan is 50 seconds.",
            "metric": "makespan_seconds", "value": 0.001
        }]})
        result = self.nodes.analyze(self.state)
        self.assertEqual(result["status"], "analysis_failed")

    def test_invalid_structured_proposal_enters_repair_path(self):
        from paper2perf.llm import StructuredOutputError
        def invalid(_):
            raise StructuredOutputError("Invalid contract")
        self.client.responses["propose"] = invalid
        result = self.nodes.propose(self.state)
        self.state.update(result)
        self.assertIsNone(self.state["proposal"])
        self.assertEqual(self.nodes.validate(self.state)["decision"], "invalid")


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
class GraphTests(unittest.TestCase):
    setUp = NodeTests.setUp
    tearDown = NodeTests.tearDown

    def test_question_routes_without_simulation(self):
        self.state["mode"], self.state["message"] = "question", "What memory capacity?"
        did = self.store.add_document(self.sid, "source.txt", ".txt", b"doc", [CHUNKS[0]["text"]])
        chunk = self.store.chunks([did])[0]
        self.state["document_ids"] = [did]
        self.client.responses["answer"] = Answer.model_validate({"claims": [{
            "text": "Memory capacity is 8 GB.",
            "citations": [{"chunk_id": chunk["id"], "quote": chunk["text"]}]
        }]})
        result = build_graph(self.nodes).invoke(self.state)
        self.assertEqual(result["status"], "completed")
        events = self.store.run(self.sid, self.rid)["events"]
        self.assertNotIn("simulate", [e["stage"] for e in events])

    def test_missing_evidence_terminates(self):
        self.client.responses = {"extract": extraction(), "propose": proposal(),
                                 "retrieve": RetrievalPlan(queries=[], reason="Unavailable")}
        self.client.responses["propose"].hardware = None
        self.client.responses["propose"].missing = ["Need bandwidth"]
        # Retrieve original chunk IDs by using a controlled test-only store search.
        self.store.search = lambda *args, **kwargs: CHUNKS
        self.state["document_ids"] = ["source"]
        result = build_graph(self.nodes).invoke(self.state)
        self.assertEqual(result["status"], "needs_information")

    def test_sqlite_checkpoints_persist_state(self):
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        self.state["mode"] = "inspect"
        with sqlite3.connect(Path(self.temp.name) / "checkpoint.sqlite",
                             check_same_thread=False) as connection:
            graph = build_graph(self.nodes, SqliteSaver(connection))
            config = {"configurable": {"thread_id": self.rid}}
            graph.invoke(self.state, config)
            saved = graph.get_state(config)
            self.assertEqual(saved.values["status"], "completed")
            self.assertFalse(saved.next)


class VendoredSourceTests(unittest.TestCase):
    def test_all_bundled_files_match_supplied_archive(self):
        import hashlib
        import json
        root = Path(__file__).resolve().parents[1] / "vendor" / "hw-perf"
        manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
        self.assertGreater(len(manifest["files"]), 30)
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest(),
                             entry["sha256"], entry["path"])
        self.assertFalse((root / ".gitmodules").exists())


class AnswerRepairTests(unittest.TestCase):
    setUp = NodeTests.setUp
    tearDown = NodeTests.tearDown

    def test_invalid_answer_is_repaired_with_diagnostics(self):
        good = Answer.model_validate({"claims": [{"text": "Memory capacity is 8 GB.",
                                                  "citations": [CITE]}]})
        bad = good.model_copy(deep=True)
        bad.claims[0].text = "Memory capacity is 80 GB."
        calls = []
        def answer(payload):
            calls.append(payload)
            return bad if len(calls) == 1 else good
        self.client.responses["answer"] = answer
        first = self.nodes.answer(self.state)
        self.assertEqual(first["decision"], "retry")
        self.assertNotIn("response", first)
        self.state.update(first)
        second = self.nodes.answer(self.state)
        self.assertEqual(second["decision"], "done")
        self.assertIn("8 GB", second["response"])
        self.assertIn("unsupported number 80", calls[1]["validation_errors"][0])
        self.assertEqual(calls[1]["previous_answer"]["claims"][0]["text"], bad.claims[0].text)

    def test_persistently_invalid_answer_stops_without_accepting_claims(self):
        self.state["action"] = "question"
        self.client.responses["answer"] = Answer.model_validate({"claims": [{
            "text": "Memory capacity is 80 GB.", "citations": [CITE]
        }]})
        for _ in range(self.settings.max_repairs + 1):
            result = self.nodes.answer(self.state)
            self.state.update(result)
        self.assertEqual(result["decision"], "unverified")
        self.assertEqual(result["status"], "needs_information")
        self.assertNotIn("Memory capacity is 80", result["response"])
        self.assertTrue((self.store.root / "runs" / self.rid / "answer-validation.json").exists())
        self.assertEqual(self.nodes.report(self.state)["status"], "needs_information")

    def test_result_inspection_uses_the_same_answer_repair_path(self):
        self.state["last_result"] = {"run_id": "prior", "metrics": {"makespan_seconds": 0.001}}
        def invalid(payload):
            chunk = payload["chunks"][0]
            return Answer.model_validate({"claims": [{"text": "Makespan is 99 seconds.",
                "citations": [{"chunk_id": chunk["id"], "quote": chunk["text"]}]}]})
        self.client.responses["answer"] = invalid
        self.assertEqual(self.nodes.inspect(self.state)["decision"], "retry")


class ExtractionRepairTests(unittest.TestCase):
    setUp = NodeTests.setUp
    tearDown = NodeTests.tearDown

    def test_rejected_quote_is_repaired_using_explicit_diagnostics(self):
        valid = extraction()
        bad = valid.model_copy(deep=True)
        bad.facts[0].citations[0].quote = "A paraphrase that does not occur in the original"
        calls = []
        def extract(payload):
            calls.append(payload)
            return bad if len(calls) == 1 else valid
        self.client.responses["extract"] = extract
        first = self.nodes.extract(self.state)
        self.assertEqual(first["decision"], "retry")
        self.assertIsNone(first["extraction"])
        self.assertFalse((self.store.root / "runs" / self.rid / "evidence.json").exists())
        self.state.update(first)
        second = self.nodes.extract(self.state)
        self.assertEqual(second["decision"], "verified")
        self.assertEqual(calls[1]["allowed_chunk_ids"], [c["id"] for c in CHUNKS])
        self.assertIn("quote does not match", calls[1]["validation_errors"][0])
        self.assertEqual(calls[1]["previous_extraction"], bad.model_dump())
        self.assertTrue((self.store.root / "runs" / self.rid / "evidence.json").exists())

    def test_wrong_chunk_reference_is_retried(self):
        bad = extraction()
        bad.facts[0].citations[0].chunk_id = "wrong-p24-c0"
        self.client.responses["extract"] = bad
        result = self.nodes.extract(self.state)
        self.assertEqual(result["decision"], "retry")
        self.assertIn("not in the retrieved context", result["extraction_errors"][0])

    def test_exhausted_extraction_cannot_reach_a_model_or_simulation(self):
        bad = extraction()
        bad.facts[0].value = 999
        self.client.responses["extract"] = bad
        for _ in range(self.settings.max_repairs + 1):
            result = self.nodes.extract(self.state)
            self.state.update(result)
        self.assertEqual(result["decision"], "unverified")
        self.assertEqual(result["status"], "needs_information")
        self.assertIsNone(result["extraction"])
        self.assertIsNone(result["proposal"])
        report = self.nodes.report(self.state)
        self.assertEqual(report["status"], "needs_information")
        self.assertIn("No simulation was run", report["response"])

    def test_malformed_extraction_enters_the_repair_path(self):
        from paper2perf.llm import StructuredOutputError
        def invalid(payload):
            raise StructuredOutputError("Missing fact citations")
        self.client.responses["extract"] = invalid
        result = self.nodes.extract(self.state)
        self.assertEqual(result["decision"], "retry")
        self.assertIn("Missing fact citations", result["extraction_errors"][0])


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
class ExtractionRepairGraphTests(unittest.TestCase):
    setUp = NodeTests.setUp
    tearDown = NodeTests.tearDown

    def test_graph_retries_extraction_before_model_construction(self):
        self.store.search = lambda *args, **kwargs: CHUNKS
        self.state["document_ids"] = ["source"]
        bad = extraction()
        bad.facts[0].citations[0].chunk_id = "wrong-p24-c0"
        calls = []
        def extract(payload):
            calls.append(payload)
            return bad if len(calls) == 1 else extraction()
        self.client.responses = {"extract": extract, "propose": proposal()}
        # The test backend deliberately returns blocked, after valid model construction.
        result = build_graph(self.nodes).invoke(self.state)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["extraction_repairs"], 1)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["decision"], "stop")

    def test_graph_stops_before_propose_on_persistently_invalid_extraction(self):
        self.store.search = lambda *args, **kwargs: CHUNKS
        self.state["document_ids"] = ["source"]
        bad = extraction()
        bad.facts[0].citations[0].chunk_id = "wrong-p24-c0"
        self.client.responses = {"extract": bad}
        result = build_graph(self.nodes).invoke(self.state)
        self.assertEqual(result["status"], "needs_information")
        self.assertEqual(self.client.calls, self.settings.max_repairs + 1)
        events = self.store.run(self.sid, self.rid)["events"]
        self.assertNotIn("propose", [e["stage"] for e in events])
        self.assertNotIn("simulate", [e["stage"] for e in events])


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph dependency not installed")
class AnswerRepairGraphTests(unittest.TestCase):
    setUp = NodeTests.setUp
    tearDown = NodeTests.tearDown

    def question_state(self):
        self.state["mode"], self.state["message"] = "question", "What memory capacity?"
        self.store.search = lambda *args, **kwargs: CHUNKS
        self.state["document_ids"] = ["source"]

    def test_graph_retries_rejected_answer_before_reporting(self):
        self.question_state()
        calls = []
        def answer(payload):
            calls.append(payload)
            return Answer.model_validate({"claims": [{
                "text": "Memory capacity is 80 GB." if len(calls) == 1 else "Memory capacity is 8 GB.",
                "citations": [CITE]
            }]})
        self.client.responses["answer"] = answer
        result = build_graph(self.nodes).invoke(self.state)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer_repairs"], 1)
        self.assertEqual(len(calls), 2)

    def test_graph_terminates_when_every_answer_is_rejected(self):
        self.question_state()
        self.client.responses["answer"] = Answer.model_validate({"claims": [{
            "text": "Memory capacity is 80 GB.", "citations": [CITE]
        }]})
        result = build_graph(self.nodes).invoke(self.state)
        self.assertEqual(result["status"], "needs_information")
        self.assertEqual(self.client.calls, self.settings.max_repairs + 1)
        self.assertEqual(result["references"], [])


if __name__ == "__main__":
    unittest.main()
