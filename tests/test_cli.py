from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paper2perf.cli import configure_line_editing, main
from paper2perf.config import Settings
from paper2perf.documents import ingest_documents
from paper2perf.runner import InvestigationRunner, execution_lock
from paper2perf.store import Store


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(data_dir=self.root / "data")
        self.source = self.root / "original document.txt"
        self.source.write_text("Synthetic test device has 8 GB memory. Memory connects to compute.")
        self.store = Store(self.settings.data_dir)
        self.requests = []
        self.config = patch("paper2perf.cli.Settings.from_env", return_value=self.settings)
        self.config.start()

    def tearDown(self):
        self.config.stop()
        self.temp.cleanup()

    def invoke(self, args, messages=None):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            if messages is None:
                code = main(args)
            else:
                with patch("builtins.input", side_effect=messages):
                    code = main(args)
        return code, out.getvalue(), err.getvalue()

    def fake_execute(self, runner, sid, rid, request, resume, on_event):
        """Test double for the model/graph boundary, not an alternative runtime backend."""
        self.requests.append({"request": request, "resume": resume, "run_id": rid})
        if on_event:
            on_event("validate", "Synthetic stage notification")
        result = {"response": "Synthetic test-only answer", "run_id": rid, "references": []}
        runner.store.artifact(rid, "evidence.json", {"facts": []})
        runner.store.set_run(rid, "completed", result)
        runner.store.message(sid, "assistant", result["response"], rid)

    def test_one_shot_json_and_progress_use_separate_streams(self):
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, out, err = self.invoke(["run", "--documents", str(self.source),
                                          "--question", "Describe memory", "--mode", "question", "--json"])
        run = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(run["status"], "completed")
        self.assertIn("[validate]", err)
        self.assertEqual(self.requests[0]["request"]["mode"], "question")
        self.assertEqual(len(self.store.documents(run["session"])), 1)

    def test_reused_session_reuses_original_documents(self):
        sid = self.store.new_session()
        ids = ingest_documents(self.settings, self.store, sid, [self.source])
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, _, _ = self.invoke(["run", "--session", sid, "--question", "Describe memory", "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(self.requests[0]["request"]["document_ids"], ids)

    def test_chat_accepts_sources_modes_assumptions_and_followups(self):
        messages = [f'/add "{self.source}"', "/documents", "/mode question",
                    "/assume bandwidth 1000 byte/s Explicit test assumption", "/assumptions",
                    "What memory is documented?", "/clear-assumptions", "Explain it again", "/quit"]
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, out, _ = self.invoke(["chat", "--quiet"], messages)
        self.assertEqual(code, 0)
        self.assertIn(self.source.name, out)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["request"]["assumptions"][0]["id"], "bandwidth")
        self.assertEqual(self.requests[1]["request"]["assumptions"], [])
        self.assertEqual(self.requests[1]["request"]["mode"], "question")
        self.assertEqual(len(self.store.sessions()), 1)

    def test_assumptions_file_is_validated_and_passed_to_runner(self):
        path = self.root / "assumptions.json"
        path.write_text(json.dumps([{"id": "bandwidth", "value": 1000, "unit": "byte/s",
                                     "rationale": "Explicit test-only assumption"}]))
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, _, _ = self.invoke(["run", "--documents", str(self.source), "--question", "Model it",
                                      "--assumptions", str(path), "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(self.requests[0]["request"]["assumptions"][0]["value"], 1000)

    def test_inspect_and_artifact_read_never_call_a_model(self):
        sid = self.store.new_session()
        rid = self.store.start_run(sid, {"message": "test"})
        self.store.set_run(rid, "completed", {"response": "Saved result"})
        self.store.artifact(rid, "metrics.json", {"makespan_seconds": 0.001})
        with patch("paper2perf.llm.ModelClient.ask", side_effect=AssertionError("Unexpected model call")):
            code, out, _ = self.invoke(["inspect", rid])
            artifact_code, data, _ = self.invoke(["inspect", rid, "--artifact", "metrics.json", "--json"])
        self.assertEqual(code, 0)
        self.assertIn("Saved result", out)
        self.assertEqual(artifact_code, 0)
        self.assertEqual(json.loads(data)["makespan_seconds"], 0.001)

    def test_resume_preserves_run_id_and_does_not_repeat_user_message(self):
        sid = self.store.new_session()
        ingest_documents(self.settings, self.store, sid, [self.source])
        request = {"message": "Original question", "mode": "question", "assumptions": [], "document_ids": []}
        rid = self.store.start_run(sid, request)
        self.store.message(sid, "user", request["message"], rid)
        self.store.set_run(rid, "interrupted")
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, out, _ = self.invoke(["resume", rid, "--json", "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["id"], rid)
        self.assertTrue(self.requests[0]["resume"])
        self.assertEqual(len([m for m in self.store.history(sid) if m["role"] == "user"]), 1)

    def test_failed_execution_is_nonzero_and_has_machine_readable_status(self):
        with patch.object(InvestigationRunner, "available", return_value=False):
            code, out, _ = self.invoke(["run", "--documents", str(self.source),
                                       "--question", "Describe memory", "--json", "--quiet"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["status"], "failed")
        self.assertIn("LangGraph", json.loads(out)["result"]["response"])

    def test_invalid_sources_and_session_are_input_errors(self):
        bad = self.root / "bad.py"
        bad.write_text("print('not a source document')")
        code, _, err = self.invoke(["run", "--documents", str(bad), "--question", "Test"])
        self.assertEqual(code, 2)
        self.assertIn("PDF", err)
        code, _, err = self.invoke(["chat", "--session", "unknown"])
        self.assertEqual(code, 2)
        self.assertIn("Conversation not found", err)

    def test_inspect_rejects_path_escape(self):
        sid = self.store.new_session()
        rid = self.store.start_run(sid, {})
        code, _, err = self.invoke(["inspect", rid, "--artifact", "../catalog.sqlite"])
        self.assertEqual(code, 2)
        self.assertIn("filename", err)

    def test_sessions_can_be_listed_without_model_credentials(self):
        sid = self.store.new_session()
        code, out, _ = self.invoke(["sessions", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)[0]["id"], sid)

    def test_chat_exits_cleanly_on_eof(self):
        code, _, _ = self.invoke(["chat"], [EOFError()])
        self.assertEqual(code, 0)

    def test_readline_enables_history_when_available(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        history = Mock()
        with patch("paper2perf.cli.importlib.import_module", return_value=SimpleNamespace(
                set_auto_history=history)):
            self.assertTrue(configure_line_editing())
        history.assert_called_once_with(True)

    def test_missing_readline_does_not_prevent_chat(self):
        with patch("paper2perf.cli.importlib.import_module", side_effect=ImportError):
            self.assertFalse(configure_line_editing())

    def test_chat_assumptions_and_clearing_persist_without_another_run(self):
        sid = self.store.new_session()
        self.invoke(["chat", "--session", sid], [
            "/assume bandwidth 1000 byte/s Explicit test assumption", "/quit"
        ])
        self.assertEqual(self.store.session_assumptions(sid)[0]["value"], 1000)
        self.invoke(["chat", "--session", sid], ["/clear-assumptions", "/quit"])
        self.assertEqual(self.store.session_assumptions(sid), [])
        code, out, _ = self.invoke(["chat", "--session", sid], ["/assumptions", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("explicit assumptions: 0", out)

    def test_replay_creates_new_run_with_explicit_policy_and_shape(self):
        sid = self.store.new_session()
        ids = ingest_documents(self.settings, self.store, sid, [self.source])
        old = self.store.start_run(sid, {"document_ids": ids, "message": "Original request"})
        self.store.set_run(old, "needs_information")
        bp = self.root / "blueprint.json"
        source = Path(__file__).resolve().parents[1] / "examples/a100-whitepaper-blueprint.json"
        bp.write_text(source.read_text())
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, out, _ = self.invoke(["replay", old, "--gemm", "128", "256", "512",
                "--blueprint", str(bp), "--bandwidth-policy", "hbm-proxy", "--quiet", "--json"])
        self.assertEqual(code, 0)
        new = json.loads(out)
        self.assertNotEqual(new["id"], old)
        request = self.requests[0]["request"]
        self.assertEqual(request["evidence_run_id"], old)
        self.assertEqual(request["bandwidth_policy"], "hbm-proxy")
        self.assertTrue(request["deterministic_analysis"])
        self.assertEqual(request["blueprint"]["workload"]["K"], 512)
        self.assertEqual(request["workload_contract"]["N"], 256)
        self.assertEqual(self.store.run(sid, old)["status"], "needs_information")

    def test_chat_policy_requires_explicit_opt_in(self):
        messages = ["/bandwidth-policy hbm-proxy", "Simulate FP16 GEMM", "/quit"]
        with patch.object(InvestigationRunner, "_execute", autospec=True, side_effect=self.fake_execute):
            code, _, _ = self.invoke(["chat", "--documents", str(self.source), "--quiet"], messages)
        self.assertEqual(code, 0)
        self.assertEqual(self.requests[0]["request"]["bandwidth_policy"], "hbm-proxy")

    def test_removed_serve_command_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["serve"])
        self.assertEqual(error.exception.code, 2)


class RunnerTests(unittest.TestCase):
    def test_process_lock_prevents_concurrent_investigations(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with execution_lock(root):
                with self.assertRaisesRegex(RuntimeError, "Another investigation"):
                    with execution_lock(root):
                        self.fail("Lock was not enforced")
            with execution_lock(root):
                pass

    def test_foreign_document_rejected_before_a_run_is_created(self):
        with TemporaryDirectory() as temp:
            settings = Settings(data_dir=Path(temp))
            store = Store(settings.data_dir)
            a, b = store.new_session(), store.new_session()
            did = store.add_document(a, "a.txt", ".txt", b"source", ["source"])
            with self.assertRaisesRegex(ValueError, "does not belong"):
                InvestigationRunner(settings, store).run(b, {"message": "test", "document_ids": [did]})
            self.assertEqual(store.runs(b), [])

    def test_document_size_limit_applies_to_cli_ingestion(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text("Too large")
            settings = Settings(data_dir=root / "data", max_document_bytes=3)
            store = Store(settings.data_dir)
            with self.assertRaisesRegex(ValueError, "limit"):
                ingest_documents(settings, store, store.new_session(), [source])


if __name__ == "__main__":
    unittest.main()
