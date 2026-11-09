"""Synchronous LangGraph execution shared by terminal chat and batch experiments."""
from contextlib import contextmanager
import importlib.util
import sqlite3
import traceback

from .graph import build_graph
from .hwperf import HwPerfBackend
from .llm import ModelClient
from .nodes import Nodes
from .schemas import ChatRequest


@contextmanager
def execution_lock(root):
    """OS-released advisory lock: a killed process cannot leave a stale execution lease."""
    import os
    with (root / "execution.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another investigation is using this data directory.") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another investigation is using this data directory.") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class InvestigationRunner:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store

    def available(self):
        return importlib.util.find_spec("langgraph") is not None

    def run(self, sid, request=None, *, resume_id=None, on_event=None):
        if not self.store.session_exists(sid):
            raise ValueError("Conversation not found")
        with execution_lock(self.store.root):
            if resume_id:
                old = self.store.run(sid, resume_id)
                if old is None:
                    raise ValueError("Investigation not found in this conversation")
                if old["status"] not in {"failed", "interrupted", "running", "queued"}:
                    raise ValueError("Only failed or interrupted investigations can resume")
                rid, data = resume_id, old["request"]
            else:
                data = ChatRequest.model_validate(request).model_dump()
                allowed = {d["id"] for d in self.store.documents(sid)}
                selected = data["document_ids"] or sorted(allowed)
                if not set(selected).issubset(allowed):
                    raise ValueError("A selected document does not belong to this conversation")
                if len(selected) > 20:
                    raise ValueError("Select at most 20 documents for an investigation")
                if not selected and data["mode"] != "inspect":
                    raise ValueError("Add an original source document first")
                data["document_ids"] = selected
                rid = self.store.start_run(sid, data)
                self.store.message(sid, "user", data["message"], rid)
            self._execute(sid, rid, data, bool(resume_id), on_event)
            return self.store.run(sid, rid)

    def replay_evidence(self, sid, rid, request):
        """Recheck original bytes, chunk quotes, and fact values before replay."""
        import hashlib
        import json
        from .schemas import Extraction
        from .evidence import verify_extraction
        source = self.store.run(sid, request["evidence_run_id"])
        if source is None:
            raise ValueError("Evidence investigation does not belong to this conversation")
        path = self.store.root / "runs" / source["id"] / "evidence.json"
        raw = path.read_bytes()
        extraction = Extraction.model_validate(json.loads(raw))
        source_ids = set(source["request"]["document_ids"])
        if not source_ids.issubset(set(request["document_ids"])):
            raise ValueError("Replay must include all original evidence documents")
        docs = {d["id"]: d for d in self.store.documents(sid)}
        if not source_ids or not source_ids.issubset(docs):
            raise ValueError("Original evidence documents are unavailable in this conversation")
        hashes = []
        for did in sorted(source_ids):
            doc = docs[did]
            original = self.store.root / "documents" / (did + doc["suffix"])
            if (not original.is_file()
                    or hashlib.sha256(original.read_bytes()).hexdigest() != doc["sha256"]):
                raise ValueError("Original document is missing or changed; re-ingest it before replay")
            hashes.append({"document_id": did, "sha256": doc["sha256"]})
        verify_extraction(extraction, self.store.chunks(sorted(source_ids)))
        self.store.artifact(rid, "evidence-reuse.json", {
            "source_run_id": source["id"], "sha256": hashlib.sha256(raw).hexdigest(),
            "source_documents": hashes, "revalidated": True})
        return extraction.model_dump()

    def _execute(self, sid, rid, request, resume, on_event):
        self.store.set_run(rid, "running")
        try:
            if not self.available():
                raise RuntimeError("LangGraph is not installed. Install the project's dependencies.")
            from langgraph.checkpoint.sqlite import SqliteSaver
            client = ModelClient(self.settings, self.store, rid, on_event=on_event)
            client.calls = len(list((self.store.root / "runs" / rid).glob("llm-*-request.json")))
            nodes = Nodes(self.settings, self.store, client, HwPerfBackend(self.settings), rid,
                          on_event=on_event)
            last_result = None
            for run in reversed(self.store.runs(sid)):
                if run["id"] == rid:
                    continue
                old = self.store.run(sid, run["id"])
                if old["result"] and old["result"].get("metrics"):
                    last_result = old["result"]
                    break
            state = {"run_id": rid, "session_id": sid, **request,
                     "history": self.store.history(sid), "last_result": last_result}
            if request.get("evidence_run_id"):
                state["seed_extraction"] = self.replay_evidence(sid, rid, request)
            with sqlite3.connect(self.store.root / "checkpoints.sqlite",
                                 check_same_thread=False, timeout=30) as conn:
                graph = build_graph(nodes, SqliteSaver(conn))
                config = {"configurable": {"thread_id": rid}, "recursion_limit": 64}
                if resume and not graph.get_state(config).next:
                    raise RuntimeError("No resumable checkpoint exists; start a new investigation.")
                graph.invoke(None if resume else state, config)
        except KeyboardInterrupt:
            message = "Investigation interrupted. Resume with: paper2perf resume " + rid
            self.store.event(rid, "interrupted", message)
            self.store.set_run(rid, "interrupted", {"response": message, "run_id": rid})
        except Exception as exc:
            self.store.artifact(rid, "error.txt", traceback.format_exc())
            message = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else (
                "Investigation failed; inspect error.txt in the run directory."
            )
            self.store.event(rid, "error", message[:2000])
            self.store.set_run(rid, "failed", {"response": message[:2000], "run_id": rid})
