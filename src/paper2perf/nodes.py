"""Graph stages. All LLM outputs cross typed and evidence checks before execution."""
import math
import re

from .evidence import (
    EvidenceError, numbers, render_answer, verify_answer, verify_citations,
    verify_extraction, verify_proposal,
)
from .hwperf import BackendUnavailable
from .llm import ModelError, StructuredOutputError
from .schemas import (
    Analysis, Answer, Assumption, Extraction, HardwareBlueprint, Intent, Proposal, RetrievalPlan,
)
from .workload import materialize


class Nodes:
    def __init__(self, settings, store, client, backend, run_id, on_event=None):
        self.settings, self.store, self.client = settings, store, client
        self.backend, self.run_id = backend, run_id
        self.on_event = on_event

    def event(self, stage, detail):
        self.store.event(self.run_id, stage, detail)
        if self.on_event:
            self.on_event(stage, detail)

    def route(self, state):
        self.event("route", "Routing request by regex" if state["mode"] == "auto"
                   else "Interpreting the question")
        if state["mode"] != "auto":
            intent = Intent(action=state["mode"], query=state["message"])
        else:
            text = state["message"].lower()
            if re.search(
                    r"\b(?:inspect|artifacts?)\b"
                    r"|\b(?:latest|previous|last)\s+(?:run|simulation|experiment|result)\b"
                    r"|\b(?:which|what)\s+assumptions?\b"
                    r"|\bassumptions?\s+(?:were\s+)?used\b"
                    r"|\b(?:why|explain)\b.*\b(?:run|simulation|experiment|result|makespan|metric)\b"
                    r"|\b(?:what|which)\s+(?:was|is)\s+the\s+(?:makespan|result|metrics?)\b"
                    r"|\brun[ _-]?id\b", text):
                action = "inspect"
            elif re.search(
                    r"\b(?:simulate|simulation|estimate|predict|calculate|re-?run|try"
                    r"|what\s+if|how\s+long|run|makespan)\b"
                    r"|\b[CMNK]\s*=\s*\d+"
                    r"|\b\d+\s*[x\u00d7]\s*\d+\b",
                    text):
                action = "simulate"
            else:
                action = "question"
            intent = Intent(action=action, query=state["message"])
        return {"action": intent.action, "query": intent.query,
                "queries": [intent.query], "status": "running", "retrieval_round": 0,
                "repairs": 0, "chunks": [], "validation_errors": [], "decision": "",
                "answer_attempts": 0, "answer_repairs": 0, "answer_errors": [],
                "extraction_attempts": 0, "extraction_repairs": 0,
                "extraction_errors": [], "rejected_extraction": None}

    def retrieve(self, state):
        self.event("retrieve", "Reading relevant passages from selected documents")
        selected = {c["id"]: c for c in state.get("chunks", [])}
        if state.get("seed_extraction") and not state.get("retrieval_round"):
            return {"chunks": self.store.chunks(state["document_ids"]), "retrieval_round": 1}
        queries = state.get("queries", [])
        if state["action"] == "simulate" and not state.get("retrieval_round"):
            queries = queries + [
                state["query"] + " memory capacity bandwidth cache tensor vector dense precision",
                "SM count L1 shared memory combined capacity L2 read bandwidth GPU Boost Clock MHz"
            ]
        if state["action"] == "simulate" and not state.get("retrieval_round"):
            queries = queries + ["GPU device memory HBM capacity",
                                 "L2 HBM memory spaces accessible SMs",
                                 "global memory into shared memory asynchronous copy data path"]
        fresh = {}
        for query in queries:
            for chunk in self.store.search(state["document_ids"], query, limit=8):
                fresh[chunk["id"]] = chunk
        if not fresh and not selected:
            for did in state["document_ids"]:
                for chunk in self.store.chunks([did])[:2]:
                    fresh[chunk["id"]] = chunk
        from .retrieval import follow_contents
        followed = follow_contents(list(fresh.values()), self.store.chunks(state["document_ids"]))
        # Keep evidence referenced by the previous verified extraction; add new
        # body passages before TOC/heading matches and other old context.
        source_ids = {cite["chunk_id"] for entry in
                      [*state.get("extraction", {}).get("facts", []),
                       *state.get("extraction", {}).get("relationships", [])]
                      for cite in entry.get("citations", [])} if state.get("extraction") else set()
        protected = [c for c in selected.values() if c["id"] in source_ids]
        ordered = {c["id"]: c for c in [*protected, *followed, *fresh.values(), *selected.values()]}
        result = list(ordered.values())[:64]
        self.store.artifact(self.run_id, f"retrieval-{state.get('retrieval_round', 0) + 1}.json", {
            "queries": queries, "followed_source_pages": sorted({c["page"] for c in followed}),
            "selected_chunk_ids": [c["id"] for c in result]})
        return {"chunks": result,
                "retrieval_round": state.get("retrieval_round", 0) + 1}

    def answer(self, state):
        self.event("answer", "Checking cited document answers")
        return self._checked_answer(state, state["chunks"], "answer")

    def _checked_answer(self, state, chunks, prefix):
        attempt = state.get("answer_attempts", 0) + 1
        repairs = state.get("answer_repairs", 0)
        answer = None
        try:
            answer = self.client.ask("answer", {
                "question": state["query"], "chunks": chunks,
                "previous_answer": state.get("answer") if state.get("answer_errors") else None,
                "validation_errors": state.get("answer_errors", []),
            }, Answer)
            self.store.artifact(self.run_id, f"{prefix}-attempt-{attempt}.json", answer.model_dump())
            verify_answer(answer, chunks)
        except (StructuredOutputError, EvidenceError) as exc:
            diagnostics = {"passed": False, "attempt": attempt, "error": str(exc)}
            self.store.artifact(self.run_id, f"{prefix}-validation-{attempt}.json", diagnostics)
            self.store.artifact(self.run_id, f"{prefix}-validation.json", diagnostics)
            self.event("answer_validation", str(exc))
            update = {"answer_attempts": attempt, "answer_errors": [str(exc)],
                      "answer": answer.model_dump() if answer else None, "references": []}
            if repairs < self.settings.max_repairs:
                self.event("repair_answer", "Revising the answer using citation and numeric diagnostics")
                return {**update, "answer_repairs": repairs + 1, "decision": "retry"}
            return {**update, "status": "needs_information", "decision": "unverified",
                    "response": "A document-grounded answer could not be verified after bounded "
                    "retries. No unverified answer is accepted. Inspect " + prefix + "-validation.json "
                    "for the rejected claim and its source numbers."}
        self.store.artifact(self.run_id, f"{prefix}-validation-{attempt}.json", {
            "passed": True, "attempt": attempt
        })
        self.store.artifact(self.run_id, f"{prefix}-validation.json", {
            "passed": True, "attempt": attempt
        })
        text, refs = render_answer(answer, chunks)
        return {"answer": answer.model_dump(), "response": text, "references": refs,
                "answer_attempts": attempt, "answer_errors": [],
                "decision": "done" if answer.claims else "missing"}

    def extract(self, state):
        self.event("extract", "Extracting source facts, scopes, and conflicts")
        attempt = state.get("extraction_attempts", 0) + 1
        repairs = state.get("extraction_repairs", 0)
        extraction = None
        try:
            if state.get("seed_extraction") and attempt == 1:
                extraction = Extraction.model_validate(state["seed_extraction"])
                self.event("evidence_reuse", "Revalidating saved extraction against original document chunks")
            else:
                extraction = self.client.ask("extract", {
                    "question": state["query"], "chunks": state["chunks"],
                    "allowed_chunk_ids": [c["id"] for c in state["chunks"]],
                    "previous_extraction": state.get("rejected_extraction"),
                    "validation_errors": state.get("extraction_errors", []),
                    "modeling_requirements": state.get("validation_errors", []),
                    "previous_verified_extraction": state.get("extraction"),
                }, Extraction)
            self.store.artifact(self.run_id, f"extraction-attempt-{attempt}.json", extraction.model_dump())
            verify_extraction(extraction, state["chunks"])
        except (StructuredOutputError, EvidenceError) as exc:
            diagnostics = {"passed": False, "attempt": attempt, "error": str(exc)}
            self.store.artifact(self.run_id, f"extraction-validation-{attempt}.json", diagnostics)
            self.store.artifact(self.run_id, "extraction-validation.json", diagnostics)
            self.event("extraction_validation", str(exc))
            update = {"extraction": None, "proposal": None, "extraction_attempts": attempt,
                      "rejected_extraction": extraction.model_dump() if extraction else None,
                      "extraction_errors": [str(exc)], "validation_errors": [str(exc)]}
            if repairs < self.settings.max_repairs:
                self.event("repair_extraction", "Correcting rejected source citations or fact values")
                return {**update, "extraction_repairs": repairs + 1, "decision": "retry"}
            return {**update, "status": "needs_information", "decision": "unverified",
                    "response": "Source extraction could not be verified after bounded retries. "
                    "No simulation was run. Inspect extraction-validation.json."}
        diagnostics = {"passed": True, "attempt": attempt}
        self.store.artifact(self.run_id, f"extraction-validation-{attempt}.json", diagnostics)
        self.store.artifact(self.run_id, "extraction-validation.json", diagnostics)
        self.store.artifact(self.run_id, "evidence.json", extraction.model_dump())
        return {"extraction": extraction.model_dump(), "extraction_attempts": attempt,
                "extraction_errors": [], "rejected_extraction": None,
                "validation_errors": [], "decision": "verified"}

    def propose(self, state):
        if state.get("construction") == "grounded":
            return self.construct(state)
        self.event("propose", "Constructing evidence-linked hardware and workload models")
        payload = {
            "question": state["query"], "user_message": state["message"],
            "recent_conversation": state["history"][-8:],
            "extraction": state["extraction"], "chunks": state["chunks"],
            "explicit_assumptions": state["assumptions"],
            "previous_proposal": state.get("proposal"),
            "diagnostics": state.get("validation_errors", []),
        }
        try:
            proposal = self.client.ask("propose", payload, Proposal)
        except StructuredOutputError as exc:
            return {"proposal": None, "validation_errors": [str(exc)]}
        self.store.artifact(self.run_id, "proposal.json", proposal.model_dump())
        self.store.artifact(self.run_id, f"proposal-{state.get('repairs', 0)}.json",
                            proposal.model_dump())
        return {"proposal": proposal.model_dump()}

    def construct(self, state):
        from .compiler import compile_blueprint
        self.event("construct", "Compiling fact references into hardware quantities and topology")
        blueprint = None
        try:
            if state.get("blueprint") is not None:
                blueprint = HardwareBlueprint.model_validate(state["blueprint"])
            else:
                blueprint = self.client.ask("design", {
                    "question": state["message"], "extraction": state["extraction"],
                    "explicit_assumptions": state["assumptions"],
                    "bandwidth_policy": state.get("bandwidth_policy", "strict"),
                    "workload_contract": state.get("workload_contract"),
                    "diagnostics": state.get("validation_errors", []),
                    "previous_blueprint": state.get("previous_blueprint"),
                    "adapter_contract": {
                        "required": ["memory capacities in bytes", "connected source-backed topology",
                                     "link rates in byte/s", "dense precision-matched peak in FLOP/s",
                                     "user-authorized bias-free GEMM dimensions"],
                        "not_modeled": ["register-file timing", "fixed memory access latency",
                                        "SM tile/wave schedules from the document", "occupancy",
                                        "MMA instruction timing", "accumulator-type timing differences"],
                        "bandwidth_resolution": (
                            "hbm-proxy was explicitly selected; unresolved link-rate references are "
                            "compiled into recorded assumptions using the device memory bandwidth fact."
                            if state.get("bandwidth_policy") == "hbm-proxy" else
                            "Strict mode: unresolved bandwidth references become required compiler gaps."),
                        "capacity_resolution": "Combined per-SM local capacity times product SM count is "
                            "supported; partition split and maximum configurable SMEM are limitations "
                            "when the selected node models the combined pool.",
                    },
                }, HardwareBlueprint)
            self.store.artifact(self.run_id, "blueprint.json", blueprint.model_dump())
            if state.get("workload_contract") is not None:
                from .schemas import Gemm
                expected = Gemm.model_validate(state["workload_contract"])
                fields = ("operator", "M", "N", "K", "dtype", "bias")
                if blueprint.workload is None or any(
                        getattr(blueprint.workload, f) != getattr(expected, f) for f in fields):
                    raise ValueError("Blueprint workload differs from the explicit workload contract")
                blueprint.workload = expected
            extraction = Extraction.model_validate(state["extraction"])
            topology_audit = {}
            proposal, assumptions, audit, gaps = compile_blueprint(
                blueprint, extraction, state["assumptions"],
                state.get("bandwidth_policy", "strict"), topology_audit=topology_audit)
        except (ValueError, StructuredOutputError) as exc:
            self.store.artifact(self.run_id, "design-validation.json", {
                "passed": False, "error": str(exc),
                "legacy_concerns": blueprint.missing if blueprint else [],
            })
            return {"proposal": None, "validation_errors": [str(exc)],
                    "previous_blueprint": blueprint.model_dump() if blueprint else None}
        self.store.artifact(self.run_id, "design-validation.json", {
            "passed": True, "legacy_concerns": [],
            "blockers": [b.model_dump() for b in blueprint.blockers],
        })
        self.store.artifact(self.run_id, "derivations.json", audit)
        self.store.artifact(self.run_id, "topology-audit.json", topology_audit)
        self.store.artifact(self.run_id, "model-gaps.json", {"required": gaps,
                            "design_blockers": [b.model_dump() for b in blueprint.blockers],
                            "limitations": proposal.limitations,
                            "policy": state.get("bandwidth_policy", "strict")})
        self.store.artifact(self.run_id, "assumptions.json", [a.model_dump() for a in assumptions])
        self.store.artifact(self.run_id, "proposal.json", proposal.model_dump())
        return {"proposal": proposal.model_dump(), "assumptions": [a.model_dump() for a in assumptions],
                "previous_blueprint": blueprint.model_dump()}

    def validate(self, state):
        self.event("validate", "Validating topology, units, evidence, and workload scope")
        if state.get("proposal") is None:
            return {"decision": "invalid", "validation_errors": state["validation_errors"]}
        proposal = Proposal.model_validate(state["proposal"])
        extraction = Extraction.model_validate(state["extraction"])
        if proposal.missing or proposal.hardware is None or proposal.workload is None:
            return {"decision": "missing", "validation_errors": proposal.missing or [
                "Hardware or workload information is incomplete."
            ]}
        try:
            verify_proposal(proposal, extraction, [
                Assumption.model_validate(a) for a in state["assumptions"]
            ], state["chunks"])
            workload = proposal.workload
            if workload.basis == "document":
                if not workload.citations:
                    raise EvidenceError("Document-defined workload needs source citations")
                verify_citations(workload.citations, state["chunks"])
                source = " ".join(c.quote for c in workload.citations)
            else:
                # Assistant/model text cannot authorize a workload shape.
                source = state["message"] + " " + " ".join(
                    m["content"] for m in state["history"] if m["role"] == "user"
                )
            for dim in [workload.M, workload.N, workload.K]:
                if dim not in numbers(source):
                    raise EvidenceError("GEMM dimensions lack document or user-request support")
            precision = source.lower().replace("float16", "fp16").replace("float32", "fp32")
            if workload.dtype not in precision:
                raise EvidenceError("Workload precision needs explicit document or user support")
        except EvidenceError as exc:
            errors = [str(exc)]
            self.store.artifact(self.run_id, "validation.json", {"passed": False, "errors": errors})
            return {"decision": "invalid", "validation_errors": errors}
        self.store.artifact(self.run_id, "validation.json", {"passed": True, "errors": []})
        self.store.artifact(self.run_id, "hardware.json", proposal.hardware.model_dump())
        self.store.artifact(self.run_id, "workload.json", workload.model_dump())
        self.store.artifact(self.run_id, "workload.py", materialize(workload))
        return {"decision": "ready", "validation_errors": []}

    def plan_retrieval(self, state):
        if state.get("evidence_run_id") or state.get("blueprint") is not None:
            return {"decision": "stop"}
        if state["retrieval_round"] >= self.settings.max_retrievals:
            return {"decision": "stop"}
        self.event("plan_retrieval", "Deciding whether another source search can resolve gaps")
        if state.get("construction") == "grounded":
            from .retrieval import contract_queries
            queries = contract_queries(state.get("validation_errors", []))
            if queries:
                self.store.artifact(self.run_id, f"retrieval-plan-{state['retrieval_round']}.json", {
                    "basis": "deterministic adapter requirements", "queries": queries,
                    "gaps": state.get("validation_errors", [])})
                return {"queries": queries, "decision": "search"}
        plan = self.client.ask("retrieve", {
            "question": state["query"], "gaps": state.get("validation_errors", []),
            "extraction": state.get("extraction"), "chunks": state["chunks"]
        }, RetrievalPlan)
        return {"queries": plan.queries, "decision": "search" if plan.queries else "stop"}

    def repair(self, state):
        self.event("repair", "Repairing a demonstrated validation error")
        return {"repairs": state.get("repairs", 0) + 1}

    def simulate(self, state):
        self.event("simulate", "Submitting validated models to hw-perf")
        try:
            metrics = self.backend.simulate(
                Proposal.model_validate(state["proposal"]),
                self.store.root / "runs" / self.run_id
            )
        except BackendUnavailable as exc:
            return {"status": "blocked", "decision": "stop", "response": str(exc)}
        metrics["scenario"] = {
            "bandwidth_policy": state.get("bandwidth_policy", "strict"),
            "assumptions": state.get("assumptions", []),
            "evidence_run_id": state.get("evidence_run_id"),
        }
        self.store.artifact(self.run_id, "metrics.json", metrics)
        return {"metrics": metrics, "decision": "done"}

    def analyze(self, state):
        if state.get("deterministic_analysis"):
            analysis = {"findings": [{"kind": "observed", "metric": "makespan_seconds",
                        "value": state["metrics"]["makespan_seconds"],
                        "text": f"hw-perf predicted {state['metrics']['makespan_seconds']:.9g} seconds."}]}
            self.store.artifact(self.run_id, "analysis.json", analysis)
            return {"analysis": analysis}
        self.event("analyze", "Grounding interpretation in simulation artifacts")
        try:
            analysis = self.client.ask("analyze", {
                "metrics": state["metrics"], "proposal": state["proposal"]
            }, Analysis)
            for finding in analysis.findings:
                if finding.kind == "observed":
                    expected = state["metrics"].get(finding.metric)
                    if (isinstance(expected, bool) or not isinstance(expected, (int, float))
                            or finding.value is None
                            or not math.isclose(expected, finding.value, rel_tol=1e-6)):
                        raise EvidenceError("Observed analysis finding does not match a numeric metric")
                    if any(not math.isclose(n, finding.value, rel_tol=1e-6)
                           for n in numbers(finding.text)):
                        raise EvidenceError("Finding text introduces an unsupported numeric claim")
        except (ModelError, EvidenceError) as exc:
            self.store.artifact(self.run_id, "analysis-validation.json", {
                "passed": False, "error": str(exc)
            })
            return {"status": "analysis_failed", "analysis": {"findings": [{
                "kind": "limitation", "text": "Simulation completed, but its LLM interpretation "
                "failed validation or could not be produced. Inspect the saved simulation artifacts."
            }]}}
        self.store.artifact(self.run_id, "analysis-validation.json", {"passed": True})
        self.store.artifact(self.run_id, "analysis.json", analysis.model_dump())
        return {"analysis": analysis.model_dump()}

    def inspect(self, state):
        self.event("inspect", "Inspecting the latest completed experiment")
        result = state.get("last_result")
        if not result or not result.get("metrics"):
            return {"response": "There is no completed simulation in this conversation to inspect.",
                    "decision": "done"}
        import json
        chunks = []
        for name, content in {"metrics.json": result["metrics"],
                              "proposal.json": result.get("proposal"),
                              "evidence.json": result.get("evidence")}.items():
            text = json.dumps(content, indent=2)
            for start in range(0, len(text), 2000):
                chunks.append({"id": f"{name}-{start}", "name": name, "page": 1,
                               "document": "", "text": text[start:start+2400],
                               "artifact": name, "run_id": result["run_id"]})
        result = self._checked_answer(state, chunks, "inspect-answer")
        if result["decision"] == "missing":
            result["decision"] = "done"
        return result

    def report(self, state):
        self.event("report", "Saving evidence, decisions, and reproducible artifacts")
        status = state.get("status", "running")
        text = state.get("response", "")
        if state.get("metrics") and state["action"] != "inspect":
            metrics = state["metrics"]
            text = f"Simulated makespan: {metrics['makespan_seconds']:.9g} seconds.\n\n"
            for f in state.get("analysis", {}).get("findings", []):
                text += f"{f['kind'].capitalize()}: {f['text']}\n\n"
            text += "This is a model prediction, not a measurement on physical hardware."
            if state.get("bandwidth_policy") == "hbm-proxy":
                text += "\n\nScenario: explicit HBM-proxy sensitivity experiment. Missing internal-link " \
                        "rates use the documented external-memory rate as an assumption, not an A100 " \
                        "specification or a validated performance bound."
            if state.get("proposal"):
                text += "\n\nModel scope: " + state["proposal"]["rationale"]
                for limitation in state["proposal"].get("limitations", []):
                    text += "\nLimitation: " + limitation
            assumed = []
            if state.get("proposal", {}).get("hardware"):
                from .evidence import quantities
                hardware = Proposal.model_validate(state["proposal"]).hardware
                assumed = [f"{name}: {q.value:g} {q.unit} — {q.rationale}"
                           for name, q, _ in quantities(hardware) if q.basis == "assumed"]
            if assumed:
                text += "\n\nExplicit modeling assumptions:\n" + "\n".join(assumed)
            text += "\n\nSimulator conventions: " + "; ".join(
                metrics.get("modeling_conventions", [])
            )
        elif state["action"] == "simulate" and status != "blocked":
            status = "needs_information"
            gaps = state.get("validation_errors", []) or ["Required modeling evidence is missing."]
            text = "No simulation was run.\n\n" + "\n".join(gaps)
        if status == "running":
            status = "completed"
        result = {"status": status, "response": text, "references": state.get("references", []),
                  "metrics": state.get("metrics"), "proposal": state.get("proposal"),
                  "evidence": state.get("extraction"), "run_id": self.run_id}
        self.store.artifact(self.run_id, "report.md", text)
        self.store.artifact(self.run_id, "manifest.json", {
            "run_id": self.run_id, "status": status, "model": self.settings.model,
            "documents": [d for d in self.store.documents(state["session_id"])
                          if d["id"] in state["document_ids"]],
            "budgets": {"model_calls": self.client.calls,
                        "retrieval_rounds": state.get("retrieval_round", 0),
                        "repairs": state.get("repairs", 0),
                        "answer_repairs": state.get("answer_repairs", 0),
                        "extraction_repairs": state.get("extraction_repairs", 0)},
            "backend": "hw-perf", "simulation_executed": bool(state.get("metrics")),
            "construction": state.get("construction", "legacy"),
            "bandwidth_policy": state.get("bandwidth_policy", "strict"),
            "evidence_run_id": state.get("evidence_run_id"),
        })
        self.store.set_run(self.run_id, status, result)
        # Idempotent under checkpoint replay.
        if not any(m.get("run_id") == self.run_id and m["role"] == "assistant"
                   for m in self.store.history(state["session_id"], limit=1000)):
            self.store.message(state["session_id"], "assistant", text, self.run_id)
        return {"status": status, "response": text}
