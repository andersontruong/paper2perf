"""Terminal entry points. Investigations run directly in the foreground process."""
import argparse
from dataclasses import replace
from functools import partial
import importlib.util
import importlib
import json
from pathlib import Path
import shlex
import sys

from .config import Settings, loaded_env_file
from .documents import ingest_documents
from .hwperf import HwPerfBackend
from .runner import InvestigationRunner
from .schemas import Assumption, ChatRequest, Gemm, HardwareBlueprint
from .store import Store


MODES = ("auto", "question", "simulate", "inspect")
SUCCESS = {"completed"}


def parser():
    root = argparse.ArgumentParser(prog="paper2perf", description="Document-grounded performance investigations")
    root.add_argument("--data-dir", type=Path, help="Local state directory (place before the command)")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check model, LangGraph, and bundled hw-perf configuration")
    sessions = commands.add_parser("sessions", help="List saved conversations")
    sessions.add_argument("--json", action="store_true")
    for name, help_text in [("chat", "Start interactive terminal chat"),
                            ("run", "Run one question or experiment in the foreground")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--documents", nargs="+", type=Path, default=[])
        command.add_argument("--session", help="Continue a saved conversation")
        command.add_argument("--mode", choices=MODES, default="auto",
                             help="auto uses regex routing (default: auto)")
        command.add_argument("--assumptions", type=Path, help="JSON array of explicit modeling assumptions")
        command.add_argument("--bandwidth-policy", choices=("strict", "hbm-proxy"), default="hbm-proxy",
                             help="Evidence-only or assumed HBM-rate sensitivity scenario (default: hbm-proxy)")
        analysis = command.add_mutually_exclusive_group()
        analysis.add_argument("--no-llm-analysis", dest="no_llm_analysis", action="store_true",
                              default=True, help="Use deterministic analysis (default)")
        analysis.add_argument("--llm-analysis", dest="no_llm_analysis", action="store_false",
                              help="Let the model interpret simulation metrics")
        command.add_argument("--quiet", action="store_true", help="Suppress stage progress")
        command.add_argument("--verbose", action="store_true",
                             help="Show all progress stages and report sections")
        if name == "run":
            command.add_argument("--question", required=True)
            command.add_argument("--json", action="store_true", help="Write one JSON result to stdout")
    replay = commands.add_parser("replay", help="New experiment from revalidated saved evidence; never resumes a terminal run")
    replay.add_argument("run_id")
    replay.add_argument("--gemm", nargs=3, type=int, required=True, metavar=("M", "N", "K"))
    replay.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    replay.add_argument("--blueprint", type=Path, help="Reference-only JSON blueprint; omits the design model call")
    replay.add_argument("--bandwidth-policy", choices=("strict", "hbm-proxy"), default="hbm-proxy")
    replay.add_argument("--assumptions", type=Path)
    replay.add_argument("--quiet", action="store_true")
    replay.add_argument("--verbose", action="store_true",
                        help="Show all progress stages and report sections")
    replay.add_argument("--json", action="store_true")
    inspect = commands.add_parser("inspect", help="Read saved results without calling a model")
    inspect.add_argument("run_id")
    inspect.add_argument("--artifact", help="Read one saved artifact by its filename")
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--verbose", action="store_true",
                         help="Show full report sections instead of the summary")
    resume = commands.add_parser("resume", help="Resume a failed or interrupted LangGraph run")
    resume.add_argument("run_id")
    resume.add_argument("--json", action="store_true")
    resume.add_argument("--quiet", action="store_true")
    resume.add_argument("--verbose", action="store_true",
                        help="Show all progress stages and report sections")
    return root


HIDDEN_STAGES = {"route", "retrieve", "extract", "extraction_validation",
                 "repair_extraction", "plan_retrieval"}


def progress(stage, detail, verbose=False):
    if not verbose and stage in HIDDEN_STAGES:
        return
    print(f"[{stage}] {detail}", file=sys.stderr, flush=True)


def assumptions_from_file(path):
    if path is None:
        return []
    values = json.loads(path.read_text())
    if not isinstance(values, list):
        raise ValueError("The assumptions file must contain a JSON array")
    return [Assumption.model_validate(a).model_dump() for a in values]


def get_run(store, rid):
    sid = store.session_for_run(rid)
    if sid is None:
        raise ValueError("Investigation not found: " + rid)
    return store.run(sid, rid)


def show_run(store, run, *, as_json=False, verbose=False):
    folder = store.root / "runs" / run["id"]
    output = {**run, "artifact_directory": str(folder.resolve()),
              "artifacts": sorted(p.name for p in folder.iterdir() if p.is_file())
              if folder.exists() else []}
    if as_json:
        print(json.dumps(output, indent=2))
        return
    print(f"Session: {run['session']}\nRun: {run['id']}\nStatus: {run['status']}")
    result = run.get("result") or {}
    response = result.get("response")
    if response and not verbose:
        kept, skipping = [], False
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.startswith(("Scenario:", "Model scope:", "Limitation:")):
                continue
            if stripped == "Explicit modeling assumptions:":
                skipping = True
                continue
            if stripped.startswith("Simulator conventions:"):
                skipping = False
                continue
            if skipping:
                skipping = bool(stripped)
                continue
            kept.append(line)
        response = "\n".join(kept).strip()
    if response:
        print("\n" + response)
    docs = {d["id"]: d for d in store.documents(run["session"])}
    references = result.get("references", [])
    seen = set()
    for ref in references:
        key = (ref.get("chunk_id"), ref.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        if ref.get("artifact"):
            source = store.root / "runs" / ref["run_id"] / ref["artifact"]
            label = str(source)
        else:
            doc = docs.get(ref.get("document_id"))
            if not doc:
                continue
            source = store.root / "documents" / (doc["id"] + doc["suffix"])
            label = f"{ref['name']}, page {ref['page']} ({source})"
        print(f"\nSource: {label}\n  {ref['quote']}")
    print(f"\nArtifacts: {folder.resolve()}")
    if output["artifacts"]:
        print("  " + ", ".join(output["artifacts"]))


def inspect_run(store, rid, artifact=None, as_json=False, verbose=False):
    run = get_run(store, rid)
    if not artifact:
        show_run(store, run, as_json=as_json, verbose=verbose)
        return
    if not artifact or Path(artifact).name != artifact or artifact.startswith("."):
        raise ValueError("Artifact must be a filename inside the run directory")
    path = store.root / "runs" / rid / artifact
    if not path.is_file():
        raise ValueError("Artifact not found: " + artifact)
    text = path.read_text()
    if as_json:
        try:
            value = json.loads(text)
        except ValueError:
            value = {"artifact": artifact, "content": text}
        print(json.dumps(value, indent=2))
    else:
        print(text)


def session(store, sid):
    if sid and not store.session_exists(sid):
        raise ValueError("Conversation not found: " + sid)
    return sid or store.new_session()


CHAT_HELP = """Enter a question, or use:
  /add PATH [PATH ...]       Add original documents (quote paths containing spaces)
  /documents                List sources in this conversation
  /mode auto|question|simulate|inspect
                            auto uses regex matching; explicit modes override it
  /assume ID VALUE UNIT RATIONALE
                            Set an explicit assumption; units: byte, byte/s, FLOP/s
  /bandwidth-policy strict|hbm-proxy
                            Select evidence-only or explicit proxy experiment (default: hbm-proxy)
  /assumptions              Show current assumptions
  /clear-assumptions        Remove assumptions from subsequent requests
  /runs                     List this conversation's investigations
  /inspect RUN_ID           Print a saved result without a model call
  /resume RUN_ID            Resume an interrupted investigation
  /help                     Show this help
  /quit                     Exit (conversation remains saved)"""


def configure_line_editing():
    """Enable input() history/editing when Python provides readline; no history file."""
    try:
        readline = importlib.import_module("readline")
    except ImportError:
        return False
    if hasattr(readline, "set_auto_history"):
        readline.set_auto_history(True)
    return True


def chat(args, settings, store, runner):
    configure_line_editing()
    sid = session(store, args.session)
    ingest_documents(settings, store, sid, args.documents)
    assumptions = assumptions_from_file(args.assumptions)
    if args.assumptions is None and args.session:
        saved = store.session_assumptions(sid)
        if saved is not None:
            assumptions = saved
        else:
            prior = store.runs(sid)
            if prior:
                assumptions = store.run(sid, prior[-1]["id"])["request"].get("assumptions", [])
    store.set_session_assumptions(sid, assumptions)
    mode = args.mode
    bandwidth_policy = args.bandwidth_policy
    print(f"Paper2Perf terminal chat\nSession: {sid}\nMode: {mode}")
    print(f"Sources: {len(store.documents(sid))}; explicit assumptions: {len(assumptions)}")
    print("Bandwidth policy: " + bandwidth_policy)
    print("Type /help for commands. Ctrl-D or /quit exits.")
    callback = None if args.quiet else partial(progress, verbose=args.verbose)
    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        try:
            if line.startswith("/"):
                parts = shlex.split(line)
                command, values = parts[0], parts[1:]
                if command in {"/quit", "/exit"}:
                    return 0
                if command == "/help":
                    print(CHAT_HELP)
                elif command == "/add" and values:
                    ingest_documents(settings, store, sid, [Path(p) for p in values])
                    print(f"Sources: {len(store.documents(sid))}")
                elif command == "/documents":
                    for doc in store.documents(sid):
                        print(f"{doc['id']}  {doc['name']}  ({doc['pages']} pages)")
                elif command == "/mode" and len(values) == 1 and values[0] in MODES:
                    mode = values[0]
                    print("Mode: " + mode)
                elif command == "/bandwidth-policy" and len(values) == 1 and values[0] in {"strict", "hbm-proxy"}:
                    bandwidth_policy = values[0]
                    print("Bandwidth policy: " + bandwidth_policy)
                elif command == "/assume" and len(values) >= 4:
                    assumption = Assumption(id=values[0], value=float(values[1]), unit=values[2],
                                            rationale=" ".join(values[3:])).model_dump()
                    assumptions = [a for a in assumptions if a["id"] != assumption["id"]]
                    assumptions.append(assumption)
                    store.set_session_assumptions(sid, assumptions)
                    print("Explicit assumption recorded: " + assumption["id"])
                elif command == "/assumptions":
                    print(json.dumps(assumptions, indent=2))
                elif command == "/clear-assumptions":
                    assumptions = []
                    store.set_session_assumptions(sid, assumptions)
                    print("Assumptions cleared for subsequent requests.")
                elif command == "/runs":
                    for run in store.runs(sid):
                        print(f"{run['id']}  {run['status']}  {run['created']}")
                elif command == "/inspect" and len(values) == 1:
                    if store.run(sid, values[0]) is None:
                        raise ValueError("Investigation not found in this conversation")
                    inspect_run(store, values[0], verbose=args.verbose)
                elif command == "/resume" and len(values) == 1:
                    show_run(store, runner.run(sid, resume_id=values[0], on_event=callback),
                             verbose=args.verbose)
                else:
                    print("Unknown command or invalid arguments. Type /help.")
                continue
            request = ChatRequest(message=line, mode=mode, assumptions=assumptions,
                                  bandwidth_policy=bandwidth_policy,
                                  deterministic_analysis=args.no_llm_analysis).model_dump()
            show_run(store, runner.run(sid, request, on_event=callback), verbose=args.verbose)
        except (ValueError, RuntimeError, OSError) as exc:
            print("Error: " + str(exc), file=sys.stderr)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.data_dir:
            settings = replace(settings, data_dir=args.data_dir.resolve())
        if args.command == "doctor":
            print(json.dumps({
                "langgraph_installed": importlib.util.find_spec("langgraph") is not None,
                "torch_installed": importlib.util.find_spec("torch") is not None,
                "model_configured": bool(settings.model and settings.api_key),
                "env_file": loaded_env_file(),
                "backend": HwPerfBackend(settings).status(),
            }, indent=2))
            return 0
        store = Store(settings.data_dir)
        runner = InvestigationRunner(settings, store)
        if args.command == "sessions":
            items = store.sessions()
            if args.json:
                print(json.dumps(items, indent=2))
            else:
                for item in items:
                    print(f"{item['id']}  {item['created']}  {item['document_count']} sources; {item['run_count']} runs")
            return 0
        if args.command == "inspect":
            inspect_run(store, args.run_id, args.artifact, args.json, args.verbose)
            return 0
        if args.command == "chat":
            return chat(args, settings, store, runner)
        callback = None if args.quiet else partial(progress, verbose=args.verbose)
        if args.command == "replay":
            old = get_run(store, args.run_id)
            sid = old["session"]
            M, N, K = args.gemm
            workload = Gemm(operator="gemm", M=M, N=N, K=K, dtype=args.dtype,
                            description="Explicit replay GEMM", basis="user_request")
            blueprint = None
            path = args.blueprint or (store.root / "runs" / args.run_id / "blueprint.json")
            if args.blueprint or path.is_file():
                blueprint = HardwareBlueprint.model_validate(json.loads(path.read_text()))
                # Dimensions in the explicit replay command replace the old experiment's workload.
                blueprint.workload = workload
            request = ChatRequest(
                message=f"Simulate dense {args.dtype} C=A@B, no bias, M={M}, N={N}, K={K}.",
                mode="simulate", document_ids=old["request"]["document_ids"],
                evidence_run_id=args.run_id, blueprint=blueprint.model_dump() if blueprint else None,
                workload_contract=workload, deterministic_analysis=True,
                bandwidth_policy=args.bandwidth_policy,
                assumptions=assumptions_from_file(args.assumptions)).model_dump()
            run = runner.run(sid, request, on_event=callback)
        elif args.command == "resume":
            old = get_run(store, args.run_id)
            run = runner.run(old["session"], resume_id=args.run_id, on_event=callback)
        else:
            sid = session(store, args.session)
            documents = ingest_documents(settings, store, sid, args.documents)
            request = ChatRequest(message=args.question, mode=args.mode, document_ids=documents,
                                  assumptions=assumptions_from_file(args.assumptions),
                                  bandwidth_policy=args.bandwidth_policy,
                                  deterministic_analysis=args.no_llm_analysis).model_dump()
            run = runner.run(sid, request, on_event=callback)
        show_run(store, run, as_json=args.json, verbose=args.verbose)
        return 0 if run["status"] in SUCCESS else 130 if run["status"] == "interrupted" else 1
    except (ValueError, RuntimeError, OSError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
