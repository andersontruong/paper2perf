"""Execute bundled hw-perf source in a separate process. No alternate predictor."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

from .torch_compat import REQUIRED_TORCH_VERSION, torch_version_status


class BackendUnavailable(RuntimeError):
    pass


def source_metadata():
    manifest = json.loads((Path(__file__).parent / "vendor_manifest.json").read_text())
    return {"archive_sha256": manifest["archive_sha256"],
            "upstream_commit": manifest["upstream_commit"], "source": "bundled hw-perf snapshot"}


def verify_bundled_source():
    """Check runtime Python files against the original archive hashes."""
    manifest = json.loads((Path(__file__).parent / "vendor_manifest.json").read_text())
    verified = 0
    for entry in manifest["files"]:
        if "/src/" not in entry["path"] or not entry["path"].endswith(".py"):
            continue
        relative = entry["path"].split("/src/", 1)[1]
        package, _, rest = relative.partition("/")
        spec = importlib.util.find_spec(package)
        if not spec or not spec.submodule_search_locations:
            raise BackendUnavailable(f"Bundled package {package} is not installed")
        path = Path(next(iter(spec.submodule_search_locations))) / rest
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise BackendUnavailable(f"Bundled simulator source differs from the snapshot: {relative}")
        verified += 1
    return verified


def validate_metrics(data):
    value = data.get("makespan_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Backend must return a numeric makespan_seconds")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Backend returned invalid makespan")
    if data.get("backend") != "hw-perf":
        raise ValueError("Unexpected performance backend")
    return data


class HwPerfBackend:
    def __init__(self, settings):
        self.settings = settings

    def status(self):
        version = torch_version_status()
        missing = [m for m in ["torch", "networkx", "pysmt", "z3", "pydot"]
                   if importlib.util.find_spec(m) is None]
        try:
            verified = verify_bundled_source()
        except BackendUnavailable as exc:
            return {"available": False, "reason": str(exc), "missing_packages": missing,
                    "torch_version": version, **source_metadata()}
        reason = "Install dependencies" if missing else (
            "Install the pinned torch==" + REQUIRED_TORCH_VERSION if not version["matches"] else ""
        )
        return {"available": not missing and version["matches"], "verified_source_files": verified,
                "missing_packages": missing, "reason": reason, "torch_version": version,
                **source_metadata()}

    def simulate(self, proposal, run_dir):
        status = self.status()
        if not status["available"]:
            raise BackendUnavailable("Bundled hw-perf is unavailable: " + status["reason"] +
                                     "; missing packages: " + ", ".join(status["missing_packages"]))
        request = run_dir / "backend-request.json"
        output = run_dir / "backend-response.json"
        if output.exists():
            data = json.loads(output.read_text())
            if data.get("request_sha256") == hashlib.sha256(
                    proposal.model_dump_json().encode()).hexdigest():
                return validate_metrics(data)
        request.write_text(proposal.model_dump_json(indent=2))
        try:
            result = subprocess.run([
                sys.executable, "-m", "paper2perf.hwperf_worker", str(request), str(output)
            ], capture_output=True, text=True, timeout=self.settings.simulation_timeout)
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailable("hw-perf simulation exceeded its time budget.") from exc
        (run_dir / "backend-stdout.txt").write_text(result.stdout[-100000:])
        (run_dir / "backend-stderr.txt").write_text(result.stderr[-100000:])
        if result.returncode != 0 or not output.exists():
            raise BackendUnavailable("hw-perf execution failed; inspect backend-stderr.txt.")
        return validate_metrics(json.loads(output.read_text()))
