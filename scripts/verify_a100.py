"""Required, non-skipping A100 backend verification using recorded source evidence.

This is a regression check, not a fresh LLM extraction or GPU measurement.
Run from an installed editable checkout: python scripts/verify_a100.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from paper2perf.compiler import compile_blueprint
from paper2perf.config import Settings
from paper2perf.evidence import verify_extraction, verify_proposal
from paper2perf.hwperf import HwPerfBackend
from paper2perf.schemas import Extraction, Gemm, HardwareBlueprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/a100-verification'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    backend = HwPerfBackend(Settings(data_dir=folder))
    status = backend.status()
    if not status['available']:
        print(json.dumps({'status': 'blocked', 'backend': status}, indent=2), file=sys.stderr)
        return 2
    evidence_path = root / 'tests/fixtures/a100-evidence.json'
    chunks_path = root / 'tests/fixtures/a100-chunks.json'
    extraction = Extraction.model_validate_json(evidence_path.read_text())
    chunks = json.loads(chunks_path.read_text())
    blueprint = HardwareBlueprint.model_validate_json(
        (root / 'examples/a100-whitepaper-blueprint.json').read_text())
    blueprint.workload = Gemm(operator='gemm', M=256, N=256, K=256, dtype='fp16',
                              description='Recorded A100 evidence regression check')
    verify_extraction(extraction, chunks)
    proposal, assumptions, audit, gaps = compile_blueprint(blueprint, extraction, [], 'hbm-proxy')
    if gaps:
        raise ValueError('A100 verification has unresolved requirements: ' + '; '.join(gaps))
    verify_proposal(proposal, extraction, assumptions, chunks)
    for name, value in {
        'proposal.json': proposal.model_dump(), 'derivations.json': audit,
        'assumptions.json': [a.model_dump() for a in assumptions],
    }.items():
        (folder / name).write_text(json.dumps(value, indent=2))
    # Delete only this verification output to require a new backend invocation,
    # rather than accepting an existing cache entry.
    (folder / 'backend-response.json').unlink(missing_ok=True)
    metrics = backend.simulate(proposal, folder)
    required = ['hardware-graph.json', 'mapping-graph.json', 'events.json', 'scopes.json',
                'numerical-validation.json']
    for artifact in required:
        if not (folder / artifact).is_file():
            raise RuntimeError('Real backend omitted required artifact: ' + artifact)
    graph = json.loads((folder / 'hardware-graph.json').read_text())
    if len(graph['nodes']) != 4 or len(graph.get('edges', graph.get('links', []))) != 4:
        raise RuntimeError('Backend graph did not preserve the full recorded A100 hierarchy')
    result = {
        'status': 'completed', 'scenario': 'explicit hbm-proxy regression experiment',
        'source_evidence_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        'metrics': metrics, 'output_directory': str(folder),
    }
    (folder / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
