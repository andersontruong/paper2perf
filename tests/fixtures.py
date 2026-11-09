"""Synthetic test-only hardware, never used as an agent input or fallback."""
from paper2perf.schemas import Extraction, Proposal

TEXT = ("TestChip X is a single device with 8 GB memory, 1 TB/s bandwidth, "
        "and 100 TFLOP/s dense FP16 matrix throughput. Memory connects to matrix compute.")
CHUNKS = [{"id": "source-p1-c0", "document": "source", "page": 1,
           "name": "Synthetic test document", "text": TEXT}]
CITE = {"chunk_id": "source-p1-c0", "quote": TEXT}


def extraction():
    return Extraction.model_validate({"facts": [
        {"id": key, "parameter": key, "value": value, "unit": unit,
         "scope": "TestChip X dense FP16 single-device", "variant": "TestChip X",
         "precision": "fp16", "mode": "dense", "aggregation": "device",
         "citations": [CITE]}
        for key, value, unit in [("capacity", 8, "GB"), ("bandwidth", 1, "TB/s"),
                                  ("throughput", 100, "TFLOP/s")]
    ], "relationships": [{"id": "memory_matrix", "source": "memory", "target": "matrix",
                            "kind": "connects", "variant": "TestChip X", "citations": [CITE]}]})


def proposal():
    def quantity(key, value, unit, scale):
        return {"value": value, "unit": unit, "basis": "derived", "fact_ids": [key],
                "scale": scale, "rationale": "Exact decimal SI unit conversion"}
    return Proposal.model_validate({
        "hardware": {"name": "TestChip", "variant": "TestChip X", "citations": [CITE],
                     "memories": [{"id": "memory", "capacity": quantity("capacity", 8e9, "byte", 1e9)}],
                     "computes": [{"id": "matrix", "kind": "matrix", "throughput":
                                   quantity("throughput", 1e14, "FLOP/s", 1e12)}],
                     "links": [{"source": "memory", "target": "matrix", "bandwidth":
                                quantity("bandwidth", 1e12, "byte/s", 1e12),
                                "relationship_id": "memory_matrix"}]},
        "workload": {"operator": "gemm", "M": 16, "N": 16, "K": 16,
                     "dtype": "fp16", "description": "Dense FP16 C=A@B"},
        "rationale": "Synthetic test fixture only"
    })
