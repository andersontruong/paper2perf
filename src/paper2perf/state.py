from typing import Any, TypedDict


class InvestigationState(TypedDict, total=False):
    run_id: str
    session_id: str
    message: str
    history: list[dict]
    document_ids: list[str]
    mode: str
    assumptions: list[dict]
    action: str
    query: str
    queries: list[str]
    chunks: list[dict]
    retrieval_round: int
    repairs: int
    extraction: dict | None
    extraction_attempts: int
    extraction_repairs: int
    extraction_errors: list[str]
    rejected_extraction: dict | None
    proposal: dict | None
    validation_errors: list[str]
    decision: str
    answer: dict | None
    answer_attempts: int
    answer_repairs: int
    answer_errors: list[str]
    metrics: dict
    analysis: dict
    status: str
    response: str
    references: list[dict]
    last_result: dict[str, Any] | None

    construction: str
    bandwidth_policy: str
    evidence_run_id: str | None
    seed_extraction: dict | None
    blueprint: dict | None
    previous_blueprint: dict | None
    workload_contract: dict | None
    deterministic_analysis: bool
