"""Mechanical evidence checks; these do not prove semantic entailment."""
import math
import re

from .schemas import Answer, Extraction, Proposal
from .calculations import UNITS, calculate


class EvidenceError(ValueError):
    pass


def normalized(text):
    return " ".join(text.split())


def verify_citations(citations, chunks):
    by_id = {c["id"]: c for c in chunks}
    for cite in citations:
        chunk = by_id.get(cite.chunk_id)
        quote = normalized(cite.quote)
        if not quote:
            raise EvidenceError(f"Citation quote is empty after whitespace normalization: {cite.chunk_id}")
        if chunk and quote in normalized(chunk["text"]):
            continue
        matches = [c["id"] for c in chunks if quote in normalized(c["text"])]
        guidance = (
            f" This quote appears verbatim in retrieved chunks: {matches}. Correct its reference."
            if matches else " Copy a contiguous verbatim substring from the cited chunk; do not paraphrase."
        )
        if chunk is None:
            raise EvidenceError(
                f"Citation chunk {cite.chunk_id!r} is not in the retrieved context. "
                f"Available chunk IDs: {list(by_id)}." + guidance
            )
        raise EvidenceError(
            f"Citation quote does not match retrieved chunk {cite.chunk_id!r}: {cite.quote[:200]!r}."
            + guidance
        )


def numbers(text):
    # PDF text can use typographic minus signs and nonbreaking thousands separators.
    # Do not combine ASCII-space-separated values: they may be adjacent table columns.
    separators = ",\u00a0\u202f\u2009"
    tokens = re.findall(
        r"(?<![\w.])[+-]?(?:(?:\d{1,3}(?:[,\u00a0\u202f\u2009]\d{3})+|\d+)"
        r"(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", text.replace("\u2212", "-")
    )
    values = [float(token.translate(str.maketrans("", "", separators))) for token in tokens]
    # Technical prose also states counts in words. Parse whole cardinal phrases
    # conservatively: never treat 'four hundred' or 'forty-four' as just four.
    small = dict(zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
    tens = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
    vocabulary = "|".join([*small, *tens, "hundred", "thousand", "million", "billion",
                           "trillion", "and", "point", "minus", "negative"])
    word = rf"(?:{vocabulary})"
    phrases = re.finditer(rf"(?<!\w){word}(?:[\s\-]+{word})*(?!\w)", text.lower())
    for match in phrases:
        parts = re.split(r"[\s\-]+", match.group())
        if len(parts) == 1 and parts[0] in small:
            values.append(float(small[parts[0]]))
        elif len(parts) == 1 and parts[0] in tens:
            values.append(float(tens[parts[0]]))
        elif (len(parts) == 2 and parts[0] in tens
              and parts[1] in small and 0 < small[parts[1]] < 10):
            values.append(float(tens[parts[0]] + small[parts[1]]))
    return values


def verify_answer(answer: Answer, chunks):
    for index, claim in enumerate(answer.claims, 1):
        verify_citations(claim.citations, chunks)
        source_numbers = numbers(" ".join(c.quote for c in claim.citations))
        for n in numbers(claim.text):
            if not any(math.isclose(n, x, rel_tol=1e-9) for x in source_numbers):
                raise EvidenceError(
                    f"Claim {index} introduces unsupported number {n:g}: {claim.text!r}. "
                    f"Numbers in its cited excerpts: {source_numbers}. "
                    "Preserve source numbers and units; omit unsupported ratios, conversions, "
                    "list numbering, and inline page/table numbers from the claim text."
                )


def verify_extraction(extraction: Extraction, chunks):
    ids = [f.id for f in extraction.facts]
    if len(set(ids)) != len(ids):
        raise EvidenceError("Duplicate fact IDs")
    relationships = [r.id for r in extraction.relationships]
    if len(set(relationships)) != len(relationships):
        raise EvidenceError("Duplicate relationship IDs")
    for relationship in extraction.relationships:
        verify_citations(relationship.citations, chunks)
    for fact in extraction.facts:
        verify_citations(fact.citations, chunks)
        if not any(math.isclose(fact.value, n, rel_tol=1e-9) for c in fact.citations
                   for n in numbers(c.quote)):
            raise EvidenceError(
                f"Fact {fact.id!r} has value {fact.value:g}, which does not preserve a source number. "
                f"Numbers in its cited excerpts: {numbers(' '.join(c.quote for c in fact.citations))}. "
                "Keep the original numeric value and unit; conversions belong in model construction."
            )




def quantities(hardware):
    for node in hardware.memories:
        yield f"{node.id}.capacity", node.capacity, "byte"
    for node in hardware.computes:
        yield f"{node.id}.throughput", node.throughput, "FLOP/s"
    for link in hardware.links:
        yield f"{link.source}->{link.target}.bandwidth", link.bandwidth, "byte/s"


def verify_proposal(proposal: Proposal, extraction: Extraction, assumptions, chunks=None):
    if proposal.hardware is None or proposal.workload is None or proposal.missing:
        raise EvidenceError("Model is incomplete")
    if extraction.conflicts:
        raise EvidenceError("Unresolved conflicting source evidence")
    facts = {f.id: f for f in extraction.facts}
    relations = {r.id: r for r in extraction.relationships}
    from .topology import check_bindings
    node_ids = {n.id for n in [*proposal.hardware.memories, *proposal.hardware.computes]}
    compute_ids = {n.id for n in proposal.hardware.computes}
    memory_ids = {n.id for n in proposal.hardware.memories}
    try:
        from .memory_pools import check_memory_pools
        aliases, pool_relationships = check_memory_pools(
            proposal.hardware.memory_pools, proposal.hardware.memories, proposal.hardware.computes,
            extraction, proposal.hardware.variant)
        mapping = {**aliases, **check_bindings(proposal.hardware.resource_bindings, extraction.relationships,
                                             memory_ids, node_ids, proposal.hardware.variant, aliases)}
    except ValueError as exc:
        raise EvidenceError(str(exc)) from exc
    covered = set(pool_relationships)
    for link in proposal.hardware.links:
        ids = list(dict.fromkeys([link.relationship_id, *link.relationship_ids]))
        projected, shared_projected, contains = False, False, False
        for rid in ids:
            r = relations.get(rid)
            if r is None or r.variant != proposal.hardware.variant:
                raise EvidenceError("Hardware link lacks matching source-backed connectivity")
            endpoints = {mapping.get(r.source, r.source), mapping.get(r.target, r.target)}
            if endpoints != {link.source, link.target}:
                raise EvidenceError("Hardware link lacks matching source-backed connectivity")
            if r.kind == "contains":
                if r.source not in compute_ids or mapping.get(r.target, r.target) not in memory_ids:
                    raise EvidenceError("Containment access requires a compute parent and memory child")
                contains = True
            elif r.kind != "connects":
                raise EvidenceError("Unsupported source relationship for connectivity")
            projected |= endpoints != {r.source, r.target}
            shared_projected |= r.source in aliases or r.target in aliases
            covered.add(rid)
        expected_basis = ("shared_memory_projection" if shared_projected else
                          "containment_projection" if projected else
                          "containment_access" if contains else "documented")
        if link.topology_basis != expected_basis:
            raise EvidenceError("Hardware connectivity abstraction is not labeled accurately")
    for r in extraction.relationships:
        if r.variant != proposal.hardware.variant:
            continue
        if r.kind in {"connects", "shares_resource"} and r.id not in covered:
            raise EvidenceError("Model omits documented hardware connectivity")
        if r.kind == "contains" and r.source in compute_ids and mapping.get(r.target, r.target) not in node_ids:
            raise EvidenceError("Model omits a documented resource inside the selected compute container")
    allowed = {a.id: a for a in assumptions}
    if chunks is not None:
        verify_citations(proposal.hardware.citations, chunks)
    for field, q, unit in quantities(proposal.hardware):
        if q.unit != unit:
            raise EvidenceError(f"Wrong dimension for {field}")
        if q.basis == "assumed":
            a = allowed.get(q.assumption_id)
            if not a or a.unit != q.unit or not math.isclose(a.value, q.value, rel_tol=1e-9):
                raise EvidenceError(f"Unapproved assumption for {field}")
            if q.fact_ids:
                raise EvidenceError("An assumption cannot masquerade as documented evidence")
            continue
        if q.assumption_id or not q.fact_ids or any(fid not in facts for fid in q.fact_ids):
            raise EvidenceError(f"Missing or invalid fact lineage for {field}")
        fact = facts[q.fact_ids[0]]
        if fact.variant != proposal.hardware.variant:
            raise EvidenceError(f"Mixed hardware variants for {field}")
        if any(facts[fid].variant != proposal.hardware.variant for fid in q.fact_ids):
            raise EvidenceError(f"Mixed hardware variants for {field}")
        if q.derivation != "unit_conversion" and q.basis != "derived":
            raise EvidenceError(f"Arithmetic quantities must be labeled derived for {field}")
        if unit == "FLOP/s" and fact.precision != proposal.workload.dtype:
            raise EvidenceError(f"Compute precision mismatch for {field}")
        if unit == "FLOP/s" and any(
                c.id + ".throughput" == field and c.kind == "matrix"
                for c in proposal.hardware.computes) and fact.mode != "dense":
            raise EvidenceError(f"Dense GEMM requires dense throughput evidence for {field}")
        try:
            expected, expected_scale = calculate(facts, q.fact_ids, q.derivation, q.unit)
        except ValueError as exc:
            raise EvidenceError(f"Invalid derivation for {field}: {exc}") from exc
        if not math.isclose(q.scale, expected_scale, rel_tol=1e-12):
            raise EvidenceError(f"Unverified conversion multiplier for {field}")
        if q.basis == "documented" and q.scale != 1:
            raise EvidenceError("Converted values must be labeled derived")
        if not math.isclose(q.value, expected, rel_tol=1e-9):
            raise EvidenceError(f"Value disagrees with source lineage for {field}")
    # This IR uses aggregate, single-device rates; no unsupported aggregation arithmetic.
    return True


def render_answer(answer, chunks):
    by_id = {c["id"]: c for c in chunks}
    text = []
    references = []
    for claim in answer.claims:
        refs = []
        for cite in claim.citations:
            c = by_id[cite.chunk_id]
            ref = {"document_id": c["document"], "page": c["page"],
                   "name": c["name"], **cite.model_dump()}
            if c.get("artifact"):
                ref.update({"artifact": c["artifact"], "run_id": c["run_id"]})
            references.append(ref)
            refs.append(f"[{c['name']}, p. {c['page']}]")
        text.append(claim.text + " " + " ".join(refs))
    text.extend("Limitation: " + x for x in answer.limitations)
    if not text:
        text.append("The selected documents do not provide enough evidence to answer.")
    return "\n\n".join(text), references
