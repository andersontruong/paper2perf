import math
import unittest

from pydantic import ValidationError

from paper2perf.evidence import (
    EvidenceError, numbers, render_answer, verify_answer, verify_citations,
    verify_extraction, verify_proposal,
)
from paper2perf.hwperf import validate_metrics
from paper2perf.schemas import Answer, Assumption, Citation, Proposal
from fixtures import CHUNKS, CITE, extraction, proposal


class EvidenceTests(unittest.TestCase):
    def test_exact_source_citations(self):
        verify_citations([Citation.model_validate(CITE)], CHUNKS)

    def test_fabricated_quote_rejected(self):
        with self.assertRaises(EvidenceError):
            verify_citations([Citation(chunk_id=CHUNKS[0]["id"], quote="Invented source text")], CHUNKS)

    def test_other_document_citation_rejected(self):
        with self.assertRaises(EvidenceError):
            verify_citations([Citation(chunk_id="another-session", quote=CITE["quote"])], CHUNKS)

    def test_missing_chunk_diagnostic_distinguishes_unknown_reference(self):
        with self.assertRaises(EvidenceError) as error:
            verify_citations([Citation(chunk_id="invented-p24-c0", quote=CITE["quote"])], CHUNKS)
        self.assertIn("not in the retrieved context", str(error.exception))
        self.assertIn(CHUNKS[0]["id"], str(error.exception))
        self.assertIn("quote appears verbatim", str(error.exception))

    def test_mismatched_quote_diagnostic_shows_what_was_rejected(self):
        quote = "Invented paraphrase of the hardware specification"
        with self.assertRaises(EvidenceError) as error:
            verify_citations([Citation(chunk_id=CHUNKS[0]["id"], quote=quote)], CHUNKS)
        self.assertIn("quote does not match", str(error.exception))
        self.assertIn(quote, str(error.exception))

    def test_wrong_chunk_is_not_accepted_when_quote_exists_elsewhere(self):
        second = {**CHUNKS[0], "id": "source-p2-c0", "text": "Different original page contents"}
        with self.assertRaises(EvidenceError) as error:
            verify_citations([Citation(chunk_id=CHUNKS[0]["id"], quote=second["text"])],
                             [*CHUNKS, second])
        self.assertIn(second["id"], str(error.exception))
        self.assertIn("Correct its reference", str(error.exception))

    def test_whitespace_normalization_is_allowed_but_empty_quotes_are_not(self):
        verify_citations([Citation(chunk_id=CHUNKS[0]["id"],
                                  quote=CITE["quote"].replace(" ", "\n"))], CHUNKS)
        with self.assertRaisesRegex(EvidenceError, "empty after whitespace"):
            verify_citations([Citation(chunk_id=CHUNKS[0]["id"], quote=" " * 8)], CHUNKS)

    def test_numerical_answer_and_references(self):
        answer = Answer.model_validate({"claims": [{"text": "The memory capacity is 8 GB.",
                                                    "citations": [CITE]}]})
        verify_answer(answer, CHUNKS)
        text, refs = render_answer(answer, CHUNKS)
        self.assertIn("p. 1", text)
        self.assertEqual(refs[0]["document_id"], "source")

    def test_answer_cannot_invent_number(self):
        answer = Answer.model_validate({"claims": [{"text": "Memory capacity is 80 GB.",
                                                    "citations": [CITE]}]})
        with self.assertRaises(EvidenceError):
            verify_answer(answer, CHUNKS)

    def test_fact_preserves_original_number(self):
        data = extraction()
        verify_extraction(data, CHUNKS)
        data.facts[0].value = 8e9
        with self.assertRaises(EvidenceError):
            verify_extraction(data, CHUNKS)

    def test_rejection_identifies_claim_number_and_cited_values(self):
        answer = Answer.model_validate({"claims": [{"text": "Memory capacity is 80 GB.",
                                                    "citations": [CITE]}]})
        with self.assertRaises(EvidenceError) as error:
            verify_answer(answer, CHUNKS)
        self.assertIn("unsupported number 80", str(error.exception))
        self.assertIn("Memory capacity is 80 GB", str(error.exception))
        self.assertIn("Numbers in its cited excerpts", str(error.exception))

    def test_signed_decimals_and_pdf_thousands_separators(self):
        self.assertEqual(numbers("−0.5 +.25 1e-6 1,555 1\u202f555 1\u00a0555"),
                         [-0.5, .25, 1e-6, 1555, 1555, 1555])
        self.assertEqual(numbers("A100 FP16 HBM2 312 624"), [312, 624])

    def test_spelled_out_count_is_validated_from_exact_excerpt(self):
        text = "This GPU has four Tensor Cores per SM, delivering 1024 operations per clock."
        chunks = [{**CHUNKS[0], "text": text}]
        cite = Citation(chunk_id=CHUNKS[0]["id"], quote=text)
        data = extraction()
        data.facts = [data.facts[0]]
        data.relationships = []
        data.facts[0].value = 4
        data.facts[0].citations = [cite]
        verify_extraction(data, chunks)
        data.facts[0].value = 8
        with self.assertRaises(EvidenceError):
            verify_extraction(data, chunks)

    def test_word_counts_are_not_partial_or_substring_matches(self):
        self.assertEqual(numbers("FOUR cores; forty-four lanes; twenty eight engines"), [4, 44, 28])
        for text in ["fourth", "fourfold", "four hundred", "one hundred and four",
                     "four thousand", "four point five", "negative four"]:
            self.assertEqual(numbers(text), [], text)

    def test_spelled_out_answer_cannot_introduce_unsupported_count(self):
        answer = Answer.model_validate({"claims": [{"text": "There are four cores.",
                                                    "citations": [CITE]}]})
        with self.assertRaises(EvidenceError):
            verify_answer(answer, CHUNKS)

    def test_pdf_numeric_formatting_preserves_values(self):
        text = "Memory bandwidth is 1\u202f555 GB/s."
        chunks = [{**CHUNKS[0], "text": text}]
        answer = Answer.model_validate({"claims": [{"text": "Memory bandwidth is 1,555 GB/s.",
             "citations": [{"chunk_id": CHUNKS[0]["id"], "quote": text}]}]})
        verify_answer(answer, chunks)

    def test_negative_or_fractional_inventions_are_rejected(self):
        for claim in ["Memory capacity is -8 GB.", "Memory capacity is .8 GB."]:
            with self.subTest(claim=claim), self.assertRaises(EvidenceError):
                verify_answer(Answer.model_validate({"claims": [{"text": claim,
                               "citations": [CITE]}]}), CHUNKS)

    def test_unsourced_ratio_is_rejected_even_with_operand_citations(self):
        text = "Dense throughput 312 TFLOPS; sparse throughput 624 TFLOPS."
        chunks = [{**CHUNKS[0], "text": text}]
        answer = Answer.model_validate({"claims": [{"text": "Sparse throughput is 2 times dense.",
             "citations": [{"chunk_id": CHUNKS[0]["id"], "quote": text}]}]})
        with self.assertRaises(EvidenceError):
            verify_answer(answer, chunks)

    def test_duplicate_fact_ids(self):
        data = extraction()
        data.facts[1].id = data.facts[0].id
        with self.assertRaises(EvidenceError):
            verify_extraction(data, CHUNKS)

    def test_unit_conversion_lineage(self):
        self.assertTrue(verify_proposal(proposal(), extraction(), [], CHUNKS))

    def test_incomplete_hardware_preserves_gaps_instead_of_topology_failure(self):
        candidate = proposal().model_dump()
        candidate["hardware"]["computes"].append(
            {**candidate["hardware"]["computes"][0], "id": "unconnected_compute"})
        candidate["missing"] = ["Required cache-to-compute bandwidth is not documented"]
        parsed = Proposal.model_validate(candidate)
        self.assertIsNone(parsed.hardware)
        self.assertEqual(parsed.missing, candidate["missing"])
        self.assertIsNotNone(parsed.workload)
        self.assertIsNotNone(candidate["hardware"])
        with self.assertRaisesRegex(EvidenceError, "incomplete"):
            verify_proposal(parsed, extraction(), [], CHUNKS)

    def test_disconnected_hardware_without_explicit_gaps_still_fails(self):
        candidate = proposal().model_dump()
        candidate["hardware"]["computes"].append(
            {**candidate["hardware"]["computes"][0], "id": "unconnected_compute"})
        with self.assertRaisesRegex(ValidationError, "reachable"):
            Proposal.model_validate(candidate)

    def test_reversed_link_matches_undirected_simulator(self):
        candidate = proposal().model_dump()
        link = candidate["hardware"]["links"][0]
        link["source"], link["target"] = link["target"], link["source"]
        parsed = Proposal.model_validate(candidate)
        self.assertTrue(verify_proposal(parsed, extraction(), [], CHUNKS))

    def test_reverse_duplicate_is_not_a_second_bandwidth_channel(self):
        candidate = proposal().model_dump()
        link = candidate["hardware"]["links"][0]
        candidate["hardware"]["links"].append(
            {**link, "source": link["target"], "target": link["source"]})
        with self.assertRaisesRegex(ValidationError, "duplicate undirected"):
            Proposal.model_validate(candidate)

    def test_documented_per_second_unit_alias(self):
        data = extraction()
        data.facts[1].unit = "TB/sec"
        self.assertTrue(verify_proposal(proposal(), data, [], CHUNKS))
        model = proposal()
        data.facts[1].unit = "GB/sec"
        data.facts[1].value = 1000
        model.hardware.links[0].bandwidth.scale = 1e9
        self.assertTrue(verify_proposal(model, data, [], CHUNKS))

    def test_arbitrary_scale_rejected(self):
        model = proposal()
        model.hardware.memories[0].capacity.scale = 2e9
        model.hardware.memories[0].capacity.value = 16e9
        with self.assertRaises(EvidenceError):
            verify_proposal(model, extraction(), [], CHUNKS)

    def test_cross_variant_rejected(self):
        data = extraction()
        data.facts[0].variant = "Different device"
        with self.assertRaises(EvidenceError):
            verify_proposal(proposal(), data, [], CHUNKS)

    def test_sparse_throughput_rejected(self):
        data = extraction()
        data.facts[2].mode = "sparse"
        with self.assertRaises(EvidenceError):
            verify_proposal(proposal(), data, [], CHUNKS)

    def test_precision_mismatch_rejected(self):
        data = extraction()
        data.facts[2].precision = "fp32"
        with self.assertRaises(EvidenceError):
            verify_proposal(proposal(), data, [], CHUNKS)

    def test_per_core_rate_not_device_rate(self):
        data = extraction()
        data.facts[2].aggregation = "component"
        with self.assertRaises(EvidenceError):
            verify_proposal(proposal(), data, [], CHUNKS)

    def test_no_implicit_assumptions(self):
        model = proposal()
        q = model.hardware.links[0].bandwidth
        q.basis, q.fact_ids, q.assumption_id = "assumed", [], "bandwidth_guess"
        with self.assertRaises(EvidenceError):
            verify_proposal(model, extraction(), [], CHUNKS)
        allowed = Assumption(id="bandwidth_guess", value=q.value, unit=q.unit,
                             rationale="Explicit synthetic test assumption")
        self.assertTrue(verify_proposal(model, extraction(), [allowed], CHUNKS))

    def test_conflicts_block_execution(self):
        data = extraction()
        data.conflicts = ["Two documents describe different device variants"]
        with self.assertRaises(EvidenceError):
            verify_proposal(proposal(), data, [], CHUNKS)

    def test_invented_connectivity_rejected(self):
        model = proposal()
        model.hardware.links[0].relationship_id = "invented"
        with self.assertRaises(EvidenceError):
            verify_proposal(model, extraction(), [], CHUNKS)

    def test_disconnected_topology_rejected(self):
        data = proposal().model_dump()
        data["hardware"]["memories"].append({"id": "orphan", "capacity": data["hardware"]["memories"][0]["capacity"]})
        with self.assertRaises(ValidationError):
            Proposal.model_validate(data)

    def test_nan_and_invalid_shapes_rejected(self):
        data = proposal().model_dump()
        data["workload"]["M"] = True
        with self.assertRaises(ValidationError):
            Proposal.model_validate(data)
        data = proposal().model_dump()
        data["hardware"]["memories"][0]["capacity"]["value"] = math.nan
        with self.assertRaises(ValidationError):
            Proposal.model_validate(data)

    def test_metrics_require_real_numeric_backend_result(self):
        for value in [True, "0.1", math.nan, -1, 0]:
            with self.assertRaises(ValueError):
                validate_metrics({"backend": "hw-perf", "makespan_seconds": value})
        validate_metrics({"backend": "hw-perf", "makespan_seconds": 1e-6})


if __name__ == "__main__":
    unittest.main()
