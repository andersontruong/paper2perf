"""Do not confuse scope selections or PDF annotations with source contradictions."""
import unittest

from paper2perf.compiler import compile_blueprint
from paper2perf.schemas import DesignBlocker
from test_grounded_compiler import inputs


class ConflictContractTests(unittest.TestCase):
    def test_reported_single_fact_resolved_comparison_requires_design_repair(self):
        ex, _, bp = inputs()
        bp.blockers = [DesignBlocker(category='source_conflict', reference_id='a100_sm_count',
            reason='A100 product count is selected; full-die count excluded; table footnote looks different.')]
        with self.assertRaisesRegex(ValueError, 'at least two verified'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_different_variants_are_not_same_scope_conflicts(self):
        ex, _, bp = inputs()
        original = next(f for f in ex.facts if f.id == 'a100_sm_count')
        other = original.model_copy(deep=True)
        other.id, other.variant, other.value = 'full_die_count', 'Different full-die variant', 128
        ex.facts.append(other)
        bp.blockers = [DesignBlocker(category='source_conflict', reason='Product differs from full die.',
                                    conflicting_fact_ids=[original.id, other.id])]
        with self.assertRaisesRegex(ValueError, 'scope selection'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')

    def test_same_scope_incompatible_values_remain_blocked(self):
        ex, _, bp = inputs()
        original = next(f for f in ex.facts if f.id == 'a100_sm_count')
        other = original.model_copy(deep=True)
        # Synthetic contradictory value for the contract test, not a source claim.
        other.id, other.value = 'contradictory_count', 107
        ex.facts.append(other)
        bp.blockers = [DesignBlocker(category='source_conflict', reason='Two same-scope counts disagree.',
                                    conflicting_fact_ids=[original.id, other.id])]
        p, _, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertIn('source_conflict', gaps[0])

    def test_unresolved_scope_uncertainty_is_not_discarded(self):
        ex, _, bp = inputs()
        bp.blockers = [DesignBlocker(category='source_scope', reason='Source applicability is unresolved.',
                                    reference_id='a100_sm_count')]
        p, _, _, gaps = compile_blueprint(bp, ex, [], 'hbm-proxy')
        self.assertIsNone(p.hardware)
        self.assertTrue(gaps)

    def test_equal_values_are_not_contradictory(self):
        ex, _, bp = inputs()
        original = next(f for f in ex.facts if f.id == 'a100_sm_count')
        other = original.model_copy(deep=True)
        other.id = 'same_count_again'
        ex.facts.append(other)
        bp.blockers = [DesignBlocker(category='source_conflict', reason='Repeated source count.',
                                    conflicting_fact_ids=[original.id, other.id])]
        with self.assertRaisesRegex(ValueError, 'matching values'):
            compile_blueprint(bp, ex, [], 'hbm-proxy')
