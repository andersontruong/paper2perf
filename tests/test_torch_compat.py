import importlib.util
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from paper2perf.torch_compat import shape_env_compatibility, torch_version_status


class ShapeEnvCompatibilityTests(unittest.TestCase):
    def test_version_pin_accepts_cpu_and_cuda_local_build_suffixes(self):
        for version in ["2.10.0", "2.10.0+cpu", "2.10.0+cu128"]:
            with self.subTest(version=version), patch(
                    "paper2perf.torch_compat.metadata.version", return_value=version):
                self.assertTrue(torch_version_status()["matches"])

    def test_version_pin_rejects_other_releases(self):
        for version in ["2.9.1", "2.10.1", "2.11.0", "2.10.0.dev20260101"]:
            with self.subTest(version=version), patch(
                    "paper2perf.torch_compat.metadata.version", return_value=version):
                self.assertFalse(torch_version_status()["matches"])

    def legacy_type(self):
        class LegacyShapeEnv:
            def __init__(self):
                self.var_to_val = {}
                self.unbacked_var_to_val = {}

            def add_var_to_val(self, expr, value):
                self.var_to_val[expr] = value
        return LegacyShapeEnv

    def test_legacy_storage_and_registration_are_aliased_together(self):
        legacy = self.legacy_type()
        with shape_env_compatibility(legacy) as metadata:
            env = legacy()
            env.add_backed_var_to_val("M", 16)
            self.assertIs(env.backed_var_to_val, env.var_to_val)
            self.assertEqual(env.backed_var_to_val["M"], 16)
            self.assertEqual(env.unbacked_var_to_val, {})
            self.assertEqual(set(metadata["aliases_applied"]), {
                "backed_var_to_val", "add_backed_var_to_val"
            })
        self.assertFalse(hasattr(legacy, "backed_var_to_val"))
        self.assertFalse(hasattr(legacy, "add_backed_var_to_val"))

    def test_modern_api_is_left_unchanged(self):
        class ModernShapeEnv:
            def __init__(self):
                self.backed_var_to_val = {}

            def add_backed_var_to_val(self, expr, value):
                self.backed_var_to_val[expr] = value
        original = ModernShapeEnv.add_backed_var_to_val
        with shape_env_compatibility(ModernShapeEnv) as metadata:
            self.assertEqual(metadata["aliases_applied"], [])
            self.assertIs(ModernShapeEnv.add_backed_var_to_val, original)
        self.assertIs(ModernShapeEnv.add_backed_var_to_val, original)

    def test_aliases_are_removed_after_failure(self):
        legacy = self.legacy_type()
        with self.assertRaisesRegex(ValueError, "capture failed"):
            with shape_env_compatibility(legacy):
                raise ValueError("capture failed")
        self.assertFalse(hasattr(legacy, "backed_var_to_val"))
        self.assertFalse(hasattr(legacy, "add_backed_var_to_val"))

    def test_nested_compatibility_does_not_remove_outer_aliases(self):
        legacy = self.legacy_type()
        with shape_env_compatibility(legacy):
            with shape_env_compatibility(legacy) as metadata:
                self.assertEqual(metadata["aliases_applied"], [])
            self.assertTrue(hasattr(legacy, "backed_var_to_val"))
        self.assertFalse(hasattr(legacy, "backed_var_to_val"))

    def test_worker_applies_compatibility_and_records_runtime(self):
        from paper2perf.hwperf_worker import execute
        legacy = self.legacy_type()

        def capture_and_map(proposal, output_dir):
            env = legacy()
            env.add_backed_var_to_val("M", 16)
            self.assertEqual(env.backed_var_to_val["M"], 16)
            return {"backend": "hw-perf", "makespan_seconds": 0.001}

        with patch.dict("sys.modules", {"torch": SimpleNamespace(__version__="2.10.0+cpu")}), \
                patch("paper2perf.torch_compat.shape_env_compatibility",
                      side_effect=lambda: shape_env_compatibility(legacy)), \
                patch("paper2perf.hwperf_worker._execute", side_effect=capture_and_map):
            result = execute(None, "unused-test-path")
        self.assertEqual(result["runtime"]["torch_version"], "2.10.0+cpu")
        self.assertEqual(result["runtime"]["shape_env_compatibility"]["backed_storage"], "var_to_val")
        self.assertFalse(hasattr(legacy, "backed_var_to_val"))

    def test_unbacked_storage_is_never_used_as_a_substitute(self):
        class UnbackedOnly:
            def __init__(self):
                self.unbacked_var_to_val = {}
        with self.assertRaisesRegex(RuntimeError, "backed-symbol storage"):
            with shape_env_compatibility(UnbackedOnly):
                self.fail("Missing backed storage should be rejected")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch dependency not installed")
    def test_actual_shape_env_can_create_and_register_backed_symbols(self):
        import sympy as sp
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from parabolic.symbol import create_symint

        with shape_env_compatibility():
            env = ShapeEnv()
            try:
                value = create_symint(env, 7)
                self.assertIs(value.node.shape_env, env)
                named = sp.Symbol("test_dimension", positive=True, integer=True)
                env.add_backed_var_to_val(named, 7)
                self.assertEqual(env.backed_var_to_val[named], 7)
            finally:
                cleanup = getattr(env, "cleanup", None)
                if callable(cleanup):
                    cleanup()


if __name__ == "__main__":
    unittest.main()
