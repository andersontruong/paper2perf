"""A real bundled-backend smoke test; never replaces the backend with a prediction stub."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fixtures import proposal


AVAILABLE = all(importlib.util.find_spec(name) for name in [
    "torch", "networkx", "pydot", "pysmt", "z3", "parabolic", "hw_model", "dataflow_mapper"
])


@unittest.skipUnless(AVAILABLE, "Real hw-perf integration requires the installed simulator dependencies")
class SimulatorIntegrationTests(unittest.TestCase):
    def test_real_capture_mapping_and_schedule(self):
        from paper2perf.hwperf_worker import execute
        with TemporaryDirectory() as temp:
            result = execute(proposal(), temp)
            self.assertGreater(result["makespan_seconds"], 0)
            self.assertEqual(result["backend"], "hw-perf")
            self.assertGreater(result["recorded_scope_count"], 0)
            self.assertTrue((Path(temp) / "events.json").exists())


if __name__ == "__main__":
    unittest.main()
