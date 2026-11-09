import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paper2perf.config import Settings, load_env_file, loaded_env_file, parse_env_file


class EnvFileTests(unittest.TestCase):
    def test_parse_env_file_handles_comments_quotes_and_export(self):
        values = parse_env_file(
            "# comment\n"
            "\n"
            "export PAPER2PERF_MODEL=gpt-test\n"
            'PAPER2PERF_API_KEY="sk-test # not a comment"\n'
            "PAPER2PERF_DATA_DIR=./data  # trailing comment\n"
            "PAPER2PERF_REASONING_EFFORT=\n"
            "not a key\n"
        )
        self.assertEqual(values, {
            "PAPER2PERF_MODEL": "gpt-test",
            "PAPER2PERF_API_KEY": "sk-test # not a comment",
            "PAPER2PERF_DATA_DIR": "./data",
            "PAPER2PERF_REASONING_EFFORT": "",
        })

    def test_load_env_file_does_not_override_shell_values(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("PAPER2PERF_MODEL=from-file\nPAPER2PERF_MAX_MODEL_CALLS=7\n")
            with patch.dict(os.environ, {"PAPER2PERF_MODEL": "from-shell"}, clear=True):
                self.assertEqual(load_env_file(path), path)
                self.assertEqual(os.environ["PAPER2PERF_MODEL"], "from-shell")
                self.assertEqual(os.environ["PAPER2PERF_MAX_MODEL_CALLS"], "7")
                self.assertEqual(loaded_env_file(), str(path.resolve()))

    def test_load_env_file_defaults_to_current_directory(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("PAPER2PERF_MODEL=from-cwd\n")
            previous = os.getcwd()
            os.chdir(folder)
            try:
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(load_env_file(), Path(folder) / ".env")
                    self.assertEqual(os.environ["PAPER2PERF_MODEL"], "from-cwd")
            finally:
                os.chdir(previous)

    def test_settings_from_env_uses_env_file(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "custom.env"
            path.write_text(
                "PAPER2PERF_MODEL=file-model\n"
                "PAPER2PERF_API_KEY=file-key\n"
                "PAPER2PERF_BASE_URL=https://provider.example/v1\n"
                "PAPER2PERF_MAX_MODEL_CALLS=9\n"
                "PAPER2PERF_REASONING_EFFORT=low\n"
            )
            with patch.dict(os.environ, {"PAPER2PERF_ENV_FILE": str(path)}, clear=True):
                settings = Settings.from_env()
            self.assertEqual(settings.model, "file-model")
            self.assertEqual(settings.api_key, "file-key")
            self.assertEqual(settings.base_url, "https://provider.example/v1")
            self.assertEqual(settings.max_model_calls, 9)
            self.assertEqual(settings.model_reasoning_effort, "low")

    def test_missing_env_file_is_ignored(self):
        with TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"PAPER2PERF_ENV_FILE": str(Path(folder) / "absent")},
                            clear=True):
                self.assertIsNone(load_env_file())
                self.assertIsNone(loaded_env_file())


if __name__ == "__main__":
    unittest.main()
