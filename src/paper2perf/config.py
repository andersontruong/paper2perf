from dataclasses import dataclass
import math
import os
from pathlib import Path
import re


ENV_FILE_VARIABLE = "PAPER2PERF_ENV_FILE"
_loaded_env_file = None


def parse_env_file(text):
    """Parse a small KEY=VALUE file without importing a dotenv dependency."""
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if value[:1] in {"'", '"'}:
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            # An unquoted "#" begins a comment only after whitespace.
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key] = value
    return values


def load_env_file(path=None):
    """Load a local .env file without overriding variables already in the shell.

    Defaults to .env in the current working directory. Set PAPER2PERF_ENV_FILE
    to load a different file. Returns the path that was read, or None.
    """
    global _loaded_env_file
    _loaded_env_file = None
    if path is None:
        override = os.getenv(ENV_FILE_VARIABLE)
        path = Path(override).expanduser() if override else Path.cwd() / ".env"
    path = Path(path)
    if not path.is_file():
        return None
    for key, value in parse_env_file(path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)
    _loaded_env_file = str(path.resolve())
    return path


def loaded_env_file():
    return _loaded_env_file


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model: str = ""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_document_bytes: int = 25 * 1024 * 1024
    max_pages: int = 400
    max_model_calls: int = 12
    max_retrievals: int = 3
    max_repairs: int = 2
    simulation_timeout: int = 120
    model_http_retries: int = 3
    model_retry_base_seconds: float = 2.0
    model_retry_max_wait_seconds: float = 60.0
    model_min_interval_seconds: float = 0.0
    model_max_completion_tokens: int = 6000
    model_reasoning_effort: str = ""

    def __post_init__(self):
        if not 0 <= self.model_http_retries <= 5:
            raise ValueError("PAPER2PERF_HTTP_RETRIES must be between 0 and 5")
        if self.max_model_calls < 1 or self.model_max_completion_tokens < 1:
            raise ValueError("Model-call and completion-token limits must be positive")
        for value in [self.model_retry_base_seconds, self.model_retry_max_wait_seconds,
                      self.model_min_interval_seconds]:
            if not math.isfinite(value) or value < 0:
                raise ValueError("Model request delays must be finite and nonnegative")
        if self.model_reasoning_effort not in {"", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported PAPER2PERF_REASONING_EFFORT value")

    @classmethod
    def from_env(cls):
        # Shell environment wins; .env only fills unset variables.
        load_env_file()
        return cls(
            data_dir=Path(os.getenv("PAPER2PERF_DATA_DIR", "data")).resolve(),
            model=os.getenv("PAPER2PERF_MODEL", ""),
            api_key=os.getenv("PAPER2PERF_API_KEY", ""),
            base_url=os.getenv("PAPER2PERF_BASE_URL", "https://api.openai.com/v1"),
            max_model_calls=int(os.getenv("PAPER2PERF_MAX_MODEL_CALLS") or "12"),
            model_http_retries=int(os.getenv("PAPER2PERF_HTTP_RETRIES") or "3"),
            model_retry_base_seconds=float(os.getenv("PAPER2PERF_RETRY_BASE_SECONDS") or "2"),
            model_retry_max_wait_seconds=float(os.getenv("PAPER2PERF_RETRY_MAX_WAIT_SECONDS") or "60"),
            model_min_interval_seconds=float(os.getenv("PAPER2PERF_MODEL_MIN_INTERVAL_SECONDS") or "0"),
            model_max_completion_tokens=int(os.getenv("PAPER2PERF_MAX_COMPLETION_TOKENS") or "6000"),
            model_reasoning_effort=(os.getenv("PAPER2PERF_REASONING_EFFORT") or "").strip(),
        )
