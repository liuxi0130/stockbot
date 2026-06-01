import os
import re
import yaml
from pathlib import Path


def _load_dotenv(path: str = ".env"):
    """Load key=value pairs from .env file into os.environ (if file exists)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k not in os.environ:
                os.environ[k] = v


def _resolve_env(value: str) -> str:
    pattern = re.compile(r'\$\{(\w+)(?::-([^}]*))?\}')

    def replacer(m):
        var_name = m.group(1)
        default = m.group(2)
        return os.environ.get(var_name, default or "")

    return pattern.sub(replacer, value)


def load_config(path: str = "config.yaml") -> dict:
    _load_dotenv(".env")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = _resolve_env(raw)
    return yaml.safe_load(raw)
