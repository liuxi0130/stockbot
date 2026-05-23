import os
import re
import yaml


def _resolve_env(value: str) -> str:
    pattern = re.compile(r'\$\{(\w+)(?::-([^}]*))?\}')

    def replacer(m):
        var_name = m.group(1)
        default = m.group(2)
        return os.environ.get(var_name, default or "")

    return pattern.sub(replacer, value)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = _resolve_env(raw)
    return yaml.safe_load(raw)
