from pathlib import Path


def ensure_data_dir(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
