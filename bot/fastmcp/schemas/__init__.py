"""Cached metadata schemas for project introspection."""

import json
from pathlib import Path

PROJECT_META_PATH = Path(__file__).parent / "project_meta.json"


def save_project_metadata(data: dict) -> None:
    PROJECT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROJECT_META_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_project_metadata() -> dict:
    if not PROJECT_META_PATH.exists():
        return {"version": 1, "last_updated": None}
    try:
        with open(PROJECT_META_PATH) as f:
            data = json.load(f)
        data["version"] = max(data.get("version", 1), 1)
        return data
    except Exception:
        return {"version": 1, "last_updated": None}
