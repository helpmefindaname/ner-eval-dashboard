import json
import os
from pathlib import Path
from typing import Any, Dict

cache_root = Path(os.getenv("NER_DASHBOARD_CACHE", Path(Path.home(), ".ner_dashboard_cache")))

cache_root.mkdir(exist_ok=True, parents=True)


def __key2path(key: str) -> Path:
    return cache_root / f"{key}.json"


def has_cache(key: str) -> bool:
    return __key2path(key).exists()


def load_cache(key: str) -> Dict[str, Any]:
    with __key2path(key).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(key: str, data: Dict[str, Any]) -> None:
    with __key2path(key).open("w", encoding="utf-8") as f:
        return json.dump(data, f)


def delete_cache(key: str) -> None:
    __key2path(key).unlink()


def delete_full_cache() -> None:
    for f in cache_root.glob("*.json"):
        f.unlink()
