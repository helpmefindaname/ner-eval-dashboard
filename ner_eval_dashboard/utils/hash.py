import hashlib
import json
from typing import Any


def json_hash(data: Any) -> str:
    return hashlib.md5(json.dumps(data).encode("utf-8")).hexdigest()
