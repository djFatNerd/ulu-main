"""Utilities for persistent JSON caching with basic corruption recovery."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheResult:
    value: Optional[Any]
    hit: bool


class JSONCache:
    """Simple filesystem-backed JSON cache with corruption handling."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in key)
        if len(safe) > 64:
            digest = sha1(key.encode("utf-8")).hexdigest()
            safe = f"{safe[:32]}_{digest[:16]}"
        return self.root / f"{safe}.json"

    def get(self, key: str) -> CacheResult:
        path = self._path_for_key(key)
        if not path.exists():
            return CacheResult(value=None, hit=False)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            self.delete_safe(key)
            return CacheResult(value=None, hit=False)
        return CacheResult(value=data, hit=True)

    def set(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False)
            tmp.replace(path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def delete_safe(self, key: str) -> None:
        path = self._path_for_key(key)
        try:
            path.unlink()
        except OSError:
            return
