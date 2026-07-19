"""Small, dependency-free I/O helpers shared by every script.

The two ideas that make the whole run "fire once and forget":
  * DONE markers  -> a unit that finished never reruns (idempotent / resumable).
  * atomic writes -> a unit killed mid-write never leaves a half-file that a
                     later stage would silently consume.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Done markers
# --------------------------------------------------------------------------- #
def marker_path(out_dir: Path | str) -> Path:
    return Path(out_dir) / "_DONE"


def is_done(out_dir: Path | str) -> bool:
    return marker_path(out_dir).exists()


def mark_done(out_dir: Path | str, meta: dict[str, Any] | None = None) -> None:
    p = marker_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, meta or {"status": "ok"})


# --------------------------------------------------------------------------- #
# Atomic JSON / JSONL / text
# --------------------------------------------------------------------------- #
def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_jsonl(path: Path | str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path | str, row: dict) -> None:
    """Append a single row. Used for streaming results; not atomic by design."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Provenance: every result file should be able to say which model revision and
# code commit produced it. Tagged onto outputs so a slide number is traceable.
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def provenance(**extra: Any) -> dict[str, Any]:
    import datetime

    meta = {
        "code_commit": git_commit(),
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    meta.update(extra)
    return meta
