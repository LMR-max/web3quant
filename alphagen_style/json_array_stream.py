from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional


def iter_json_objects_from_array_file(
    path: str,
    *,
    max_objects: Optional[int] = None,
    max_bytes: Optional[int] = None,
    chunk_size: int = 256_000,
    encoding: str = "utf-8",
) -> Iterator[Dict[str, Any]]:
    """Iterate objects from a JSON file shaped like: [ {...}, {...}, ... ].

    This avoids loading the whole file into memory.

    Notes:
    - It is *not* a general-purpose JSON parser.
    - It assumes top-level is an array of objects.
    """

    def flush_one(obj_text: str) -> Optional[Dict[str, Any]]:
        obj_text = obj_text.strip()
        if not obj_text:
            return None
        if obj_text.endswith(","):
            obj_text = obj_text[:-1]
        obj = json.loads(obj_text)
        return obj if isinstance(obj, dict) else None

    bytes_read = 0
    in_string = False
    escape = False
    depth = 0
    started = False  # have we seen '['
    capturing = False
    buf_chars: List[str] = []

    with open(path, "rb") as f:
        while True:
            if max_bytes is not None and bytes_read >= max_bytes:
                break

            read_n = chunk_size
            if max_bytes is not None:
                read_n = min(read_n, max_bytes - bytes_read)

            chunk = f.read(read_n)
            if not chunk:
                break
            bytes_read += len(chunk)

            text = chunk.decode(encoding, errors="ignore")
            for ch in text:
                if not started:
                    if ch == "[":
                        started = True
                    continue

                if not capturing:
                    if ch in " \t\r\n,":
                        continue
                    if ch == "]":
                        return
                    if ch == "{":
                        capturing = True
                        depth = 1
                        in_string = False
                        escape = False
                        buf_chars = [ch]
                    continue

                buf_chars.append(ch)

                if in_string:
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == "{":
                    depth += 1
                    continue

                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        obj = flush_one("".join(buf_chars))
                        buf_chars = []
                        capturing = False

                        if obj is not None:
                            yield obj
                            if max_objects is not None:
                                max_objects -= 1
                                if max_objects <= 0:
                                    return
