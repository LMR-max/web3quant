import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class SampleResult:
    path: str
    objects_sampled: int
    keys: List[str]
    min_timestamp: Optional[int]
    max_timestamp: Optional[int]


def _iter_first_json_objects_from_array_file(
    path: str,
    max_objects: int = 50,
    max_bytes: int = 2_000_000,
    chunk_size: int = 256_000,
    encoding: str = "utf-8",
) -> Iterable[Dict[str, Any]]:
    """Stream-ish parser for JSON files of the form: [ {...}, {...}, ... ]

    It reads at most `max_bytes` and yields at most `max_objects` dict objects.
    Avoids loading the whole file into memory.
    """

    def flush_one(obj_text: str) -> Optional[Dict[str, Any]]:
        obj_text = obj_text.strip()
        if not obj_text:
            return None
        # tolerate trailing commas
        if obj_text.endswith(","):
            obj_text = obj_text[:-1]
        return json.loads(obj_text)

    bytes_read = 0
    in_string = False
    escape = False
    depth = 0
    started = False  # have we seen '['
    capturing = False
    buf_chars: List[str] = []

    with open(path, "rb") as f:
        while bytes_read < max_bytes:
            chunk = f.read(min(chunk_size, max_bytes - bytes_read))
            if not chunk:
                break
            bytes_read += len(chunk)

            text = chunk.decode(encoding, errors="ignore")
            for ch in text:
                if not started:
                    if ch == "[":
                        started = True
                    continue

                # skip whitespace and commas between elements when not capturing
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
                    else:
                        # unexpected token; ignore
                        continue
                    continue

                # capturing object
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

                # not in string
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
                        if isinstance(obj, dict):
                            yield obj
                            max_objects -= 1
                            if max_objects <= 0:
                                return
                        capturing = False
                        buf_chars = []


def sample_schema(path: str, max_objects: int = 50) -> SampleResult:
    keys: Set[str] = set()
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None
    sampled = 0

    for obj in _iter_first_json_objects_from_array_file(path, max_objects=max_objects):
        sampled += 1
        keys.update(obj.keys())
        ts = obj.get("timestamp")
        if isinstance(ts, int):
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)

    return SampleResult(
        path=path,
        objects_sampled=sampled,
        keys=sorted(keys),
        min_timestamp=min_ts,
        max_timestamp=max_ts,
    )


def find_merged_files(root: str) -> List[str]:
    merged: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_merged.json"):
                merged.append(os.path.join(dirpath, fn))
    merged.sort()
    return merged


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(repo_root, "data_manager_storage")

    merged_files = find_merged_files(data_root)
    if not merged_files:
        print(f"No *_merged.json files found under: {data_root}")
        return

    print(f"Found {len(merged_files)} merged files. Sampling first 50 objects each...\n")

    union_keys: Set[str] = set()
    for path in merged_files:
        res = sample_schema(path, max_objects=50)
        union_keys.update(res.keys)
        rel = os.path.relpath(path, repo_root)
        print(f"- {rel}")
        print(f"  sampled={res.objects_sampled} keys={res.keys}")
        if res.min_timestamp is not None:
            print(f"  ts_range=[{res.min_timestamp}, {res.max_timestamp}]")
        print("")

    print("Union keys across sampled files:")
    print(sorted(union_keys))


if __name__ == "__main__":
    main()
