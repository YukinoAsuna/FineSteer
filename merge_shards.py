from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic run_truthfulqa JSONL shards by test index.")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=409)
    args = parser.parse_args()
    rows = []
    for path in args.shards:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows.sort(key=lambda row: int(row["index"]))
    indexes = [int(row["index"]) for row in rows]
    if len(indexes) != len(set(indexes)):
        raise SystemExit("duplicate indexes across shards")
    if args.expected and indexes != list(range(args.expected)):
        missing = sorted(set(range(args.expected)) - set(indexes))
        raise SystemExit(f"expected indexes 0..{args.expected - 1}; missing={missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"merged {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
