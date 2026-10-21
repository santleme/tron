#!/usr/bin/env python3
"""Verify every byte-preserved file in the CIRCUIT AI history export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/circuit-ai/history/MIGRATION_MANIFEST.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest.get("file_count"):
        raise SystemExit("history manifest file count does not match")
    total_bytes = 0
    for item in files:
        path = ROOT / item["path"]
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing or unsafe archived file: {item['path']}")
        size = path.stat().st_size
        if size != item["size_bytes"]:
            raise SystemExit(f"size mismatch: {item['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {item['path']}")
        total_bytes += size
    if total_bytes != manifest.get("total_bytes"):
        raise SystemExit("history manifest byte count does not match")
    print(f"verified {len(files)} files ({total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
