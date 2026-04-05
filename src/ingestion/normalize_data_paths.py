"""Rewrite image_path fields in JSON to project-relative POSIX paths.

Run once after cloning on a machine that has the image files:

    python -m src.ingestion.normalize_data_paths

Optional: point at another project root (e.g. Kaggle extract):

    ML_PROJECT_ROOT=/path/to/root python -m src.ingestion.normalize_data_paths
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.default import (  # noqa: E402
    DATA_EVAL,
    DATA_PROCESSED,
    DATA_RAW,
    path_for_storage,
    resolve_data_path,
)


def _normalize_file(json_path: Path, path_keys: tuple[str, ...] = ("image_path",)) -> int:
    if not json_path.exists():
        return 0
    with open(json_path) as f:
        data = json.load(f)
    changed = 0

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            for key in path_keys:
                v = item.get(key)
                if not isinstance(v, str) or not v.strip():
                    continue
                resolved = resolve_data_path(v)
                if resolved is not None and resolved.is_file():
                    new_v = path_for_storage(resolved)
                    if new_v != v:
                        item[key] = new_v
                        changed += 1
    if changed:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {json_path}: updated {changed} path(s)")
    else:
        print(f"  {json_path}: no changes")
    return changed


def main() -> None:
    targets: list[tuple[Path, tuple[str, ...]]] = [
        (DATA_RAW / "scienceqa" / "physics_samples.json", ("image_path",)),
        (DATA_RAW / "ai2d" / "ai2d_samples.json", ("image_path",)),
        (DATA_EVAL / "eval_dataset.json", ("image_path",)),
        (DATA_EVAL / "synthetic_physics_qa.json", ("image_path",)),
    ]
    for p in DATA_PROCESSED.rglob("images.json"):
        targets.append((p, ("image_path",)))
    finetune_dir = DATA_PROCESSED / "finetune"
    if finetune_dir.is_dir():
        for split in ("train", "val", "test"):
            targets.append((finetune_dir / f"{split}.json", ("image",)))

    total = 0
    print("Normalizing stored paths to project-relative form...")
    for path, keys in targets:
        total += _normalize_file(path, keys)
    print(f"Done. Total path fields updated: {total}")


if __name__ == "__main__":
    main()
