"""Load datasets that env/download_assets.py saved to the local assets dir.

Falls back to a live HF load in dev (when nothing was pre-downloaded) so the
scripts are runnable on a laptop for a smoke check.
"""

from __future__ import annotations

from atax import config


def load_local(key: str):
    """Return a datasets.Dataset for a key in config.DATASETS."""
    from datasets import load_from_disk, load_dataset

    spec = config.DATASETS[key]
    dest = config.ASSETS_DIR / "datasets" / key / "data"
    if dest.exists():
        return load_from_disk(str(dest))
    # Dev fallback: fetch live.
    return load_dataset(spec.repo, spec.config, split=spec.split, revision=spec.revision)
