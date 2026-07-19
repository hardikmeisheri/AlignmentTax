#!/usr/bin/env python
"""CLI wrapper to build + inspect a rarity mixture (debugging aid).

Normally the trainer builds the mixture in-process; this is for eyeballing it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atax import config
from atax.mixture import build_mixture, save_mixture


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.SWEEP_PRIMARY_BASE)
    ap.add_argument("--signal", default="benign", choices=config.SWEEP_SIGNALS)
    ap.add_argument("--freq", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mix = build_mixture(args.base, args.signal, args.freq, args.seed, args.total)
    save_mixture(mix, Path(args.out))
    print("[mixture]", mix["meta"])
    print("[mixture] sample train example:\n", mix["train"][0][:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
