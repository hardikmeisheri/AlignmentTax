"""The island scheduler: fire one manifest across all local GPUs, no NCCL.

Mental model
------------
The whole experiment is a bag of independent *units*. A unit is one command that
needs either 1 GPU (inference / eval / analysis) or a whole machine's GPUs
(single-node training). Units never talk to each other. There is therefore no
collective communication anywhere in the inference path -- the #1 source of
multi-GPU debugging pain is simply absent.

Parallelism model
-----------------
  * Within one run: a pool of the local GPUs. We pop a unit, reserve the GPUs it
    needs, launch it as a subprocess with CUDA_VISIBLE_DEVICES pinned to exactly
    those GPUs, and move on. When it exits we free the GPUs.
  * Across parallel passes (optional): there is NO cross-pass scheduler. Each
    pass runs this same program over the SAME manifest but keeps only the shard
    `hash(unit.id) %% num_shards == shard_rank`. Passes share a filesystem for
    results but never coordinate -- each shard is an independent island.

Robustness
----------
  * Idempotent / resumable: a unit whose output dir already has a _DONE marker is
    skipped. Re-running run_all.sh after a crash resumes where it stopped.
  * Failure isolation: a unit that exits non-zero is logged and abandoned; the
    rest of the run continues. One bad model never sinks the batch.

This file intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Unit definition
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    id: str
    kind: str                 # "infer" | "train" | "cpu"
    cmd: list[str]
    out_dir: str
    ngpu: int = 1             # GPUs required; train units pass the machine's gpu count
    env: dict[str, str] = field(default_factory=dict)
    shard_key: str = ""       # cross-shard co-location key; "" -> shard by id
    shard_index: int = -1     # >=0 -> balanced round-robin (index % num_shards)

    @staticmethod
    def from_dict(d: dict) -> "Unit":
        return Unit(
            id=d["id"],
            kind=d.get("kind", "infer"),
            cmd=list(d["cmd"]),
            out_dir=d["out_dir"],
            ngpu=int(d.get("ngpu", 1)),
            env=dict(d.get("env", {})),
            shard_key=str(d.get("shard_key") or d["id"]),
            shard_index=int(d.get("shard_index", -1)),
        )


# --------------------------------------------------------------------------- #
# GPU discovery
# --------------------------------------------------------------------------- #
def discover_gpus() -> list[int]:
    """Physical GPU indices visible to this process.

    Honors a pre-set CUDA_VISIBLE_DEVICES (so a machine's GPUs can be carved up),
    else asks nvidia-smi, else falls back to a CPU-only single slot.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [int(x) for x in cvd.split(",") if x.strip() != ""]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode()
        n = len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])
        return list(range(n))
    except Exception:
        return []  # CPU-only (smoke/dev box)


# --------------------------------------------------------------------------- #
# Manifest sharding (across parallel passes, coordination-free)
# --------------------------------------------------------------------------- #
def shard_units(units: list[Unit], node_rank: int, num_nodes: int) -> list[Unit]:
    if num_nodes <= 1:
        return units
    keep = []
    for u in units:
        # Two placement modes, both deterministic so every pass computes the SAME
        # split from the same manifest with no coordination:
        #   * shard_index >= 0 : balanced round-robin (index % num_shards). Used
        #     for the sweep so the full-finetunes spread evenly AND a
        #     checkpoint's train+eval share an index => same shard (local-disk
        #     safe). crc32(tag) alone co-locates but clumps load badly.
        #   * else crc32(shard_key or id) : hash split for everything else.
        if u.shard_index >= 0:
            take = (u.shard_index % num_nodes == node_rank)
        else:
            key = u.shard_key or u.id
            take = (zlib.crc32(key.encode()) % num_nodes == node_rank)
        if take:
            keep.append(u)
    return keep


# --------------------------------------------------------------------------- #
# Running process bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class Running:
    unit: Unit
    proc: subprocess.Popen
    gpus: list[int]
    log_fh: object
    started: float


def _log_path(log_dir: Path, unit: Unit) -> Path:
    safe = unit.id.replace("/", "_")
    return log_dir / f"{safe}.log"


def run_manifest(
    units: list[Unit],
    gpus: list[int],
    log_dir: Path,
    poll_s: float = 2.0,
    dry_run: bool = False,
) -> int:
    """Greedy bin-packing of units onto the local GPU pool. Returns #failures."""
    from atax.io_utils import is_done, mark_done  # local import: keep module importable

    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-finished units up front.
    pending: list[Unit] = []
    skipped = 0
    for u in units:
        if is_done(u.out_dir):
            skipped += 1
        else:
            pending.append(u)
    print(f"[sched] {len(units)} units | {skipped} already done | {len(pending)} to run "
          f"| gpus={gpus or 'CPU-only'}", flush=True)

    free = list(gpus)
    running: list[Running] = []
    failures: list[str] = []
    done = 0
    total = len(pending)

    def can_start(u: Unit) -> bool:
        need = u.ngpu if gpus else 0
        return len(free) >= need

    # Sort train (whole-machine) units first so they are not starved by a full
    # pool of 1-GPU jobs. Within a kind, stable order.
    pending.sort(key=lambda u: (0 if u.kind == "train" else 1, u.id))

    while pending or running:
        # Launch as many as fit.
        i = 0
        while i < len(pending):
            u = pending[i]
            if can_start(u):
                need = u.ngpu if gpus else 0
                assigned = [free.pop(0) for _ in range(need)]
                if dry_run:
                    print(f"[sched] DRY would run {u.id} on gpus={assigned}: {' '.join(u.cmd)}")
                    for g in assigned:
                        free.append(g)
                    pending.pop(i)
                    done += 1
                    continue
                env = dict(os.environ)
                env.update(u.env)
                if gpus:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in assigned)
                env["ATAX_UNIT_ID"] = u.id
                env["ATAX_OUT_DIR"] = u.out_dir
                Path(u.out_dir).mkdir(parents=True, exist_ok=True)
                lp = _log_path(log_dir, u)
                fh = open(lp, "w", encoding="utf-8")
                fh.write(f"# unit={u.id} gpus={assigned} cmd={u.cmd}\n")
                fh.flush()
                proc = subprocess.Popen(u.cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                running.append(Running(u, proc, assigned, fh, time.time()))
                pending.pop(i)
                print(f"[sched] start {u.id} on gpus={assigned} "
                      f"({len(running)} running, {len(pending)} queued)", flush=True)
            else:
                i += 1

        # Reap finished.
        time.sleep(poll_s)
        still: list[Running] = []
        for r in running:
            rc = r.proc.poll()
            if rc is None:
                still.append(r)
                continue
            dt = time.time() - r.started
            r.log_fh.flush()
            r.log_fh.close()
            for g in r.gpus:
                free.append(g)
            if rc == 0:
                mark_done(r.unit.out_dir, {"unit": r.unit.id, "seconds": round(dt, 1)})
                done += 1
                print(f"[sched] OK   {r.unit.id} ({dt:.0f}s) [{done}/{total}]", flush=True)
            else:
                failures.append(r.unit.id)
                print(f"[sched] FAIL {r.unit.id} rc={rc} ({dt:.0f}s) -> {_log_path(log_dir, r.unit)}",
                      flush=True)
        running = still

    print(f"[sched] finished: {done} ok, {len(failures)} failed", flush=True)
    if failures:
        print("[sched] failures:\n  " + "\n  ".join(failures), flush=True)
    return len(failures)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Fire a unit manifest across local GPUs (no NCCL).")
    ap.add_argument("manifest", help="JSON file: list of unit dicts")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--node-rank", type=int, default=int(os.environ.get("ATAX_NODE_RANK", "0")))
    ap.add_argument("--num-nodes", type=int, default=int(os.environ.get("ATAX_NUM_NODES", "1")))
    ap.add_argument("--gpus", default=None, help="comma list to override discovery")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        raw = json.load(f)
    units = [Unit.from_dict(d) for d in raw]
    units = shard_units(units, args.node_rank, args.num_nodes)

    gpus = ([int(x) for x in args.gpus.split(",")] if args.gpus else discover_gpus())

    log_dir = Path(args.log_dir) if args.log_dir else Path(args.manifest).with_suffix("").parent / "logs"
    n_fail = run_manifest(units, gpus, log_dir, dry_run=args.dry_run)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
