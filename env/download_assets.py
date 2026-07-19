#!/usr/bin/env python
"""Phase-0 asset prefetch: pull every model + dataset to local shared storage.

Why a dedicated step: on bare hardware you want all network I/O to happen ONCE,
up front, deterministically, and pinned -- not lazily during a GPU job where a
download stall wastes H100 hours or a floating `main` silently changes a number.

Idempotent: a model/dataset already present (with a _DONE marker) is skipped.
Run:  python env/download_assets.py            # everything
      python env/download_assets.py --models   # just models
      python env/download_assets.py --smoke    # primary ladder + tiny datasets
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Allow running before `atax.pth` is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from atax import config  # noqa: E402
from atax.io_utils import is_done, mark_done, write_json  # noqa: E402


def _model_dir(repo: str) -> Path:
    return config.ASSETS_DIR / "models" / repo.replace("/", "__")


def _warn_gated(repo: str) -> None:
    print(
        f"[dl] GATED model {repo}: access is restricted on HuggingFace.\n"
        f"     Treating it as OPTIONAL -- the run proceeds without it and only the\n"
        f"     units that need this exact model are skipped (logged by the scheduler).\n"
        f"     To include it, either:\n"
        f"       (a) request access at https://huggingface.co/{repo} then, on every machine you run this on,\n"
        f"           export HF_TOKEN=hf_xxx   (huggingface_hub reads it automatically), or\n"
        f"       (b) point that stage at an ungated mirror in src/atax/config.py."
    )


def download_model(repo: str, revision: str | None) -> str:
    """Return 'ok' | 'gated' | 'fail'. 'gated' is a non-fatal, optional skip."""
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.errors import GatedRepoError
    except Exception:  # older/newer hub: fall back to message sniffing below
        GatedRepoError = ()  # type: ignore[assignment]

    dest = _model_dir(repo)
    if is_done(dest):
        print(f"[dl] skip model {repo} (done)")
        return "ok"
    print(f"[dl] model {repo}@{revision or 'main'} -> {dest}")
    try:
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(dest),
            ignore_patterns=["*.pt", "*.pth", "*.bin.index.json.bak", "original/*"],
        )
        # Record exactly what we pinned.
        write_json(dest / "_PIN.json", {"repo": repo, "revision": revision or "main"})
        mark_done(dest, {"repo": repo, "revision": revision or "main"})
        return "ok"
    except GatedRepoError:
        _warn_gated(repo)
        return "gated"
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "gated" in msg or "restricted" in msg or "awaiting a review" in msg:
            _warn_gated(repo)
            return "gated"
        print(f"[dl] FAILED model {repo}\n{traceback.format_exc()}")
        return "fail"


def _dataset_dir(key: str) -> Path:
    return config.ASSETS_DIR / "datasets" / key


def download_dataset(spec: config.DatasetSpec) -> bool:
    from datasets import load_dataset

    dest = _dataset_dir(spec.key)
    if is_done(dest):
        print(f"[dl] skip dataset {spec.key} (done)")
        return True
    print(f"[dl] dataset {spec.key} ({spec.repo} cfg={spec.config} split={spec.split})")
    try:
        # Materialise to an arrow cache under our assets dir so every run reads
        # the same files. We save to disk for a stable, offline-loadable copy.
        ds = load_dataset(
            spec.repo,
            spec.config,
            split=spec.split,
            revision=spec.revision,
        )
        dest.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(dest / "data"))
        mark_done(dest, {"repo": spec.repo, "config": spec.config, "split": spec.split,
                         "num_rows": ds.num_rows})
        return True
    except Exception:
        print(f"[dl] FAILED dataset {spec.key}\n{traceback.format_exc()}")
        return False


def download_capability_tasks() -> bool:
    """Pre-fetch the lm-eval (D4 capability) task datasets into the STANDARD
    HuggingFace cache.

    WHY a dedicated step (and not just config.DATASETS): eval/run_capability.py
    scores via lm-evaluation-harness, which loads each task's data with its OWN
    datasets.load_dataset(...) call into the STANDARD HF cache (~/.cache/
    huggingface) -- NOT the assets/datasets/ dir that download_dataset() +
    load_local() use. So MMLU (cais/mmlu, 57 subject configs), HellaSwag and
    TruthfulQA-mc2 are never materialised by the dataset loop above. run_all.sh
    then forces HF_HUB_OFFLINE=1 for the GPU phases (to dodge HuggingFace's API
    rate limit when many parallel runs hit it at once), so the capability units cannot reach the Hub and
    die with "Couldn't find cache for cais/mmlu for config 'prehistory'" (offline,
    partial cache) or "Couldn't reach ... on the Hub" (no network).

    We trigger lm-eval's OWN task loading here, while still ONLINE (this runs in
    run_all.sh BEFORE the offline switch in step 1b), so the cache the eval reads
    is warm before that run starts. Using lm-eval's loader -- rather than a hand-maintained
    repo->task list -- guarantees the cache key is identical to what
    run_capability.py later requests, and it automatically materialises every
    subject of a task GROUP like 'mmlu'.

    Best-effort: a failure here is a loud WARNING, not a hard error. Capability is
    one of five probes and each unit fails in isolation; the rest of the run still
    proceeds.
    """
    import hashlib
    import os
    import sys

    tasks = sorted(set(config.CAPABILITY_TASKS) | set(config.SWEEP_EVAL_TASKS))
    # Prefetch the opt-in commonsense battery too when the domain-SFT OOD run is
    # requested (ATAX_COMMONSENSE=1), so the offline GPU phase finds them in cache.
    if os.environ.get("ATAX_COMMONSENSE", "").strip() not in ("", "0", "false", "False"):
        tasks = sorted(set(tasks) | set(config.COMMONSENSE_TASKS))

    # The DONE marker is keyed by (venv interpreter + EXACT task set), NOT a fixed
    # name. WHY (two bugs this fixes): (1) lm-eval's task->HF-repo mapping differs
    # across versions -- the pinned .venv lm-eval resolves `winogrande` to a different
    # repo than .venv-next's lm-eval>=0.4.8 (which uses `allenai/winogrande`) -- so a
    # warm done under one venv must NOT satisfy a run under the other; (2) the
    # commonsense battery is opt-in, so a warm done WITHOUT ATAX_COMMONSENSE (smaller
    # task set) must NOT satisfy a later run WITH it. A single fixed marker did both:
    # it let a stale/mismatched warm mark this step "done", the offline GPU phase then
    # hit an uncached repo and died with `OfflineModeIsEnabled` on allenai/winogrande.
    # Hashing sys.executable + the sorted task list gives each (venv, task-set) its own
    # marker, so each distinct combination warms exactly once and never masks another.
    _tag = hashlib.md5((sys.executable + "|" + ",".join(tasks)).encode()).hexdigest()[:10]
    marker = config.ASSETS_DIR / "datasets" / f"_lm_eval_capability_{_tag}"
    if is_done(marker):
        print(f"[dl] skip lm-eval capability tasks (done; venv+tasks tag {_tag})")
        return True

    print(f"[dl] warming lm-eval capability datasets into the HF cache: {tasks}")
    try:
        import lm_eval  # noqa: F401  (heavy import, only needed for this step)
        from lm_eval.tasks import TaskManager
        try:
            from lm_eval.tasks import get_task_dict
        except Exception:  # pragma: no cover - public API moved in some versions
            get_task_dict = None

        tm = TaskManager()
        # Instantiating each task runs ConfigurableTask.download() ->
        # datasets.load_dataset(...), which fills the standard HF cache. For a
        # GROUP like 'mmlu' this materialises all 57 subject configs. get_task_dict
        # is the stable public entry across 0.4.x; TaskManager.load is the newer
        # (0.4.8+) equivalent shown in the failing traceback.
        if get_task_dict is not None:
            get_task_dict(tasks, tm)
        else:
            tm.load(tasks)

        marker.mkdir(parents=True, exist_ok=True)
        mark_done(marker, {"tasks": tasks})
        print(f"[dl] capability datasets warmed ({len(tasks)} tasks)")
        return True
    except Exception:
        print(
            "[dl] WARNING: could not pre-fetch the lm-eval capability datasets "
            "(MMLU / HellaSwag / TruthfulQA-mc2).\n"
            "     The D4 capability units will FAIL under the offline GPU phase "
            "until these are cached. Fix options:\n"
            "       (a) re-run this download step WITH network access (it warms "
            "the cache, then the offline eval finds it), or\n"
            "       (b) run the GPU phases online: ATAX_HF_OFFLINE=0 bash run_all.sh\n"
            f"{traceback.format_exc()}"
        )
        return False


def download_domain_corpora() -> bool:
    """Warm the HF cache for the DOMAIN-SFT corpora (opt-in: ATAX_DOMAINS).

    The domain-SFT trainer (train/sft_domain.py) builds its text split at run time
    via atax.domain_data.load_domain_texts -> datasets.load_dataset(...). Under the
    forced-offline GPU phase that load must hit a WARM cache, so we trigger the SAME
    loader here (online). Calling the real loader -- not a hand-copied load_dataset
    -- guarantees an identical cache key AND validates each formatter against the
    live schema before any GPU time is spent (a bad field name fails HERE, cheaply).

    Best-effort per corpus: a gated/failed corpus is a loud WARNING (that domain's
    units are skipped at run time), never a hard abort.
    """
    sel = _domain_allowlist()
    if sel is None:
        return True  # ATAX_DOMAINS unset -> domain-SFT not requested; nothing to do
    from atax.domain_data import load_domain_texts
    ok = True
    for key, corpus in config.DOMAIN_CORPORA.items():
        if key not in sel:
            continue
        marker = config.ASSETS_DIR / "datasets" / f"_domain_{key}"
        if is_done(marker):
            print(f"[dl] skip domain corpus {key} (done)")
            continue
        print(f"[dl] warming domain corpus {key} ({corpus.repo} cfg={corpus.config} "
              f"split={corpus.split}){' [GATED]' if corpus.gated else ''}")
        try:
            texts = load_domain_texts(key, config.DOMAIN_SFT_TRAIN_SAMPLES, seed=0)
            marker.mkdir(parents=True, exist_ok=True)
            mark_done(marker, {"repo": corpus.repo, "config": corpus.config,
                               "split": corpus.split, "n_texts": len(texts)})
            print(f"[dl] domain corpus {key} warmed ({len(texts)} texts)")
        except Exception:
            ok = False
            if corpus.gated:
                _warn_gated(corpus.repo)
            else:
                print(f"[dl] WARNING: domain corpus {key} prefetch FAILED; its "
                      f"domain-SFT units will be skipped.\n{traceback.format_exc()}")
    return ok


def check_access() -> int:
    """Metadata-only access preflight over EVERY model the FULL run will load.

    Does NOT download. Catches auth problems (gated repos, wrong ids) that the
    smoke download -- which only fetches the primary ladder -- would never hit,
    so `smoke_test.sh` can surface them BEFORE the real launch. Informational:
    returns 0 even when some models are gated, since gated models are an optional
    skip at run time (only the units needing them are dropped).
    """
    from huggingface_hub import HfApi
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except Exception:  # pragma: no cover - version drift
        GatedRepoError = RepositoryNotFoundError = ()  # type: ignore[assignment,misc]

    api = HfApi()
    only = _ladder_allowlist()
    seen: set[str] = set()
    repos: list[tuple[str, str | None]] = []
    for _key, stage in config.all_stage_models():
        if only is not None and _key not in only:
            continue
        if only is None and _key in config.DEFAULT_EXCLUDED_LADDER_KEYS:
            continue  # opt-in ladders (nextgen + scale study) skip by default
        if stage.repo not in seen:
            seen.add(stage.repo)
            repos.append((stage.repo, stage.revision))

    reachable: list[str] = []
    gated: list[str] = []
    missing: list[str] = []
    other: list[str] = []
    for repo, rev in repos:
        try:
            api.model_info(repo, revision=rev, timeout=20)
            reachable.append(repo)
        except Exception as e:  # noqa: BLE001 - classify by type/message
            msg = str(e).lower()
            if isinstance(e, GatedRepoError) or "gated" in msg or "401" in msg \
                    or "restricted" in msg or "awaiting a review" in msg:
                gated.append(repo)
            elif isinstance(e, RepositoryNotFoundError) or "404" in msg or "not found" in msg:
                missing.append(repo)
            else:
                other.append(f"{repo} ({type(e).__name__})")

    print(f"[access] preflight over {len(repos)} models "
          f"(HF_TOKEN {'set' if os.environ.get('HF_TOKEN') else 'NOT set'}):")
    print(f"[access]   reachable: {len(reachable)}/{len(repos)}")
    if gated:
        print(f"[access]   GATED ({len(gated)}): {', '.join(gated)}")
        print("[access]     -> OPTIONAL: run proceeds, only units needing them are")
        print("[access]        skipped. Set HF_TOKEN on every machine you run this on, or use a mirror.")
    if missing:
        print(f"[access]   MISSING/404 ({len(missing)}): {', '.join(missing)}")
        print("[access]     -> repo id likely wrong in config.py; THIS loses units.")
    if other:
        print(f"[access]   OTHER errors: {', '.join(other)}")
    if not (gated or missing or other):
        print("[access]   all models reachable.")
    return 0


def _ladder_allowlist() -> set[str] | None:
    """Optional ATAX_LADDERS allow-list (comma-separated ladder keys) so a v2 run
    only fetches/checks a SUBSET of families. None => all (default). Unknown keys
    abort loudly rather than silently doing nothing."""
    raw = os.environ.get("ATAX_LADDERS", "").strip()
    if not raw:
        return None
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = keys - set(config.LADDERS)
    if unknown:
        raise SystemExit(
            f"[dl] ATAX_LADDERS has unknown ladder(s): {sorted(unknown)}; "
            f"known: {sorted(config.LADDERS)}"
        )
    return keys


def _d2_requested_datasets() -> set[str]:
    """D2 sources named in ATAX_D2_DATASETS (empty if unset). Mirrors
    build_manifests._d2_datasets so the downloader fetches exactly the opt-in D2
    sources a run will build units for (mmlu/medqa); arc_challenge is unaffected."""
    raw = os.environ.get("ATAX_D2_DATASETS", "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _domain_allowlist() -> set[str] | None:
    """ATAX_DOMAINS allow-list (comma-sep domain keys, or 'all') selecting which
    DOMAIN-SFT corpora to prefetch. None => domain-SFT NOT requested (default; zero
    domain prefetch). Same semantics as build_manifests._domain_filter. Unknown
    keys abort loudly."""
    raw = os.environ.get("ATAX_DOMAINS", "").strip()
    if not raw:
        return None
    if raw.lower() == "all":
        return set(config.DOMAIN_CORPORA)
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = keys - set(config.DOMAIN_CORPORA)
    if unknown:
        raise SystemExit(
            f"[dl] ATAX_DOMAINS has unknown domain(s): {sorted(unknown)}; "
            f"known: {sorted(config.DOMAIN_CORPORA)}"
        )
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="only models")
    ap.add_argument("--datasets", action="store_true", help="only datasets")
    ap.add_argument("--smoke", action="store_true", help="primary ladder + small datasets only")
    ap.add_argument("--check-access", action="store_true",
                    help="metadata-only auth preflight over EVERY model in the full run; no download")
    args = ap.parse_args()

    if args.check_access:
        return check_access()

    do_models = args.models or not args.datasets
    do_datasets = args.datasets or not args.models

    hard_fail = 0          # real failures (network/disk/etc) -> non-zero exit
    gated: list[str] = []  # access-restricted, optional -> warning only

    if do_models:
        if args.smoke:
            models = [(config.PRIMARY_LADDER, s) for s in config.LADDERS[config.PRIMARY_LADDER].stages]
        else:
            models = config.all_stage_models()
            only = _ladder_allowlist()
            if only is not None:
                models = [(k, s) for (k, s) in models if k in only]
            else:
                # default download stays byte-identical -> exclude ALL opt-in
                # ladders (nextgen need requirements-next.txt; the scale/generation
                # study is pinned-stack but opt-in). Fetch via ATAX_LADDERS.
                models = [(k, s) for (k, s) in models
                          if k not in config.DEFAULT_EXCLUDED_LADDER_KEYS]
        # de-dup repos (sweep base overlaps the ladder base)
        seen = set()
        for _key, stage in models:
            if stage.repo in seen:
                continue
            seen.add(stage.repo)
            st = download_model(stage.repo, stage.revision)
            if st == "gated":
                gated.append(stage.repo)
            elif st == "fail":
                hard_fail += 1
        # Domain-SFT bases (opt-in via ATAX_DOMAINS): ensure each base we will
        # fine-tune is present even if its ladder was not named in ATAX_LADDERS
        # (e.g. qwen35_9b's base lives in a nextgen ladder excluded by default).
        if _domain_allowlist() is not None:
            for _bk, _bstage in config.DOMAIN_BASES.items():
                if _bstage.repo in seen:
                    continue
                seen.add(_bstage.repo)
                st = download_model(_bstage.repo, _bstage.revision)
                if st == "gated":
                    gated.append(_bstage.repo)
                elif st == "fail":
                    hard_fail += 1

    if do_datasets:
        keys = list(config.DATASETS)
        if args.smoke:
            # popqa+truthfulqa drive the smoke D1/cap checks; tulu3_sft_olmo is
            # required by the smoke SFT's mixture builder (otherwise it silently
            # live-downloads the full multi-GB mixture inside the GPU job).
            keys = ["popqa", "truthfulqa_mc", "tulu3_sft_olmo"]
        else:
            # Opt-in D2 sources (mmlu/medqa) download ONLY when ATAX_D2_DATASETS
            # names them, so a default pull stays byte-identical. arc_challenge is
            # NOT in the extra set (default D4 dataset) and is always fetched.
            req = _d2_requested_datasets()
            keys = [k for k in keys
                    if k not in config.D2_EXTRA_DATASET_KEYS or k in req]
        for k in keys:
            if not download_dataset(config.DATASETS[k]):
                hard_fail += 1
        # Warm the lm-eval capability-task cache (MMLU 57 subjects, HellaSwag,
        # TruthfulQA-mc2) so the forced-offline GPU phase can score D4. Skipped in
        # smoke: smoke runs capability ONLINE with --limit and lazily fetches them.
        # Best-effort (loud WARNING on failure, never aborts the run).
        if not args.smoke:
            download_capability_tasks()
            # Domain-SFT corpora (opt-in via ATAX_DOMAINS): warm the HF cache so the
            # forced-offline train phase finds them. Best-effort (a gated code
            # corpus => warning, that domain skipped; medical/legal still run).
            download_domain_corpora()

    if gated:
        print(f"[dl] SKIPPED {len(gated)} gated/optional model(s): {', '.join(gated)}")
    if hard_fail:
        print(f"[dl] {hard_fail} HARD failure(s) — see log above")
        return 1
    print("[dl] OK" + (" (gated models skipped; run will proceed without them)" if gated else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
