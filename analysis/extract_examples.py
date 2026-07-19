#!/usr/bin/env python
"""Extract REAL, verbatim transcripts from Track-1 outputs into deck-ready examples.

NOTHING here is synthesised. Every quoted prompt and completion is read verbatim
from the probe outputs

    <results>/track1/{d1,d2,d3}/<ladder>__<stage>/raw.jsonl

and every statistic is read from the matching dN.json summary. Each example is
tagged with its source path and the code commit that produced it, so a slide
number can be traced back to the exact bytes that justify it.

The script picks ONE illustrative item per probe (by a fixed, principled rule,
not by eyeballing for effect) to sit next to the real aggregate numbers. The
quantitative claim is always the aggregate; the transcript is the illustration.

Usage
-----
  # primary ladder (olmo2_7b), base vs instruct, reading local results/
  python analysis/extract_examples.py

  # a different ladder / stages
  python analysis/extract_examples.py --ladder qwen3_8b --aligned-stage instruct

  # read a results tree synced down from remote storage instead of the local one
  python analysis/extract_examples.py --results-root /path/to/pulled/results

Writes <out>/examples.json (structured) and <out>/examples.md (paste-ready).
Missing inputs are reported as status="MISSING" and never fabricated.
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any

from atax import config
from atax.io_utils import read_json, read_jsonl

LETTERS = string.ascii_uppercase
_MAX_QUOTE = 600  # hard cap on a single quoted completion; longer is truncated + flagged


# --------------------------------------------------------------------------- #
# Loading (defensive: a missing file is reported, never invented)
# --------------------------------------------------------------------------- #
def _unit_dir(root: Path, demo: str, ladder: str, stage: str) -> Path:
    return root / "track1" / demo / f"{ladder}__{stage}"


def _load_unit(root: Path, demo: str, ladder: str, stage: str) -> dict[str, Any]:
    """Return {dir, summary, raw, status} for one (demo, ladder, stage)."""
    d = _unit_dir(root, demo, ladder, stage)
    summary_path = d / f"{demo}.json"
    raw_path = d / "raw.jsonl"
    info: dict[str, Any] = {"dir": str(d), "summary": None, "raw": None, "status": "ok"}
    missing = []
    if summary_path.exists():
        info["summary"] = read_json(summary_path)
    else:
        missing.append(str(summary_path))
    if raw_path.exists():
        info["raw"] = read_jsonl(raw_path)
    else:
        missing.append(str(raw_path))
    if missing:
        info["status"] = "MISSING"
        info["missing"] = missing
    return info


def _commit(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "unknown"
    prov = summary.get("provenance", {})
    return f"{prov.get('code_commit', 'nogit')} @ {prov.get('utc', '?')}"


def _quote(text: str) -> dict[str, Any]:
    """Verbatim text, capped only to keep a slide sane; truncation is flagged."""
    t = text if text is not None else ""
    if len(t) > _MAX_QUOTE:
        return {"text": t[:_MAX_QUOTE], "truncated_from": len(t)}
    return {"text": t, "truncated_from": None}


# --------------------------------------------------------------------------- #
# D1 — creativity collapse
# --------------------------------------------------------------------------- #
def d1_example(root: Path, ladder: str, base: str, aligned: str,
               prompt_key: str, n_show: int) -> dict[str, Any]:
    out: dict[str, Any] = {"demo": "d1", "title": "Creativity collapse",
                           "prompt_key": prompt_key, "stages": {}}
    for label, stage in (("base", base), ("aligned", aligned)):
        u = _load_unit(root, "d1", ladder, stage)
        entry: dict[str, Any] = {"stage": stage, "status": u["status"],
                                 "dir": u["dir"], "commit": _commit(u["summary"])}
        if u["status"] != "ok":
            entry["missing"] = u.get("missing")
            out["stages"][label] = entry
            continue
        pp = u["summary"].get("per_prompt", {}).get(prompt_key)
        if pp is None:
            entry["status"] = "MISSING"
            entry["note"] = f"prompt_key {prompt_key!r} not in d1.json"
            out["stages"][label] = entry
            continue
        out["prompt"] = pp["prompt"]  # the exact text the model saw
        entry["top_mode_share"] = pp["top_mode_share"]
        entry["norm_entropy"] = pp["norm_entropy"]
        entry["norm_entropy_ci"] = pp.get("norm_entropy_ci")
        entry["n_valid"] = pp["n_valid"]
        entry["top5_answers"] = pp["top5_answers"]
        # First n_show completions for THIS prompt, in generation order (no
        # cherry-picking): vLLM preserves order and the seed is fixed.
        comps = [r for r in u["raw"] if r.get("prompt_key") == prompt_key][:n_show]
        entry["samples"] = [{"completion": _quote(r["completion"]),
                             "answer": r.get("answer")} for r in comps]
        out["stages"][label] = entry
    return out


# --------------------------------------------------------------------------- #
# D2 — sycophancy (find an item the base held but the aligned model caved on)
# --------------------------------------------------------------------------- #
def _letter(idx: int | None) -> str:
    return LETTERS[idx] if isinstance(idx, int) and 0 <= idx < len(LETTERS) else "?"


def _choice_text(row: dict[str, Any], idx: int | None) -> str | None:
    """Choice text if the probe logged it (newer runs); else None."""
    choices = row.get("choices")
    if isinstance(choices, list) and isinstance(idx, int) and 0 <= idx < len(choices):
        return choices[idx]
    return None


def _index_by_question(rows: list[dict], nudge: str) -> dict[str, dict]:
    return {r["question"]: r for r in rows if r.get("nudge") == nudge}


def d2_example(root: Path, ladder: str, base: str, aligned: str,
               nudge: str) -> dict[str, Any]:
    out: dict[str, Any] = {"demo": "d2", "title": "Sycophancy (caves under pressure)",
                           "nudge": nudge, "stages": {}}
    bu = _load_unit(root, "d2", ladder, base)
    au = _load_unit(root, "d2", ladder, aligned)
    out["stages"]["base"] = {"stage": base, "status": bu["status"], "dir": bu["dir"],
                             "commit": _commit(bu["summary"])}
    out["stages"]["aligned"] = {"stage": aligned, "status": au["status"], "dir": au["dir"],
                                "commit": _commit(au["summary"])}
    # Aggregate flip rates (the quantitative claim).
    for label, u in (("base", bu), ("aligned", au)):
        if u["status"] == "ok":
            pn = u["summary"].get("per_nudge", {}).get(nudge, {})
            out["stages"][label]["flip_rate"] = pn.get("flip_rate")
            out["stages"][label]["flip_rate_ci"] = pn.get("flip_rate_ci")
            out["stages"][label]["neutral_accuracy"] = u["summary"]["summary"].get("neutral_accuracy")
    if bu["status"] != "ok" or au["status"] != "ok":
        out["stages"]["base"]["missing"] = bu.get("missing")
        out["stages"]["aligned"]["missing"] = au.get("missing")
        return out

    b_nudge = _index_by_question(bu["raw"], nudge)
    a_nudge = _index_by_question(au["raw"], nudge)
    b_neutral = _index_by_question(bu["raw"], "neutral")
    a_neutral = _index_by_question(au["raw"], "neutral")
    shared = sorted(set(b_nudge) & set(a_nudge))

    # The clean story: base did NOT flip, aligned DID. Deterministic pick (first
    # in sorted order). Fall back honestly if no such item exists.
    clean = [q for q in shared
             if not b_nudge[q].get("flipped") and a_nudge[q].get("flipped")]
    note = None
    if clean:
        q = clean[0]
    elif shared:
        # No item where base holds AND aligned flips; pick any aligned flip and
        # report base's actual behaviour on the same item (do not pretend).
        flips = [q for q in shared if a_nudge[q].get("flipped")]
        q = flips[0] if flips else shared[0]
        note = ("No item found where the base held and the aligned model flipped; "
                "showing an aligned flip with the base's ACTUAL behaviour on the "
                "same item (base may also have flipped).")
    else:
        out["item"] = None
        out["note"] = "No question answered correctly at neutral by BOTH stages."
        return out

    br, ar = b_nudge[q], a_nudge[q]
    correct_idx = br.get("correct_idx")
    wrong_idx = br.get("wrong_idx")
    out["note"] = note
    out["item"] = {
        "question": q,
        "choices": br.get("choices"),  # present only on newer runs
        "correct_letter": _letter(correct_idx),
        "correct_text": _choice_text(br, correct_idx),
        "asserted_wrong_letter": _letter(wrong_idx),
        "asserted_wrong_text": _choice_text(br, wrong_idx),
        "user_assertion": (config.D2_PERSUASION_TEMPLATES.get(nudge, "").format(
            wrong=f"{_letter(wrong_idx)}) {_choice_text(br, wrong_idx) or ''}".strip())),
        "base": {
            "neutral_pick": _letter(b_neutral.get(q, {}).get("pick")),
            "neutral_completion": _quote(b_neutral.get(q, {}).get("completion", "")),
            "nudged_pick": _letter(br.get("pick")),
            "nudged_completion": _quote(br.get("completion", "")),
            "flipped": br.get("flipped"),
        },
        "aligned": {
            "neutral_pick": _letter(a_neutral.get(q, {}).get("pick")),
            "neutral_completion": _quote(a_neutral.get(q, {}).get("completion", "")),
            "nudged_pick": _letter(ar.get("pick")),
            "nudged_completion": _quote(ar.get("completion", "")),
            "flipped": ar.get("flipped"),
        },
    }
    return out


# --------------------------------------------------------------------------- #
# D3 — who pays the tax (a rare entity the aligned model dropped; a popular one
# both keep). Picked by extreme popularity, not by eyeballing answers.
# --------------------------------------------------------------------------- #
def d3_example(root: Path, ladder: str, base: str, aligned: str) -> dict[str, Any]:
    out: dict[str, Any] = {"demo": "d3", "title": "Who pays the tax (rare vs popular)",
                           "stages": {}}
    bu = _load_unit(root, "d3", ladder, base)
    au = _load_unit(root, "d3", ladder, aligned)
    out["stages"]["base"] = {"stage": base, "status": bu["status"], "dir": bu["dir"],
                             "commit": _commit(bu["summary"])}
    out["stages"]["aligned"] = {"stage": aligned, "status": au["status"], "dir": au["dir"],
                                "commit": _commit(au["summary"])}
    for label, u in (("base", bu), ("aligned", au)):
        if u["status"] == "ok":
            s = u["summary"]["summary"]
            out["stages"][label]["overall_accuracy"] = s.get("overall_accuracy")
            out["stages"][label]["tail_accuracy"] = s.get("tail_accuracy")
            out["stages"][label]["head_accuracy"] = s.get("head_accuracy")
    if bu["status"] != "ok" or au["status"] != "ok":
        out["stages"]["base"]["missing"] = bu.get("missing")
        out["stages"]["aligned"]["missing"] = au.get("missing")
        return out

    b_by_q = {r["question"]: r for r in bu["raw"]}
    a_by_q = {r["question"]: r for r in au["raw"]}
    shared = [q for q in b_by_q if q in a_by_q]

    def _build(q: str) -> dict[str, Any]:
        br, ar = b_by_q[q], a_by_q[q]
        return {
            "question": q,
            "s_pop": br.get("s_pop"),
            "gold_answers": br.get("answers"),
            "base": {"correct": br.get("correct"),
                     "generation": _quote(br.get("generation", ""))},
            "aligned": {"correct": ar.get("correct"),
                        "generation": _quote(ar.get("generation", ""))},
        }

    # Rare illustration: lowest-popularity item where base was right, aligned wrong.
    tax = [q for q in shared if b_by_q[q].get("correct") and not a_by_q[q].get("correct")]
    tax.sort(key=lambda q: (b_by_q[q].get("s_pop") or 0.0))
    out["rare_example"] = _build(tax[0]) if tax else None
    if not tax:
        out["rare_note"] = ("No rare item where base was right and aligned wrong; "
                            "the tail gap is the aggregate decile curve, not this item.")

    # Popular control: highest-popularity item both got right.
    both = [q for q in shared if b_by_q[q].get("correct") and a_by_q[q].get("correct")]
    both.sort(key=lambda q: (b_by_q[q].get("s_pop") or 0.0), reverse=True)
    out["popular_example"] = _build(both[0]) if both else None
    return out


# --------------------------------------------------------------------------- #
# Markdown rendering (paste-ready; provenance on every block)
# --------------------------------------------------------------------------- #
def _fmt_pct(x: Any) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _fmt_ci(ci: Any) -> str:
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(v, (int, float)) for v in ci):
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    return ""


def _block(text: str) -> str:
    """Render a verbatim completion as an indented block (safe inside markdown)."""
    safe = (text or "").rstrip("\n")
    return "\n".join("    " + ln for ln in safe.splitlines()) or "    (empty)"


def to_markdown(ladder: str, base: str, aligned: str, exs: list[dict]) -> str:
    label = config.LADDERS[ladder].label if ladder in config.LADDERS else ladder
    L: list[str] = []
    L.append(f"# Real examples — {label} ({base} \u2192 {aligned})")
    L.append("")
    L.append("Every prompt and completion below is verbatim from `raw.jsonl`; every "
             "percentage is from the matching `dN.json`. Source dir + commit are noted "
             "per block. One illustrative item per probe sits next to the real aggregate.")
    L.append("")

    for ex in exs:
        if ex["demo"] == "d1":
            L.append("## D1 — Creativity collapse")
            if "prompt" in ex:
                L.append(f"**Prompt** (`{ex['prompt_key']}`): {ex['prompt']!r}")
            for who in ("base", "aligned"):
                s = ex["stages"].get(who, {})
                L.append("")
                L.append(f"### {who.upper()} — stage `{s.get('stage')}`  "
                         f"<sub>src: `{s.get('dir')}` · {s.get('commit')}</sub>")
                if s.get("status") != "ok":
                    L.append(f"> MISSING: {s.get('missing') or s.get('note')}")
                    continue
                L.append(f"- top-mode share: **{_fmt_pct(s.get('top_mode_share'))}** "
                         f"· normalized entropy: **{s.get('norm_entropy'):.3f}** "
                         f"{_fmt_ci(s.get('norm_entropy_ci'))} · n={s.get('n_valid')}")
                top5 = ", ".join(f"{a!r}\u00d7{c}" for a, c in (s.get("top5_answers") or []))
                L.append(f"- top-5 answers (of {s.get('n_valid')}): {top5}")
                L.append(f"- first {len(s.get('samples', []))} completions (generation order):")
                for smp in s.get("samples", []):
                    tr = smp["completion"].get("truncated_from")
                    note = f"  _(truncated from {tr} chars)_" if tr else ""
                    L.append(f"    - {smp['completion']['text']!r} \u2192 answer={smp.get('answer')!r}{note}")
            L.append("")

        elif ex["demo"] == "d2":
            L.append("## D2 — Sycophancy")
            bs, as_ = ex["stages"].get("base", {}), ex["stages"].get("aligned", {})
            L.append(f"- flip rate under `{ex['nudge']}` — base **{_fmt_pct(bs.get('flip_rate'))}** "
                     f"{_fmt_ci(bs.get('flip_rate_ci'))} vs aligned "
                     f"**{_fmt_pct(as_.get('flip_rate'))}** {_fmt_ci(as_.get('flip_rate_ci'))}")
            L.append(f"  <sub>base src: `{bs.get('dir')}` · {bs.get('commit')} | "
                     f"aligned src: `{as_.get('dir')}` · {as_.get('commit')}</sub>")
            if ex.get("note"):
                L.append(f"> NOTE: {ex['note']}")
            item = ex.get("item")
            if not item:
                L.append(f"> {ex.get('note', 'no item')}")
            else:
                L.append("")
                L.append(f"**Question:** {item['question']}")
                if item.get("choices"):
                    for i, c in enumerate(item["choices"]):
                        L.append(f"  - {LETTERS[i]}) {c}")
                ct = f" ({item['correct_text']})" if item.get("correct_text") else ""
                wt = f" ({item['asserted_wrong_text']})" if item.get("asserted_wrong_text") else ""
                L.append(f"- correct = **{item['correct_letter']}**{ct}; "
                         f"user falsely asserts **{item['asserted_wrong_letter']}**{wt}")
                if item.get("user_assertion"):
                    L.append(f"- user turn: {item['user_assertion']!r}")
                for who in ("base", "aligned"):
                    t = item[who]
                    L.append(f"- **{who}**: neutral picked {t['neutral_pick']}, "
                             f"after pressure picked {t['nudged_pick']} "
                             f"(flipped={t['flipped']}); completion:")
                    L.append(_block(t["nudged_completion"]["text"]))
            L.append("")

        elif ex["demo"] == "d3":
            L.append("## D3 — Who pays the tax")
            bs, as_ = ex["stages"].get("base", {}), ex["stages"].get("aligned", {})
            L.append(f"- tail (rarest decile) accuracy — base **{_fmt_pct(bs.get('tail_accuracy'))}** "
                     f"vs aligned **{_fmt_pct(as_.get('tail_accuracy'))}**; "
                     f"head (most popular) — base **{_fmt_pct(bs.get('head_accuracy'))}** "
                     f"vs aligned **{_fmt_pct(as_.get('head_accuracy'))}**")
            L.append(f"  <sub>base src: `{bs.get('dir')}` · {bs.get('commit')} | "
                     f"aligned src: `{as_.get('dir')}` · {as_.get('commit')}</sub>")
            for tag, key in (("RARE entity", "rare_example"), ("POPULAR entity", "popular_example")):
                item = ex.get(key)
                L.append("")
                if not item:
                    L.append(f"_{tag}: {ex.get('rare_note', 'no qualifying item')}_")
                    continue
                L.append(f"**{tag}** (s_pop={item['s_pop']}): {item['question']}")
                L.append(f"- gold: {item['gold_answers']}")
                L.append(f"- base (correct={item['base']['correct']}):")
                L.append(_block(item["base"]["generation"]["text"]))
                L.append(f"- aligned (correct={item['aligned']['correct']}):")
                L.append(_block(item["aligned"]["generation"]["text"]))
            L.append("")

    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", default=config.PRIMARY_LADDER)
    ap.add_argument("--base-stage", default="base")
    ap.add_argument("--aligned-stage", default="instruct")
    ap.add_argument("--results-root", default=str(config.RESULTS_DIR),
                    help="root of a results tree (local run output or a synced remote pull)")
    ap.add_argument("--out", default=None, help="output dir (default <results-root>/examples)")
    ap.add_argument("--d1-prompt", default="rand_1_100",
                    help="which D1 prompt_key to illustrate")
    ap.add_argument("--nudge", default="user_expert_wrong",
                    help="which D2 nudge to illustrate")
    ap.add_argument("--n-show", type=int, default=12,
                    help="how many verbatim D1 completions to show per stage")
    args = ap.parse_args()

    root = Path(args.results_root)
    out = Path(args.out) if args.out else root / "examples"
    out.mkdir(parents=True, exist_ok=True)

    exs = [
        d1_example(root, args.ladder, args.base_stage, args.aligned_stage,
                   args.d1_prompt, args.n_show),
        d2_example(root, args.ladder, args.base_stage, args.aligned_stage, args.nudge),
        d3_example(root, args.ladder, args.base_stage, args.aligned_stage),
    ]

    payload = {
        "ladder": args.ladder,
        "base_stage": args.base_stage,
        "aligned_stage": args.aligned_stage,
        "results_root": str(root),
        "examples": exs,
    }
    (out / "examples.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (out / "examples.md").write_text(
        to_markdown(args.ladder, args.base_stage, args.aligned_stage, exs),
        encoding="utf-8")

    # Console summary: flag any missing inputs loudly so nothing silently fabricates.
    print(f"[examples] ladder={args.ladder} {args.base_stage}->{args.aligned_stage} "
          f"root={root}")
    for ex in exs:
        statuses = {w: ex["stages"][w]["status"] for w in ex["stages"]}
        print(f"[examples] {ex['demo']}: {statuses}")
    print(f"[examples] wrote {out / 'examples.json'}")
    print(f"[examples] wrote {out / 'examples.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
