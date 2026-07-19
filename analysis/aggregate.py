#!/usr/bin/env python
"""Collect every result JSON into tidy CSV/JSON tables for plotting + slides.

Robust to partial runs: missing files are skipped, not fatal. Re-runnable.

Writes under results/tables/:
  track1_diversity.csv     ladder, stage, mean_norm_entropy, mean_top_mode_share
  track1_sycophancy.csv    ladder, stage, nudge, flip_rate, ci_lo, ci_hi
  track1_sycophancy_likelihood.csv  ladder, stage, nudge, flip_rate, ci_lo/hi, n, neutral_accuracy
                           (Perez base-robust gate; the only valid base flip number)
  track1_popqa.csv         ladder, stage, decile, accuracy, logpop_lo/hi
  track1_capability.csv    ladder, stage, task, score
  track1_capability_reasoning.csv  ladder, stage, task, score, mode
                           (GENERATIVE-reasoning cap: thinking models scored by
                           reason->extract; task=gsm8k is the flexible-extract
                           headline, gsm8k_strict the strict-match value)
  track1_calibration.csv   ladder, stage, ece, accuracy, overconfidence
  track2_sweep.csv         base, signal, method, freq, seed, acquisition, capability, tax
  track2_probe.csv         base, decile, mean_grad_norm, logfreq_lo/hi (+spearman.json)
  track3_soup.csv          ladder, alpha, capability_gsm8k, diversity_entropy
  summary.json             headline numbers for quick sanity
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from atax import config
from atax.io_utils import read_json

R = config.RESULTS_DIR
T = R / "tables"
T.mkdir(parents=True, exist_ok=True)


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa
        print(f"[agg] skip ({e})")
        return None


def diversity():
    rows = []
    for p in glob.glob(str(R / "track1" / "d1" / "*" / "d1.json")):
        d = read_json(p)["summary"]
        rows.append({"ladder": d["ladder"], "stage": d["stage"],
                     "mean_norm_entropy": d["mean_norm_entropy"],
                     "mean_top_mode_share": d["mean_top_mode_share"],
                     "mean_modes_covering_80": d["mean_modes_covering_80"]})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_diversity.csv", index=False)
        print(f"[agg] diversity: {len(rows)} rows")


def sycophancy():
    rows = []
    # Read the headline d2/ AND any sibling d2_<dataset>/ folders (the ARC control
    # and other non-adversarial sources land in track1/d2_<dataset>/ so they never
    # overwrite the headline aggregate). The literal 'd2/' glob alone would MISS
    # those siblings, silently dropping data the run actually produced.
    for p in glob.glob(str(R / "track1" / "d2" / "*" / "d2.json")) \
            + glob.glob(str(R / "track1" / "d2_*" / "*" / "d2.json")):
        d = read_json(p)
        s = d["summary"]
        dataset = s.get("dataset", "truthfulqa_mc")  # old runs predate the field
        for nudge, v in d["per_nudge"].items():
            rows.append({"ladder": s["ladder"], "stage": s["stage"], "nudge": nudge,
                         "dataset": dataset,
                         "flip_rate": v["flip_rate"], "ci_lo": v["flip_rate_ci"][0],
                         "ci_hi": v["flip_rate_ci"][1],
                         "n": v.get("n"), "n_parsefail": v.get("n_parsefail"),
                         "n_hedge": v.get("n_hedge"), "n_agree": v.get("n_agree"),
                         "flip_rate_incl_agree": v.get("flip_rate_incl_agree"),
                         "neutral_accuracy": s["neutral_accuracy"]})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_sycophancy.csv", index=False)
        print(f"[agg] sycophancy: {len(rows)} rows")


def sycophancy_likelihood():
    # The Perez-faithful LIKELIHOOD gate (run_d2 --likelihood) lives in
    # summary["likelihood"] of each d2.json: BASE-ROBUST sycophancy by argmax over
    # option-letter logprobs (no generation, no parsing). It is the integrity check
    # for any base->instruct flip delta, because the generation aggregate above
    # collapses to n<=1 on base/pretrained stages (they emit EOS/prose under the
    # generation protocol, so their flip denominator is noise). Emitted to its OWN
    # CSV so the generation headline table stays byte-identical; units run without
    # --likelihood have no block and are skipped.
    rows = []
    for p in glob.glob(str(R / "track1" / "d2" / "*" / "d2.json")) \
            + glob.glob(str(R / "track1" / "d2_*" / "*" / "d2.json")):
        d = read_json(p)
        s = d["summary"]
        lik = s.get("likelihood")
        if not lik:
            continue
        dataset = s.get("dataset", "truthfulqa_mc")
        for nudge, v in lik.get("per_nudge", {}).items():
            ci = v.get("flip_rate_ci") or [None, None]
            rows.append({"ladder": s["ladder"], "stage": s["stage"], "nudge": nudge,
                         "dataset": dataset, "flip_rate": v.get("flip_rate"),
                         "ci_lo": ci[0], "ci_hi": ci[1], "n": v.get("n"),
                         "n_unscored": v.get("n_unscored"),
                         "neutral_accuracy": lik.get("neutral_accuracy"),
                         "n_correct_neutral": lik.get("n_correct_neutral")})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_sycophancy_likelihood.csv", index=False)
        print(f"[agg] sycophancy_likelihood: {len(rows)} rows")


def popqa():
    rows = []
    for p in glob.glob(str(R / "track1" / "d3" / "*" / "d3.json")):
        d = read_json(p)
        s = d["summary"]
        for b in d["buckets"]:
            rows.append({"ladder": s["ladder"], "stage": s["stage"], "decile": b["decile"],
                         "accuracy": b["accuracy"], "logpop_lo": b["logpop_lo"],
                         "logpop_hi": b["logpop_hi"], "n": b["n"]})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_popqa.csv", index=False)
        print(f"[agg] popqa: {len(rows)} rows")


def capability():
    rows = []
    for p in glob.glob(str(R / "track1" / "cap" / "*" / "capability.json")):
        d = read_json(p)
        for task, score in d["scores"].items():
            rows.append({"ladder": d["ladder"], "stage": d["stage"], "task": task, "score": score})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_capability.csv", index=False)
        print(f"[agg] capability: {len(rows)} rows")


def capability_reasoning():
    """GENERATIVE-reasoning capability (run_capability --reasoning-gen) -> a SEPARATE
    csv so the loglikelihood track1_capability.csv stays untouched. One row per
    (ladder, stage, task): the flexible-extract HEADLINE is task=gsm8k, the
    strict-match value is task=gsm8k_strict; `mode` carried for audit."""
    rows = []
    for p in glob.glob(str(R / "track1" / "cap_reasoning" / "*" / "capability.json")):
        d = read_json(p)
        for task, score in d["scores"].items():
            rows.append({"ladder": d["ladder"], "stage": d["stage"], "task": task,
                         "score": score, "mode": d.get("mode", "reasoning_gen")})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_capability_reasoning.csv", index=False)
        print(f"[agg] capability_reasoning: {len(rows)} rows")


def calibration():
    rows = []
    for p in glob.glob(str(R / "track1" / "calib" / "*" / "calibration.json")):
        d = read_json(p)
        rows.append({"ladder": d["ladder"], "stage": d["stage"], "ece": d["ece"],
                     "accuracy": d["accuracy"], "overconfidence": d["overconfidence"]})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track1_calibration.csv", index=False)
        print(f"[agg] calibration: {len(rows)} rows")


def sweep():
    # base references first
    refs = {}
    for p in glob.glob(str(R / "track2" / "eval" / "*__BASEREF" / "sweep_eval.json")):
        d = read_json(p)
        refs[(d["base"], d["signal"])] = d["capability"]["average"]
    rows = []
    for p in glob.glob(str(R / "track2" / "eval" / "*" / "sweep_eval.json")):
        d = read_json(p)
        if d.get("base_ref"):
            continue
        # parse tag from path: base__signal__fFREQ__sSEED[__METHOD]
        # (the method suffix is absent for the default full-FT arm => "full").
        tag = Path(p).parent.name
        try:
            parts = tag.split("__")
            if len(parts) == 4:
                base, signal, fpart, spart = parts
                method = "full"
            elif len(parts) == 5:
                base, signal, fpart, spart, method = parts
            else:
                continue
            freq = float(fpart[1:]); seed = int(spart[1:])
        except Exception:
            continue
        ref = refs.get((d["base"], d["signal"]))
        cap = d["capability"]["average"]
        rows.append({"base": d["base"], "signal": d["signal"], "method": method,
                     "freq": freq, "seed": seed,
                     "acquisition": d["acquisition"], "capability": cap,
                     "base_capability": ref,
                     "tax": (ref - cap) if ref is not None else None})
    if rows:
        df = pd.DataFrame(rows).sort_values(["base", "signal", "method", "freq", "seed"])
        df.to_csv(T / "track2_sweep.csv", index=False)
        print(f"[agg] sweep: {len(rows)} rows")


def probe():
    rows = []
    spear = {}
    for p in glob.glob(str(R / "track2" / "probe" / "*" / "token_grad.json")):
        d = read_json(p)
        spear[d["base"]] = {"spearman": d["spearman_logfreq_vs_gradnorm"],
                            "rare_callout": d["rare_signal_callout"]}
        for b in d["buckets"]:
            rows.append({"base": d["base"], **b})
    if rows:
        pd.DataFrame(rows).to_csv(T / "track2_probe.csv", index=False)
        from atax.io_utils import write_json
        write_json(T / "track2_probe_spearman.json", spear)
        print(f"[agg] probe: {len(rows)} rows")


def soup():
    rows = []
    for p in glob.glob(str(R / "track3" / "*" / "*" / "soup_point.json")):
        d = read_json(p)
        rows.append({"ladder": d["ladder"], "alpha": d["alpha"],
                     "capability_gsm8k": d["capability_gsm8k"],
                     "diversity_entropy": d["diversity_entropy"]})
    if rows:
        pd.DataFrame(rows).sort_values(["ladder", "alpha"]).to_csv(T / "track3_soup.csv", index=False)
        print(f"[agg] soup: {len(rows)} rows")


def domain():
    """Track4 DOMAIN-SFT tax: for each (base, domain) compare the narrowly fine-tuned
    checkpoint against its PRE-SFT base model on every alignment-tax axis, plus the
    in-domain held-out PPL (did it actually learn the domain?).

    Baseline (pre-SFT) = the base ladder stage results, results/track1/<probe>/<base>__base/
    (DOMAIN_BASES stage name is "base"). SFT = results/track4_domain/<probe>/<base>__<domain>/.
    The domain probes write the SAME schema as a track1 stage, so the field paths match
    the track1 readers above. Missing files -> None (robust to a partial domain run).

    Delta convention is sft - base, so for the column's "bad" direction:
      d_cap_avg/d_popqa/d_entropy < 0  = capability / factual-recall / diversity LOST,
      d_flip_uw > 0  = MORE sycophantic,  d_ece > 0  = WORSE calibration.
    """
    def _rj(*parts):
        p = Path(*parts)
        return read_json(str(p)) if p.exists() else None

    def _avg_scores(cap):
        if not cap or "scores" not in cap:
            return None
        s = cap["scores"]
        # Average ONLY the canonical capability tasks: apples-to-apples vs the base,
        # and immune to the opt-in commonsense suite (which can carry sample counts)
        # + the precomputed "average" key that would otherwise pollute the mean.
        core = [k for k in config.CAPABILITY_TASKS if k in s and isinstance(s[k], (int, float))]
        if core:
            vals = [s[k] for k in core]
        else:
            vals = [v for k, v in s.items()
                    if k != "average" and isinstance(v, (int, float)) and 0.0 <= v <= 1.0]
        return sum(vals) / len(vals) if vals else None

    def _flip_uw(d2):
        # headline sycophancy = user_wrong flip from per_nudge (same schema as track1);
        # fall back to summary.flip_rate dict if an older writer lacked per_nudge.
        if not d2:
            return None
        pn = d2.get("per_nudge")
        if isinstance(pn, dict) and "user_wrong" in pn:
            return pn["user_wrong"].get("flip_rate")
        fr = d2.get("summary", {}).get("flip_rate")
        return fr.get("user_wrong") if isinstance(fr, dict) else None

    def _flip_n(d2):
        # valid denominator behind the user_wrong flip. A near-zero n (e.g. a base
        # completion model that emits nothing under the nudge -> 100% parsefail)
        # makes the flip rate -- and any delta against it -- NOISE / uncomputable.
        if not d2:
            return None
        pn = d2.get("per_nudge")
        if isinstance(pn, dict) and "user_wrong" in pn:
            return pn["user_wrong"].get("n")
        return None

    def _sg(d, key):  # summary getter
        return d.get("summary", {}).get(key) if d else None

    def _d(a, b):
        return (a - b) if (a is not None and b is not None) else None

    rows = []
    for base_key in config.DOMAIN_BASES:
        bd1 = _rj(R, "track1", "d1", f"{base_key}__base", "d1.json")
        bd2 = _rj(R, "track1", "d2", f"{base_key}__base", "d2.json")
        bd3 = _rj(R, "track1", "d3", f"{base_key}__base", "d3.json")
        bcap = _rj(R, "track1", "cap", f"{base_key}__base", "capability.json")
        bcal = _rj(R, "track1", "calib", f"{base_key}__base", "calibration.json")
        for dom_key in config.DOMAIN_CORPORA:
            tag = f"{base_key}__{dom_key}"
            sd1 = _rj(R, "track4_domain", "d1", tag, "d1.json")
            sd2 = _rj(R, "track4_domain", "d2", tag, "d2.json")
            sd3 = _rj(R, "track4_domain", "d3", tag, "d3.json")
            scap = _rj(R, "track4_domain", "cap", tag, "capability.json")
            scal = _rj(R, "track4_domain", "calib", tag, "calibration.json")
            ppl = _rj(R, "track4_domain", "models", tag, "indomain.json")
            if not any([sd1, sd2, sd3, scap]):
                continue  # this (base, domain) was not evaluated -> skip, don't fabricate
            r = {"base": base_key, "domain": dom_key,
                 "method": (ppl or {}).get("method", "full"),
                 "heldout_ppl": (ppl or {}).get("heldout_ppl"),
                 "base_cap_avg": _avg_scores(bcap), "sft_cap_avg": _avg_scores(scap),
                 "base_popqa": _sg(bd3, "overall_accuracy"), "sft_popqa": _sg(sd3, "overall_accuracy"),
                 "base_entropy": _sg(bd1, "mean_norm_entropy"), "sft_entropy": _sg(sd1, "mean_norm_entropy"),
                 "base_topmode": _sg(bd1, "mean_top_mode_share"), "sft_topmode": _sg(sd1, "mean_top_mode_share"),
                 "base_flip_uw": _flip_uw(bd2), "sft_flip_uw": _flip_uw(sd2),
                 "base_flip_n": _flip_n(bd2), "sft_flip_n": _flip_n(sd2),
                 "base_neutral_acc": _sg(bd2, "neutral_accuracy"), "sft_neutral_acc": _sg(sd2, "neutral_accuracy"),
                 "base_ece": (bcal or {}).get("ece"), "sft_ece": (scal or {}).get("ece")}
            r["d_cap_avg"] = _d(r["sft_cap_avg"], r["base_cap_avg"])
            r["d_popqa"] = _d(r["sft_popqa"], r["base_popqa"])
            r["d_entropy"] = _d(r["sft_entropy"], r["base_entropy"])
            r["d_topmode"] = _d(r["sft_topmode"], r["base_topmode"])
            r["d_flip_uw"] = _d(r["sft_flip_uw"], r["base_flip_uw"])
            r["d_neutral_acc"] = _d(r["sft_neutral_acc"], r["base_neutral_acc"])
            r["d_ece"] = _d(r["sft_ece"], r["base_ece"])
            rows.append(r)
    if rows:
        cols = ["base", "domain", "method", "heldout_ppl",
                "base_cap_avg", "sft_cap_avg", "d_cap_avg",
                "base_popqa", "sft_popqa", "d_popqa",
                "base_entropy", "sft_entropy", "d_entropy",
                "base_topmode", "sft_topmode", "d_topmode",
                "base_flip_uw", "sft_flip_uw", "d_flip_uw",
                "base_flip_n", "sft_flip_n",
                "base_neutral_acc", "sft_neutral_acc", "d_neutral_acc",
                "base_ece", "sft_ece", "d_ece"]
        df = pd.DataFrame(rows)
        df[[c for c in cols if c in df.columns]].sort_values(["base", "domain"]).to_csv(
            T / "track4_domain.csv", index=False)
        print(f"[agg] domain: {len(rows)} rows -> track4_domain.csv")


def main() -> int:
    for fn in (diversity, sycophancy, sycophancy_likelihood, popqa, capability,
               capability_reasoning, calibration, sweep, probe, soup, domain):
        _safe(fn)
    print(f"[agg] tables in {T}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
