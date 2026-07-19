#!/usr/bin/env python
"""Render every talk figure from the aggregated tables. Saves PNG + the CSV it
used so each slide is traceable. Robust to missing data: a figure with no inputs
is skipped with a note, never crashes the run.
"""

from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Warm "Alignment Tax" palette so figures sit on the cream slide background
# instead of floating on a white rectangle. Keys applied defensively to
# tolerate older matplotlib versions that lack some of them.
_ATAX_CREAM = "#f7efe1"
_ATAX_RC = {
    "figure.facecolor": _ATAX_CREAM,
    "savefig.facecolor": _ATAX_CREAM,
    "axes.facecolor": _ATAX_CREAM,
    "axes.edgecolor": "#5c4631",
    "axes.labelcolor": "#2a211a",
    "axes.titlecolor": "#2a211a",
    "text.color": "#2a211a",
    "xtick.color": "#5c4631",
    "ytick.color": "#5c4631",
    "grid.color": "#b8a684",
    "legend.facecolor": "#efe6d8",
    "legend.edgecolor": "#e0d2bd",
}
for _k, _v in _ATAX_RC.items():
    try:
        plt.rcParams[_k] = _v
    except KeyError:
        pass

from atax import config
from atax.io_utils import read_json, read_jsonl

R = config.RESULTS_DIR
T = R / "tables"
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = ["base", "sft", "dpo", "instruct"]
PRIMARY = config.PRIMARY_LADDER


def _order_stages(df):
    df = df.copy()
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    return df.sort_values("stage")


def _csv(name):
    p = T / name
    if not p.exists():
        print(f"[plot] no {name}; skip")
        return None
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
def fig_d1_collapse():
    df = _csv("track1_diversity.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for lad in df["ladder"].unique():
        sub = _order_stages(df[df["ladder"] == lad])
        ax[0].plot(sub["stage"].astype(str), sub["mean_norm_entropy"], marker="o", label=lad)
    ax[0].set_title("Creativity collapse: answer diversity vs alignment stage")
    ax[0].set_ylabel("mean normalized entropy (1=diverse, 0=collapsed)")
    ax[0].set_xlabel("alignment stage")
    ax[0].set_ylim(0, 1)
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # histogram of the random-number prompt, base vs instruct (primary ladder)
    def hist_for(stage):
        p = R / "track1" / "d1" / f"{PRIMARY}__{stage}" / "raw.jsonl"
        if not p.exists():
            return None
        rows = read_jsonl(p)
        vals = [int(r["answer"]) for r in rows
                if r["prompt_key"] == "rand_1_100" and r["answer"] not in (None, "")
                and str(r["answer"]).lstrip("-").isdigit()]
        return vals

    base_vals = hist_for("base")
    inst_vals = hist_for("instruct")
    if base_vals and inst_vals:
        ax[1].hist(base_vals, bins=range(1, 102), alpha=0.5, label="base", density=True)
        ax[1].hist(inst_vals, bins=range(1, 102), alpha=0.5, label="instruct", density=True)
        ax[1].set_title("'Pick 1-100': base spreads, aligned spikes")
        ax[1].set_xlabel("answer")
        ax[1].set_ylabel("density")
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "random-number raw data\nnot available", ha="center")
    fig.tight_layout()
    fig.savefig(F / "d1_creativity_collapse.png", dpi=150)
    plt.close(fig)
    print("[plot] d1_creativity_collapse.png")


def fig_d2_sycophancy():
    df = _csv("track1_sycophancy.csv")
    if df is None:
        return
    # Headline plot uses the headline dataset only; the ARC control rows (if any)
    # live in the same CSV tagged by 'dataset' and are plotted/quoted separately.
    if "dataset" in df.columns:
        df = df[df["dataset"] == "truthfulqa_mc"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for nudge in sorted(df["nudge"].unique()):
        for lad in [PRIMARY]:
            sub = _order_stages(df[(df["nudge"] == nudge) & (df["ladder"] == lad)])
            if sub.empty:
                continue
            yerr = [sub["flip_rate"] - sub["ci_lo"], sub["ci_hi"] - sub["flip_rate"]]
            ax.errorbar(sub["stage"].astype(str), sub["flip_rate"], yerr=yerr,
                        marker="o", capsize=3, label=nudge)
    ax.set_title(f"Sycophancy: correct->agree-with-wrong flip rate ({PRIMARY})")
    ax.set_ylabel("flip rate")
    ax.set_xlabel("alignment stage")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "d2_sycophancy.png", dpi=150)
    plt.close(fig)
    print("[plot] d2_sycophancy.png")


def fig_d3_popqa():
    df = _csv("track1_popqa.csv")
    if df is None:
        return
    sub = df[df["ladder"] == PRIMARY]
    base = sub[sub["stage"] == "base"].sort_values("decile")
    inst = sub[sub["stage"] == "instruct"].sort_values("decile")
    if base.empty or inst.empty:
        print("[plot] popqa missing base/instruct; skip")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(base["decile"], base["accuracy"], marker="o", label="base")
    ax[0].plot(inst["decile"], inst["accuracy"], marker="o", label="instruct")
    ax[0].set_title("Who pays the tax: PopQA accuracy by popularity decile")
    ax[0].set_xlabel("popularity decile (0=rarest tail -> 9=head)")
    ax[0].set_ylabel("accuracy")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    merged = base.merge(inst, on="decile", suffixes=("_base", "_inst"))
    merged["delta"] = merged["accuracy_inst"] - merged["accuracy_base"]
    ax[1].bar(merged["decile"], merged["delta"], color=["crimson" if d < 0 else "seagreen" for d in merged["delta"]])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Per-decile change (instruct - base): the tail pays")
    ax[1].set_xlabel("popularity decile")
    ax[1].set_ylabel("accuracy delta")
    fig.tight_layout()
    fig.savefig(F / "d3_who_pays_tax.png", dpi=150)
    plt.close(fig)
    print("[plot] d3_who_pays_tax.png")


def fig_d4_dashboard():
    """The punchline: one visible KPI looks fine while the hidden side channels
    crater (Gemma-3-27B base -> instruct, a 2026 anchor).

    CAPABILITY is the reasoning-appropriate gsm8k(generative)+mmlu delta, NOT the
    all-task average: that average is dragged negative (-0.03) purely by
    loglikelihood-completion tasks (hellaswag/arc/piqa/winogrande) that score an
    instruct/reasoning model "cold" and understate it -- the SAME broken-ruler
    artifact the capability figure calls out, so using it here would contradict our
    own argument. Sycophancy is deliberately NOT a bar: it is an instruct-level,
    recipe-dependent cost, not a clean base->instruct delta.
    """
    LAD = "gemma3_27b"          # 2026 anchor for this slide (was PRIMARY/OLMo)
    GOOD, BAD = "#5b7b3a", "#b5402f"
    cap = _csv("track1_capability.csv")
    div = _csv("track1_diversity.csv")
    pop = _csv("track1_popqa.csv")
    cal = _csv("track1_calibration.csv")

    def delta(df, col, where=None):
        if df is None:
            return None
        sub = df[df["ladder"] == LAD]
        if where:
            sub = where(sub)
        b = sub[sub["stage"] == "base"][col].mean()
        i = sub[sub["stage"] == "instruct"][col].mean()
        if pd.isna(b) or pd.isna(i):
            return None
        return i - b

    bars = {}
    if cap is not None:
        bars["capability\n(dashboard)"] = delta(
            cap, "score", where=lambda s: s[s["task"].isin(["gsm8k", "mmlu"])])
    if div is not None:
        bars["diversity\n(entropy)"] = delta(div, "mean_norm_entropy")
    if pop is not None:
        bars["rare-tail\naccuracy"] = delta(pop, "accuracy", where=lambda s: s[s["decile"] == 0])
    if cal is not None:
        ece_col = "ece" if "ece" in cal.columns else cal.columns[2]
        bars["miscalibration\n(ECE)"] = delta(cal, ece_col)
    bars = {k: v for k, v in bars.items() if v is not None}
    if not bars:
        print("[plot] dashboard: no inputs; skip")
        return
    keys = list(bars)
    vals = [bars[k] for k in keys]
    colors = [GOOD if ("capability" in k and v > 0) else BAD for k, v in zip(keys, vals)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(keys, vals, color=colors, width=0.6)
    ax.axhline(0, color="#2a211a", lw=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax.set_ylabel("change from base to instruct")
    ax.set_title("The dashboard lies: one KPI up, three hidden axes worse\n"
                 "(Gemma-3-27B, base \u2192 instruct)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "d4_dashboard_lies.png", dpi=150)
    plt.close(fig)
    print(f"[plot] d4_dashboard_lies.png  bars="
          f"{dict(zip([k.replace(chr(10), ' ') for k in keys], [round(v, 3) for v in vals]))}")


def fig_d5_rarity():
    df = _csv("track2_sweep.csv")
    if df is None:
        return
    df = df[df["tax"].notna()]
    if df.empty:
        print("[plot] sweep has no tax values yet; skip")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (base, signal), sub in df.groupby(["base", "signal"]):
        g = sub.groupby("freq")["tax"].agg(["mean", "std", "count"]).reset_index().sort_values("freq")
        yerr = g["std"].fillna(0) / g["count"].pow(0.5)
        ax[0].errorbar(g["freq"], g["mean"], yerr=yerr, marker="o", capsize=3,
                       label=f"{base}/{signal}")
    ax[0].set_xscale("log")
    ax[0].set_title("The rarity dial: capability tax vs signal frequency")
    ax[0].set_xlabel("rare-signal frequency (log)")
    ax[0].set_ylabel("capability tax (base - finetuned)")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    for (base, signal), sub in df.groupby(["base", "signal"]):
        g = sub.groupby("freq")["acquisition"].mean().reset_index().sort_values("freq")
        ax[1].plot(g["freq"], g["acquisition"], marker="s", label=f"{base}/{signal}")
    ax[1].set_xscale("log")
    ax[1].set_title("Behaviour acquisition vs frequency (fixed budget)")
    ax[1].set_xlabel("rare-signal frequency (log)")
    ax[1].set_ylabel("acquisition rate")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(F / "d5_rarity_curve.png", dpi=150)
    plt.close(fig)
    print("[plot] d5_rarity_curve.png")


def fig_d5_gradnorm():
    df = _csv("track2_probe.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for base, sub in df.groupby("base"):
        sub = sub.sort_values("decile")
        xc = (sub["logfreq_lo"] + sub["logfreq_hi"]) / 2
        ax[0].plot(xc, sub["mean_grad_norm"], marker="o", label=base)
    ax[0].set_title("Mechanism: gradient magnitude vs token frequency")
    ax[0].set_xlabel("log10 token frequency (per million)")
    ax[0].set_ylabel("mean ||dL/dlogit||")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Scatter from one probe when raw probe samples are available. Some result
    # pulls only include aggregate tables; in that case, draw the verified
    # rare-signal callout instead of leaving a blank second panel.
    scat = sorted(glob.glob(str(R / "track2" / "probe" / "*" / "scatter.jsonl")))
    if scat:
        rows = read_jsonl(scat[0])
        import numpy as np

        py = np.array([r["p_y"] for r in rows])
        gn = np.array([r["grad_norm"] for r in rows])
        ax[1].scatter(py, gn, s=2, alpha=0.2)
        ax[1].set_title("||dL/dlogit|| = |p - e_y|: rare (low p) -> big update")
        ax[1].set_xlabel("p(realised token)")
        ax[1].set_ylabel("gradient magnitude")
    else:
        sp = T / "track2_probe_spearman.json"
        if sp.exists():
            meta = read_json(sp)
            base, info = next(iter(meta.items()))
            callout = info.get("rare_callout", {})
            signal = callout.get("mean_grad_norm_signal")
            overall = callout.get("mean_grad_norm_overall")
            n_signal = callout.get("n_signal_tokens")
            if signal is not None and overall is not None:
                ratio = signal / overall if overall else float("nan")
                bars = ax[1].bar(["overall", "rare-signal\ntokens"], [overall, signal],
                                  color=["#9c6b3f", "#b5402f"])
                ax[1].set_title("Rare-signal tokens get larger updates")
                ax[1].set_ylabel("mean ||dL/dlogit||")
                ax[1].set_ylim(0, max(overall, signal) * 1.42)
                ax[1].bar_label(bars, labels=[f"{overall:.2f}", f"{signal:.2f}"],
                                padding=4, fontsize=10)
                ax[1].text(0.5, 0.93, f"{ratio:.1f}x larger",
                           transform=ax[1].transAxes, ha="center", va="top", fontsize=11)
                ax[1].text(0.5, 0.86, f"N={n_signal:,} trigger-token positions",
                           transform=ax[1].transAxes, ha="center", va="top", fontsize=9)
                ax[1].grid(axis="y", alpha=0.3)
            else:
                ax[1].axis("off")
                ax[1].text(0.5, 0.5, "rare-signal callout\nnot available", ha="center")
        else:
            ax[1].axis("off")
            ax[1].text(0.5, 0.5, "scatter/raw probe data\nnot available", ha="center")
    fig.tight_layout()
    fig.savefig(F / "d5_gradient_mechanism.png", dpi=150)
    plt.close(fig)
    print("[plot] d5_gradient_mechanism.png")


def fig_d7_pareto():
    df = _csv("track3_soup.csv")
    if df is None:
        return
    df = df.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(df["capability_gsm8k"], df["diversity_entropy"], "-o", color="navy")
    for _, r in df.iterrows():
        ax.annotate(f"a={r['alpha']:.1f}", (r["capability_gsm8k"], r["diversity_entropy"]),
                    fontsize=7, alpha=0.7)
    ax.set_title("Buy it back cheap: capability-diversity Pareto (model soup)")
    ax.set_xlabel("capability (GSM8K)")
    ax.set_ylabel("diversity (entropy)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "d7_soup_pareto.png", dpi=150)
    plt.close(fig)
    print("[plot] d7_soup_pareto.png")


def fig_domain_tax():
    """Track4 domain-SFT: left = did the narrow fine-tune learn its domain (held-out
    PPL, lower = better); right = what it cost the base model's breadth (Δ general
    capability + Δ PopQA factual recall vs the pre-SFT base; negative = lost)."""
    df = _csv("track4_domain.csv")
    if df is None or len(df) == 0:
        return
    df = df.sort_values(["base", "domain"]).reset_index(drop=True)
    labels = [f"{r.base}\n{r.domain}" for r in df.itertuples()]
    xs = list(range(len(df)))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    if "heldout_ppl" in df:
        ax[0].bar(xs, df["heldout_ppl"].fillna(0), color="#5b7b3a")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("in-domain held-out PPL (lower = learned)")
    ax[0].set_title("Did the narrow fine-tune learn its domain?")
    ax[0].grid(axis="y", alpha=0.3)
    w = 0.38
    if "d_cap_avg" in df and df["d_cap_avg"].notna().any():
        ax[1].bar([i - w / 2 for i in xs], df["d_cap_avg"].fillna(0), width=w,
                  color="#b5402f", label="Δ capability (OOD avg)")
    if "d_popqa" in df and df["d_popqa"].notna().any():
        ax[1].bar([i + w / 2 for i in xs], df["d_popqa"].fillna(0), width=w,
                  color="#9c6b3f", label="Δ PopQA (factual recall)")
    ax[1].axhline(0, color="#2a211a", lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("change vs pre-SFT base (negative = lost)")
    ax[1].set_title("...and what did narrow fine-tuning cost? (breadth tax)")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "d8_domain_tax.png", dpi=150)
    plt.close(fig)
    print("[plot] d8_domain_tax.png")


def main() -> int:
    for fn in (fig_d1_collapse, fig_d2_sycophancy, fig_d3_popqa, fig_d4_dashboard,
               fig_d5_rarity, fig_d5_gradnorm, fig_d7_pareto, fig_domain_tax):
        try:
            fn()
        except Exception as e:  # noqa
            print(f"[plot] {fn.__name__} failed: {e}")
    print(f"[plot] figures in {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
