#!/usr/bin/env python3
"""Report figures from results/ (development OOF arrays) and the training data.  python make_figures.py"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
args = parser.parse_args()
RES, DATA, FIG = args.results_dir.resolve(), args.data_dir.resolve(), args.output_dir.resolve()
FIG.mkdir(parents=True, exist_ok=True)
CUTOFF_ROW = 71  # March 13 23:00; row 72 is March 14 00:00, the first withheld hour.
assert pd.Timestamp("2026-03-11") + pd.Timedelta(hours=CUTOFF_ROW) == pd.Timestamp("2026-03-13 23:00")
HORIZONS = (1, 6, 24, 48)
# Validated categorical palette (dataviz reference instance): blue, orange, aqua, violet; greys for context.
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#0b0b0b", "#6b6a66", "#c9c8c2"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.linewidth": .6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": .5, "legend.frameon": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})

d = np.load(RES / "development_oof.npz")
y, fips, last = d["truth"], d["fips"], d["last_osi"]
stack, kin, direct, clim, prior = d["stack_bucket_nested"], d["kinetics_gbm"], d["direct_gbm"], d["mean_curve"], d["prior_only"]
# The submitted model is the equal-weight average of the tree stack and the sequence model (combine.py); when the
# sequence model's out-of-fold predictions are present, the figures and text numbers describe that average.
seq_path = RES / "sequence_oof.npz"
if seq_path.exists():
    s_ = np.load(seq_path)
    assert np.array_equal(s_["fips"], fips) and np.allclose(s_["truth"], y), "sequence_oof.npz is not aligned with development_oof.npz"
    seq = s_["oof"]; final = 0.5 * stack + 0.5 * seq; FINAL_LABEL = "final: average of stack and sequence model"
else:
    seq = None; final = stack; FINAL_LABEL = "lead-weighted stack"
manifest = json.load(open(RES / "run_manifest.json"))
H = manifest["final_half_life_h"]
train = pd.read_csv(DATA / "DM_Train.csv", parse_dates=["timestamp_et"]).sort_values(["fipsCode", "timestamp_et"])
osi_full = train.osi.to_numpy().reshape(-1, 216)
gust_full = train.gust.to_numpy().reshape(-1, 216)
meta = train.drop_duplicates("fipsCode").set_index("fipsCode")
hours = pd.date_range("2026-03-11 00:00", periods=216, freq="h")
lead = np.arange(1, 145)


def day_ticks(ax, start=0, end=216, step=24, offset=0):
    ax.set_xticks(np.arange(start, end + 1, step))
    ax.set_xticklabels([f"Mar {11 + (t + offset) // 24}" if (t + offset) % 24 == 0 else "" for t in np.arange(start, end + 1, step)])


# ---------------------------------------------------------------- Figure 1: event structure and scored windows
fig, axes = plt.subplots(3, 1, figsize=(6.6, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.35, 1, .45], "hspace": .1})
t = np.arange(216)
ax = axes[0]
ax.fill_between(t, np.quantile(osi_full, .1, axis=0), np.quantile(osi_full, .9, axis=0), color=BLUE, alpha=.15, lw=0, label="10th–90th percentile of counties")
ax.plot(t, osi_full.mean(0), color=BLUE, lw=1.6, label="mean OSI, 239 training counties")
ax.axvspan(0, CUTOFF_ROW, color=LIGHT, alpha=.35, lw=0)
ax.axvline(CUTOFF_ROW, color=INK, lw=.8, ls="--")
ax.text(36, .056, "outage record observed", ha="center", va="top", fontsize=7.5, color=INK)
ax.text(CUTOFF_ROW + 1.5, .056, "cutoff Mar 13 23:00", fontsize=7.5, color=INK, va="top")
ax.annotate("wave 1", xy=(84, .030), fontsize=7.5, color=MUTED); ax.annotate("wave 2", xy=(138, .013), fontsize=7.5, color=MUTED)
ax.set_ylabel("OSI"); ax.set_ylim(0, .06); ax.legend(loc="upper right")
ax = axes[1]
ax.fill_between(t, np.quantile(gust_full, .1, axis=0), np.quantile(gust_full, .9, axis=0), color=ORANGE, alpha=.15, lw=0)
ax.plot(t, gust_full.mean(0), color=ORANGE, lw=1.4, label="mean 10 m gust (URMA), 10th–90th pct. band")
ax.axvspan(0, CUTOFF_ROW, color=LIGHT, alpha=.35, lw=0); ax.axvline(CUTOFF_ROW, color=INK, lw=.8, ls="--")
ax.set_ylabel("gust (mph)"); ax.set_ylim(0, 60); ax.legend(loc="upper right")
ax = axes[2]
for i, h in enumerate(HORIZONS):
    ax.plot([72 + h, 215], [3 - i, 3 - i], color=VIOLET, lw=2.4, solid_capstyle="butt")
    ax.text(72 + h - 1.5, 3 - i, f"t+{h} h column scores hours from Mar {14 + h // 24} {h % 24:02d}:00", ha="right", va="center", fontsize=6.5, color=VIOLET)
ax.axvline(CUTOFF_ROW, color=INK, lw=.8, ls="--"); ax.set_ylim(-.7, 3.7); ax.set_yticks([]); ax.grid(False)
ax.spines["left"].set_visible(False); ax.set_ylabel("scored\nwindows", fontsize=7)
day_ticks(ax); ax.set_xlim(0, 215)
fig.savefig(FIG / "fig1_event_structure.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: restoration kinetics
from scipy.optimize import minimize_scalar
big = last > .02
norm = y[big] / last[big][:, None]
hl = []
for yy, l in zip(y[big], last[big]):
    hl.append(minimize_scalar(lambda h: np.sum((yy[:48] - l * np.exp2(-lead[:48] / h)) ** 2), bounds=(1, 400), method="bounded").x)
hl = np.array(hl)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": .3})
ax = axes[0]
for row in norm:
    ax.plot(lead[:72], row[:72], color=LIGHT, lw=.5, alpha=.8)
ax.plot(lead[:72], np.median(norm[:, :72], axis=0), color=BLUE, lw=1.8, label="median of counties")
ax.plot(lead[:72], np.exp2(-lead[:72] / H), color=ORANGE, lw=1.6, label=f"prior: half-life {H:.1f} h (estimated)")
ax.plot(lead[:72], np.exp2(-lead[:72] / 24), color=VIOLET, lw=1.2, ls="--", label="24 h half-life")
ax.set_xlabel("hours after cutoff"); ax.set_ylabel("OSI / OSI at cutoff"); ax.set_ylim(0, 1.6); ax.set_xlim(1, 72)
ax.legend(loc="upper right"); ax.set_title(f"(a) {big.sum()} training counties with OSI$_{{cut}}$ > 0.02", loc="left")
ax = axes[1]
ax.hist(hl, bins=np.arange(0, 62, 4), color=BLUE, alpha=.85, edgecolor="white", lw=.6)
ax.axvline(np.median(hl), color=ORANGE, lw=1.4); ax.text(np.median(hl) + 1.5, ax.get_ylim()[1] * .78, f"median {np.median(hl):.1f} h", color=ORANGE, fontsize=7.5)
ax.set_xlabel("best-fit half-life (h), first 48 leads"); ax.set_ylabel("counties"); ax.set_title("(b) per-county restoration half-life", loc="left")
fig.savefig(FIG / "fig2_kinetics.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: where the error is, and skill by lead
def rmse_by_lead(p):
    return np.sqrt(((p - y) ** 2).mean(0))

def smooth(v, k=6):
    return pd.Series(v).rolling(k, min_periods=1, center=True).mean().to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1.5, 1.3, 1], "wspace": .38})
ax = axes[0]
j = np.arange(144)                      # trajectory index: hours after Mar 14 00:00 (= j+1 hours after the cutoff); j=0 unscored
base = smooth(rmse_by_lead(clim)[1:])
lines = [(kin, BLUE, "kinetics GBM", 1.1), (direct, ORANGE, "direct GBM", 1.1), (stack, VIOLET, "lead-weighted stack", 1.4 if seq is not None else 1.9)]
if seq is not None:
    lines += [(seq, AQUA, "sequence model (SSM)", 1.1), (final, INK, "final: equal-weight average", 1.9)]
for p, c, lab, lw in lines:
    ax.plot(j[1:], 100 * (1 - smooth(rmse_by_lead(p)[1:]) / base), color=c, lw=lw, label=lab)
ax.axhline(0, color=MUTED, lw=.8); ax.set_ylim(-15, 75)
ax.set_xlabel("forecast hour $j$ (hours after Mar 14 00:00)"); ax.set_ylabel("skill vs. climatology (%)\n1 − RMSE/RMSE$_{clim}$, 6 h smoothed")
ax.set_xlim(1, 143); ax.legend(loc="upper right", fontsize=6.5); ax.set_title("(a) skill by forecast hour", loc="left")
for b in (12.5, 48.5):                  # stacking-bucket boundaries: j in 1-12, 13-48, 49-143
    ax.axvline(b, color=LIGHT, lw=.8)
ax = axes[1]
ax.plot(j, y.mean(0), color=INK, lw=1.6, label="observed mean")
ax.plot(j, prior.mean(0), color=ORANGE, lw=1.1, ls="--", label="restoration prior alone")
ax.plot(j, final.mean(0), color=VIOLET, lw=1.4, label="final model (OOF)" if seq is not None else "stack (OOF)")
ax.set_xlabel("forecast hour $j$"); ax.set_ylabel("mean OSI over counties"); ax.set_xlim(0, 143); ax.legend(loc="upper right")
ax.set_title("(b) mean trajectory", loc="left")
ax = axes[2]
w2 = slice(48, 96)
tp, pp = y[:, w2].max(1), final[:, w2].max(1)
ax.scatter(tp, pp, s=9, color=AQUA, alpha=.8, lw=0)
m = max(tp.max(), pp.max()) * 1.05
ax.plot([0, m], [0, m], color=LIGHT, lw=.8); ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel("observed peak OSI, Mar 16–17"); ax.set_ylabel("predicted peak (OOF)")
ax.set_title(f"(c) wave-2 peaks, r = {np.corrcoef(tp, pp)[0, 1]:.2f}", loc="left")
fig.savefig(FIG / "fig3_skill.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4: case studies
cases = [42053, 39117, 54015]
titles = {42053: "Forest Co., PA — still rising at cutoff", 39117: "Morrow Co., OH — deep first-wave outage", 54015: "Clay Co., WV — second-wave hit"}
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.15), sharey=False, gridspec_kw={"wspace": .3})
for ax, f in zip(axes, cases):
    i = int(np.where(fips == f)[0][0]); tr_i = int(np.where(meta.index.values[np.argsort(meta.index.values)] == f)[0][0]) if False else None
    row = np.where(train.drop_duplicates("fipsCode").fipsCode.to_numpy() == f)[0][0]
    ax.plot(np.arange(72), osi_full[row, :72], color=MUTED, lw=1.1, label="observed record")
    ax.plot(72 + lead - 1, y[i], color=INK, lw=1.3, label="truth (held out)")
    ax.plot(72 + lead - 1, prior[i], color=ORANGE, lw=1.1, ls="--", label="restoration prior")
    if seq is not None:
        ax.plot(72 + lead - 1, stack[i], color=BLUE, lw=.9, ls=":", label="tree stack (OOF)")
    ax.plot(72 + lead - 1, final[i], color=VIOLET, lw=1.5, label="final model (OOF)" if seq is not None else "stack (OOF)")
    ax.axvline(71, color=INK, lw=.7, ls=":")
    name = meta.loc[f, "countyName"]; st = meta.loc[f, "stateAbbr"]
    ax.set_title(f"{name}, {st} ({f})", loc="left", fontsize=8)
    ax.set_xticks(np.arange(0, 217, 48)); ax.set_xticklabels([f"Mar {11 + k * 2}" for k in range(5)], fontsize=7)
    ax.set_xlim(0, 215); ax.set_ylim(0, max(y[i].max(), osi_full[row].max()) * 1.12)
axes[0].set_ylabel("OSI")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7.2, bbox_to_anchor=(0.5, -0.06))
fig.savefig(FIG / "fig4_cases.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5: feature importance (gain), averaged over folds
imps = {"direct_gbm": {}, "kinetics_gbm": {}}
for k in range(5):
    p = RES / f"feature_importance_fold{k}.json"
    if not p.exists():
        continue
    j = json.load(open(p))
    for m in imps:
        for name, v in j[f"{m}_feature_importance"].items():
            imps[m][name] = imps[m].get(name, 0) + v / 5

def family(name):
    if name.startswith("hist_") or name in ("log_customers_observed_median",):
        return "observed outage history"
    if name.startswith("past_osi_decay") or name.startswith(("lead_", "log_lead")):
        return "lead / decay coordinate"
    return "weather"

fam_color = {"observed outage history": BLUE, "weather": ORANGE, "lead / decay coordinate": VIOLET}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"wspace": .9})
for ax, (m, title) in zip(axes, [("kinetics_gbm", "(a) kinetics GBM (residual target)"), ("direct_gbm", "(b) direct GBM")]):
    s = pd.Series(imps[m]); s = 100 * s / s.sum(); top = s.sort_values(ascending=True).tail(12)
    ax.barh(range(len(top)), top.values, color=[fam_color[family(n)] for n in top.index], height=.7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([n.replace("hist_observed_osi", "osi").replace("hist_", "").replace("wx_", "") for n in top.index], fontsize=6.8)
    ax.set_xlabel("share of total split gain (%)"); ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
axes[1].legend(handles=[Patch(color=c, label=l) for l, c in fam_color.items()], loc="lower right", fontsize=6.8)
fig.savefig(FIG / "fig5_importance.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- numbers used in the text
def text_numbers(p):
    """Error decomposition of one out-of-fold trajectory array (the t+1h window, cells j >= 1)."""
    e2 = (p - y) ** 2
    by_day = pd.DataFrame({"day": [f"Mar {14 + (j // 24)}" for j in range(1, 144)], "sse": e2[:, 1:].sum(0), "sst": (y[:, 1:] ** 2).sum(0)}).groupby("day").sum()
    by_day["share_sse"] = 100 * by_day.sse / by_day.sse.sum(); by_day["share_energy"] = 100 * by_day.sst / by_day.sst.sum()
    cs = pd.Series(e2[:, 1:].sum(1), index=fips).sort_values(ascending=False)
    e2w = e2[:, w2]                                                    # j = 48..95 -> March 16-17 (same window as the by-day table and Fig. 3c)
    p2 = p[:, w2].max(1)
    return {"share_sse_by_day": by_day.round(1).to_dict(), "top10_county_share_sse_pct": float(100 * cs.head(10).sum() / cs.sum()),
            "top1_county": int(cs.index[0]), "top1_share_pct": float(100 * cs.iloc[0] / cs.sum()),
            "top1_county_prediction_at_true_peak": float(p[np.where(fips == cs.index[0])[0][0], np.argmax(y[np.where(fips == cs.index[0])[0][0]])]),
            "top1_county_max_prediction": float(p[np.where(fips == cs.index[0])[0][0]].max()),
            "top1_county_true_peak": float(y[np.where(fips == cs.index[0])[0][0]].max()),
            "wave2_peak_corr": float(np.corrcoef(tp, p2)[0, 1]),
            "severe_cells_ratio": float(p[:, 1:][y[:, 1:] > .05].mean() / y[:, 1:][y[:, 1:] > .05].mean()),
            "mar14_rmse": float(np.sqrt(e2[:, 1:24].mean())),
            "mar16_17_rmse": float(np.sqrt(e2w.mean())), "mar16_17_rmse_clim": float(np.sqrt(((clim - y) ** 2)[:, w2].mean())),
            "mar16_17_skill_vs_clim_pct": float(100 * (1 - np.sqrt(e2w.mean()) / np.sqrt(((clim - y) ** 2)[:, w2].mean())))}

summary = text_numbers(final)
summary["submitted_model"] = FINAL_LABEL
summary["tree_stack"] = text_numbers(stack)
if seq is not None:
    summary["sequence_model"] = text_numbers(seq)
summary.update({"half_life_median_h": float(np.median(hl)), "half_life_iqr_h": [float(np.quantile(hl, .25)), float(np.quantile(hl, .75))],
                "n_big_counties": int(big.sum()),
                "mar14_rmse_reference": {"prior": float(np.sqrt(((prior - y) ** 2)[:, 1:24].mean())),
                                         "decay24": float(np.sqrt(((d["decay_24h"] - y) ** 2)[:, 1:24].mean())),
                                         "clim": float(np.sqrt(((clim - y) ** 2)[:, 1:24].mean()))}})
summary["figure1_cutoff"] = {"full_record_index": CUTOFF_ROW, "timestamp": "2026-03-13 23:00",
                               "first_withheld_index": CUTOFF_ROW + 1}
summary["feature_gain_share_pct"] = {
    m: {f: float(100 * sum(v for n, v in imp.items() if family(n) == f) / sum(imp.values()))
        for f in fam_color} for m, imp in imps.items()}
(FIG / "text_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(RES / "figure_text_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
print("figures written to", FIG)
