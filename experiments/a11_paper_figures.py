"""A11 — regenerate every power-grid figure in the paper from released artifacts.

Why this exists. Five of the nine figures in the submitted manuscript
(confusion_matrices_baselines, tsne_classes, shap_dependence_h3,
shap_rank_stability, leakage_ladder) had no generating code anywhere in the
repository: they were produced ad hoc, like the round-1 compute-cost JSON. Only
four traced to rebuild_experiments.py. For a paper whose contribution is
reproducibility that is a gap worth closing on its own merits, and in round 2 it
became blocking: the deep-model results were regenerated on local hardware, so
every figure showing a model-dependent number disagreed with the updated tables
until it could be redrawn.

Every figure below is built from files in outputs_revision/, so the figures and
the tables are guaranteed to describe the same model set.

Not covered: cwru_real_shap.pdf. The CWRU cross-domain check is a separate
experiment with its own models and its own input data; it is unaffected by the
power-grid model set and is still produced by colab/cwru_real_bearing_validation.ipynb.

Designs deliberately reproduce the submitted figures rather than improving on
them -- Reviewer 1 accepted the manuscript, so this round changes values, not
visual language.

Usage (from the repo root):
    python -m experiments.a11_paper_figures
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "outputs_revision"
ART = OUT / "paper_artifacts"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "SLG", "3-Phase", "Transient", "Harmonic", "Under-Freq"]
FEATURES = ["V_a_RMS", "V_b_RMS", "V_c_RMS", "V_imbalance", "V_magnitude",
            "Freq_mean", "Freq_std", "Freq_min", "Freq_max", "Freq_deviation",
            "RoCoF_mean", "RoCoF_max", "3rd_Harmonic", "5th_Harmonic", "THD"]
# Features whose discriminative content is spectral -- highlighted in the SHAP bars.
HARMONIC_FEATURES = {"3rd_Harmonic", "5th_Harmonic", "THD"}

TEX = {"V_a_RMS": r"$V_{a,RMS}$", "V_b_RMS": r"$V_{b,RMS}$",
       "V_c_RMS": r"$V_{c,RMS}$", "V_imbalance": r"$V_{imb}$",
       "V_magnitude": r"$V_{mag}$", "Freq_mean": r"$f_{mean}$",
       "Freq_std": r"$\sigma_f$", "Freq_min": r"$f_{min}$",
       "Freq_max": r"$f_{max}$", "Freq_deviation": r"$\Delta f$",
       "RoCoF_mean": r"RoCoF$_{mean}$", "RoCoF_max": r"RoCoF$_{max}$",
       "3rd_Harmonic": r"$H_3$", "5th_Harmonic": r"$H_5$", "THD": "THD"}

BLUE, GOLD, STEEL = "#1f4e79", "#c8a02c", "#6b8fb4"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── helpers ───────────────────────────────────────────────────────────────────
def _heatmap(ax, cm, title):
    """Counts heatmap in the submitted style: Blues, blank zeros, white-on-dark."""
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    n = cm.shape[0]
    thresh = cm.max() * 0.55
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if v == 0:
                continue
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "#1a1a1a")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#999999")


def _acc(cm):
    return np.trace(cm) / cm.sum()


# ── 1. CNN confusion matrix ───────────────────────────────────────────────────
def fig_confusion_cnn(cms):
    cm = np.array(cms["CNN"])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    _heatmap(ax, cm, f"1D-CNN (seed 42, acc {_acc(cm) * 100:.1f}\\%)"
             .replace("\\%", "%"))
    ax.set_ylabel("true")
    ax.set_xlabel("predicted")
    fig.savefig(FIG / "confusion_matrix_cnn.pdf")
    plt.close(fig)
    # Numbers the caption quotes.
    swap = int(cm[3, 5] + cm[5, 3])
    return {"accuracy": float(_acc(cm)),
            "transient_underfreq_both_directions": swap,
            "transient_to_underfreq": int(cm[3, 5]),
            "underfreq_to_transient": int(cm[5, 3]),
            "total_errors": int(cm.sum() - np.trace(cm))}


# ── 2. Baseline confusion matrices ────────────────────────────────────────────
def fig_confusion_baselines(cms):
    order = ["TCN", "Gradient Boosting", "XGBoost", "Random Forest"]
    order = [k for k in order if k in cms]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0))
    out = {}
    for ax, name in zip(axes.ravel(), order):
        cm = np.array(cms[name])
        out[name] = float(_acc(cm))
        _heatmap(ax, cm, f"{name} ({_acc(cm) * 100:.1f}%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("true")
    for ax in axes[1, :]:
        ax.set_xlabel("predicted")
    for ax in axes[0, :]:
        ax.set_xticklabels([])
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices_baselines.pdf")
    plt.close(fig)
    return out


# ── 3. t-SNE ──────────────────────────────────────────────────────────────────
def fig_tsne():
    z = np.load(ART / "tsne_seed42.npz")
    emb, y, correct = z["emb"], z["y"], z["cnn_correct"]
    colors = ["#9e9e9e", "#4f9ae0", "#1f4e79", "#2e7d32", "#c8a02c", "#c0392b"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k, (name, c) in enumerate(zip(CLASSES, colors)):
        m = y == k
        ax.scatter(emb[m, 0], emb[m, 1], s=4, c=c, label=name,
                   linewidths=0, alpha=0.85)
    bad = ~correct
    ax.scatter(emb[bad, 0], emb[bad, 1], s=26, marker="x", c="black",
               linewidths=0.8, label="CNN misclassified")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
              frameon=False, markerscale=2.2, handletextpad=0.3,
              columnspacing=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG / "tsne_classes.pdf")
    plt.close(fig)
    return {"n_misclassified": int(bad.sum()), "n_points": int(len(y))}


# ── 4. CNN SHAP: channels and time ────────────────────────────────────────────
def fig_shap_cnn():
    z = np.load(ART / "cnn_shap_full_seed42.npz")
    per_ch, per_t = z["per_channel"], z["per_timestep"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 2.9))

    labels = [r"$V_a$", r"$V_b$", r"$V_c$", "freq"]
    cols = [BLUE if i == int(np.argmax(per_ch[:3])) else STEEL for i in range(3)]
    cols.append(GOLD)
    a1.bar(labels, per_ch, color=cols, edgecolor="none")
    a1.set_ylabel(r"mean $|$SHAP$|$")
    a1.set_title("Per-channel attribution")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    a2.plot(np.arange(len(per_t)), per_t, lw=0.8, color=BLUE)
    a2.axvspan(100, 167, color=GOLD, alpha=0.25, lw=0,
               label=r"fault-onset window $t_0$")
    a2.set_xlabel("time step (ms)")
    a2.set_ylabel(r"mean $|$SHAP$|$")
    a2.set_title("Per-time-step attribution")
    a2.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "shap_cnn_channels_time.pdf")
    plt.close(fig)
    tot = per_ch.sum()
    return {"voltage_share": float(per_ch[:3].sum() / tot),
            "frequency_share": float(per_ch[3] / tot),
            "dominant_voltage_channel": labels[int(np.argmax(per_ch[:3]))]}


# ── 5. RF SHAP feature ranking ────────────────────────────────────────────────
def fig_shap_rf():
    z = np.load(ART / "rf_shap_values_seed42.npz", allow_pickle=True)
    sv = z["shap_values"]                    # (n, 15, 6)
    names = [str(s) for s in z["feature_names"]]
    mean_abs = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ypos = np.arange(len(order))[::-1]
    for y, i in zip(ypos, order):
        c = GOLD if names[i] in HARMONIC_FEATURES else BLUE
        ax.barh(y, mean_abs[i], color=c, height=0.72, edgecolor="none")
        ax.text(mean_abs[i] + mean_abs.max() * 0.015, y,
                f"{mean_abs[i]:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([TEX.get(names[i], names[i]) for i in order], fontsize=8)
    ax.set_xlabel(r"mean $|$SHAP value$|$ (all classes)")
    ax.set_xlim(0, mean_abs.max() * 1.18)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG / "shap_rf_features.pdf")
    plt.close(fig)
    return {"top3": [names[i] for i in order[:3]],
            "top3_values": [float(mean_abs[i]) for i in order[:3]],
            "harmonic_mass_share": float(
                sum(mean_abs[i] for i in range(len(names))
                    if names[i] in HARMONIC_FEATURES) / mean_abs.sum())}


# ── 6. SHAP dependence, H3 coloured by H5 ─────────────────────────────────────
def fig_shap_dependence():
    z = np.load(ART / "rf_shap_values_seed42.npz", allow_pickle=True)
    sv, Xu = z["shap_values"], z["X_unscaled"]
    names = [str(s) for s in z["feature_names"]]
    i3, i5 = names.index("3rd_Harmonic"), names.index("5th_Harmonic")
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    sc = ax.scatter(Xu[:, i3], sv[:, i3, :].sum(axis=1), c=Xu[:, i5],
                    cmap="viridis", s=14, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$H_5$ magnitude")
    ax.set_xlabel(r"$H_3$ magnitude")
    ax.set_ylabel(r"SHAP value of $H_3$ (summed over classes)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG / "shap_dependence_h3.pdf")
    plt.close(fig)
    return {"n_points": int(Xu.shape[0])}


# ── 7. SHAP rank stability across seeds ───────────────────────────────────────
def fig_rank_stability():
    z = np.load(OUT / "a5_shap_vectors.npz")
    rf, seeds = z["rf"], z["seeds"]          # (5, 15)
    # rank 1 = largest mean |SHAP|
    ranks = np.argsort(np.argsort(-rf, axis=1), axis=1) + 1
    mean_rank = ranks.mean(axis=0)
    top = np.argsort(mean_rank)[:8]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cmap = plt.get_cmap("tab10")
    for n, i in enumerate(top):
        ax.plot(range(len(seeds)), ranks[:, i], marker="o", ms=4.5, lw=1.1,
                color=cmap(n % 10), label=TEX.get(FEATURES[i], FEATURES[i]))
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("SHAP rank")
    ax.invert_yaxis()
    ax.set_yticks(range(1, int(ranks[:, top].max()) + 1))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG / "shap_rank_stability.pdf")
    plt.close(fig)
    return {"top8": [FEATURES[i] for i in top],
            "rank1_stable": bool((ranks[:, top[0]] == 1).all())}


# ── 8. Graded leakage ladder ──────────────────────────────────────────────────
def fig_leakage_ladder():
    s = json.loads((OUT / "a4_graded_leakage_summary.json").read_text())
    detail = json.loads((OUT / "a4_graded_leakage.json").read_text())
    rs = sorted(s.keys(), key=float)
    x = [s[r]["r_achieved_mean"] for r in rs]
    y = [s[r]["shap_rank_mean"] for r in rs]

    # Per-seed points, so the spread is visible rather than only the mean.
    # a4_graded_leakage.json is a flat list of one record per (r_target, seed).
    pts_x = [row["r_achieved"] for row in detail]
    pts_y = [row["shap_rank_of_proxy"] for row in detail]

    n_feat = 16
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhspan(4.5, n_feat + 0.5, color="#2e7d32", alpha=0.10, lw=0,
               label="physical-feature rank band (observed)")
    ax.axhline(1, ls="--", lw=0.9, color="black")
    # Sits just under the dashed rank-1 line; at y=1.6 it collides with it.
    ax.text(0.295, 2.3, "rank 1 = flagged at top", fontsize=7.5, color="#444444")
    if pts_x:
        ax.scatter(pts_x, pts_y, s=34, facecolor="#8b2b2b", edgecolor="black",
                   linewidths=0.6, zorder=3, alpha=0.85)
    ax.plot(x, y, color="#c0392b", lw=1.3, marker="^", ms=6, zorder=2,
            label="proxy SHAP rank (mean of 5 seeds)")
    ax.set_xlabel(r"Proxy label correlation $r$")
    ax.set_ylabel(f"SHAP rank of proxy (of {n_feat})")
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 4, 6, 8, 12, 16])
    ax.legend(loc="center right", frameon=False, fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(FIG / "leakage_ladder.pdf")
    plt.close(fig)
    return {"r": x, "shap_rank_mean": y}


def run(X_base=None, X_raw_base=None, y_base=None):
    print("\n" + "=" * 70)
    print("A11 — PAPER FIGURES (regenerated from outputs_revision/)")
    print("=" * 70)

    cms = json.loads((ART / "confusion_matrices_seed42.json").read_text())
    facts = {}
    facts["confusion_cnn"] = fig_confusion_cnn(cms)
    print(f"  confusion_matrix_cnn.pdf        {facts['confusion_cnn']}")
    facts["confusion_baselines"] = fig_confusion_baselines(cms)
    print(f"  confusion_matrices_baselines.pdf {facts['confusion_baselines']}")
    facts["tsne"] = fig_tsne()
    print(f"  tsne_classes.pdf                {facts['tsne']}")
    facts["shap_cnn"] = fig_shap_cnn()
    print(f"  shap_cnn_channels_time.pdf      {facts['shap_cnn']}")
    facts["shap_rf"] = fig_shap_rf()
    print(f"  shap_rf_features.pdf            {facts['shap_rf']}")
    facts["shap_dependence"] = fig_shap_dependence()
    print(f"  shap_dependence_h3.pdf          {facts['shap_dependence']}")
    facts["rank_stability"] = fig_rank_stability()
    print(f"  shap_rank_stability.pdf         {facts['rank_stability']}")
    facts["leakage"] = fig_leakage_ladder()
    print(f"  leakage_ladder.pdf              {facts['leakage']}")

    path = OUT / "a11_figure_facts.json"
    path.write_text(json.dumps(facts, indent=2, default=float))
    print(f"\n  saved {path.name} — the numbers each figure asserts, for the captions")
    return facts


if __name__ == "__main__":
    run()
