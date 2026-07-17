"""A5 — quantitative SHAP consistency metrics (answers Reviewer 1.6).

Replaces the paper's visual claims of SHAP stability with numbers:

  1. Across seeds (RF/TreeExplainer): the 15-feature mean |SHAP| ranking is
     recomputed for each of the 5 seeds (independent retraining + splits);
     report pairwise Spearman rho, Kendall tau, and top-5 Jaccard overlap.
  2. Across seeds (CNN/GradientExplainer): per-channel (4) and per-time-bin
     (50 bins of 10 ms) attribution vectors per seed; pairwise Spearman rho.
  3. Across operating conditions: RF SHAP rankings recomputed under test-time
     measurement noise sigma in {0, 5, 15} V (features re-extracted from the
     corrupted waveforms); Spearman rho vs the clean ranking.
  4. Cross-explainer concept agreement: attribution mass fractions per physical
     concept group (voltage magnitude/asymmetry, frequency dynamics, harmonics)
     for both explainers.
"""
import itertools

import numpy as np
import shap
import torch
from scipy.stats import kendalltau, spearmanr

from experiments import common as C
from experiments.a2_robustness import _corrupt, _features_from_raw

CONCEPTS = {
    "voltage": ["V_a_RMS", "V_b_RMS", "V_c_RMS", "V_imbalance", "V_magnitude"],
    "frequency": ["Freq_mean", "Freq_std", "Freq_min", "Freq_max",
                  "Freq_deviation", "RoCoF_mean", "RoCoF_max"],
    "harmonics": ["3rd_Harmonic", "5th_Harmonic", "THD"],
}
CNN_CHANNEL_CONCEPT = {0: "voltage", 1: "voltage", 2: "voltage", 3: "frequency"}


def _cnn_shap(model, X_raw_test, seed, n_bg=100, n_test=200):
    """GradientExplainer per-channel/per-timestep attribution, robust to both
    SHAP return conventions (list of per-class arrays vs single stacked array)."""
    model = model.to("cpu").eval()
    X_t = torch.from_numpy(C.rx._normalize_raw(X_raw_test)).permute(0, 2, 1).float()
    rng = np.random.default_rng(seed)
    n_bg, n_test = min(n_bg, len(X_t)), min(n_test, len(X_t))
    bg = X_t[rng.choice(len(X_t), n_bg, replace=False)]
    test = X_t[rng.choice(len(X_t), n_test, replace=False)]
    sv = shap.GradientExplainer(model, bg).shap_values(test)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:                       # (n, 4, 500) — single-output edge case
        sv = sv[..., None]
    assert sv.shape[1] == 4 and sv.shape[2] == X_t.shape[2], f"unexpected SHAP shape {sv.shape}"
    mean_abs = np.abs(sv).mean(axis=(0, 3))          # (4, 500)
    return mean_abs.mean(axis=1), mean_abs.mean(axis=0)   # per-channel, per-timestep


def _rf_shap_vector(rf, Xte_s, seed, n=500):
    rng = np.random.default_rng(seed)
    Xs = Xte_s[rng.choice(len(Xte_s), min(n, len(Xte_s)), replace=False)]
    sv = shap.TreeExplainer(rf).shap_values(Xs)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    return np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1))


def _pairwise(vectors, fn):
    vals = [fn(a, b) for a, b in itertools.combinations(vectors, 2)]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), [float(v) for v in vals]


def _spear(a, b):
    return spearmanr(a, b).statistic


def _kend(a, b):
    return kendalltau(a, b).statistic


def _jaccard_topk(a, b, k=5):
    ta, tb = set(np.argsort(a)[::-1][:k]), set(np.argsort(b)[::-1][:k])
    return len(ta & tb) / len(ta | tb)


def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print("A5 — QUANTITATIVE SHAP CONSISTENCY METRICS")
    print("=" * 70)

    ckpt_dir = C.OUT / "checkpoints"
    rf_vecs, cnn_ch_vecs, cnn_time_vecs = {}, {}, {}
    noise_rank_corr = []

    for seed in C.SEEDS:
        print(f"\n----- Seed {seed} -----")
        split = C.seed_split(X_base, X_raw_base, y_base, seed)
        sc = C.StandardScaler()
        Xtr_s = sc.fit_transform(split["Xtr"])
        Xte_s = sc.transform(split["Xte"])

        rf = C.RandomForestClassifier(n_estimators=200, max_depth=12,
                                      n_jobs=-1, random_state=seed)
        rf.fit(Xtr_s, split["ytr"])
        rf_vecs[seed] = _rf_shap_vector(rf, Xte_s, seed)
        top = np.argsort(rf_vecs[seed])[::-1][:3]
        print("  RF top-3: " + ", ".join(C.FEATURE_NAMES[i] for i in top))

        # CNN GradientExplainer (checkpoint from A3)
        cnn = C.rx.FaultCNN1D()
        cnn.load_state_dict(torch.load(ckpt_dir / f"1D-CNN_proposed_seed{seed}.pt",
                                       map_location="cpu"))
        per_channel, per_timestep = _cnn_shap(cnn, split["Rte"], seed)
        cnn_ch_vecs[seed] = per_channel
        cnn_time_vecs[seed] = per_timestep.reshape(50, 10).mean(axis=1)  # 50 bins

        # RF SHAP ranking under test-time noise (vs clean, same seed)
        for sigma in (5, 15):
            rng = np.random.default_rng(seed * 77 + sigma)
            sub = rng.choice(len(split["Rte"]), 1500, replace=False)
            Rc = _corrupt(split["Rte"][sub], "noise", sigma, rng)
            Xc_s = sc.transform(_features_from_raw(Rc))
            noisy_vec = _rf_shap_vector(rf, Xc_s, seed)
            noise_rank_corr.append({"seed": seed, "sigma_V": sigma,
                                    "spearman_vs_clean": _spear(rf_vecs[seed], noisy_vec),
                                    "top5_jaccard_vs_clean": _jaccard_topk(rf_vecs[seed], noisy_vec)})
            print(f"  noise sigma={sigma}V: Spearman vs clean = "
                  f"{noise_rank_corr[-1]['spearman_vs_clean']:.3f}")

    rf_list = [rf_vecs[s] for s in C.SEEDS]
    cnn_ch_list = [cnn_ch_vecs[s] for s in C.SEEDS]
    cnn_time_list = [cnn_time_vecs[s] for s in C.SEEDS]

    metrics = {
        "rf_across_seeds": {
            "spearman": dict(zip(("mean", "std", "pairs"), _pairwise(rf_list, _spear))),
            "kendall": dict(zip(("mean", "std", "pairs"), _pairwise(rf_list, _kend))),
            "top5_jaccard": dict(zip(("mean", "std", "pairs"), _pairwise(rf_list, _jaccard_topk))),
        },
        "cnn_time_across_seeds": {
            "spearman": dict(zip(("mean", "std", "pairs"), _pairwise(cnn_time_list, _spear))),
        },
        "cnn_channel_across_seeds": {
            "kendall": dict(zip(("mean", "std", "pairs"), _pairwise(cnn_ch_list, _kend))),
            "top_channel_agreement": float(np.mean(
                [np.argmax(v) == np.argmax(cnn_ch_list[0]) for v in cnn_ch_list])),
        },
        "rf_under_noise": noise_rank_corr,
    }

    # Cross-explainer concept mass fractions
    concept_mass = {"rf": {}, "cnn": {}}
    rf_mean = np.mean(rf_list, axis=0)
    for cname, feats in CONCEPTS.items():
        idxs = [C.FEATURE_NAMES.index(f) for f in feats]
        concept_mass["rf"][cname] = float(rf_mean[idxs].sum() / rf_mean.sum())
    cnn_mean = np.mean(cnn_ch_list, axis=0)
    for cname in CONCEPTS:
        idxs = [ch for ch, cc in CNN_CHANNEL_CONCEPT.items() if cc == cname]
        concept_mass["cnn"][cname] = float(cnn_mean[idxs].sum() / cnn_mean.sum()) if idxs else 0.0
    metrics["cross_explainer_concept_mass"] = concept_mass

    # Persist the raw per-seed vectors for figure generation (A7)
    np.savez(C.OUT / "a5_shap_vectors.npz",
             rf=np.array(rf_list), cnn_channel=np.array(cnn_ch_list),
             cnn_timebins=np.array(cnn_time_list), seeds=np.array(C.SEEDS))
    C.save_json(metrics, "a5_shap_consistency.json")

    print(f"\n  RF across-seed Spearman: {metrics['rf_across_seeds']['spearman']['mean']:.3f}"
          f" ± {metrics['rf_across_seeds']['spearman']['std']:.3f}")
    return metrics
