"""A4 — graded-correlation label-leakage study (answers Reviewer 2.1).

Extends the paper's corruption experiment (Section 5.2) from the single
extreme proxy (r = 0.961) to a graded ladder of proxy strengths, and reports
where each proxy ranks against the 15 physical features by correlation with
the label, versus where SHAP ranks it. This answers Reviewer 2's question
directly: can SHAP still surface the proxy when its correlation rank sits
mid-pack among the physical features?

Per variant (r_target in {0.3, 0.5, 0.7, 0.96}):
  proxy z = y + N(0, sigma^2), sigma chosen analytically so corr(z, y) = r_target
  (sigma = sigma_y * sqrt(1/r^2 - 1)).

Outputs:
  - a4_feature_label_correlation.json : Pearson r, R^2, mutual information of
    all 15 physical features vs the label (the table Reviewer 2 asked for)
  - a4_graded_leakage.json : per variant x seed: achieved r, R^2 rank of the
    proxy among 16 features, SHAP rank of the proxy, accuracy
"""
import numpy as np
import shap
from sklearn.feature_selection import mutual_info_classif

from experiments import common as C

R_TARGETS = [0.3, 0.5, 0.7, 0.96]


def _proxy(y, r_target, rng):
    sigma = y.std() * np.sqrt(1.0 / r_target**2 - 1.0)
    return y + rng.normal(0, sigma, size=len(y))


def _pearson_with_label(col, y):
    return float(np.corrcoef(col, y)[0, 1])


def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print("A4 — GRADED-CORRELATION LEAKAGE STUDY")
    print("=" * 70)

    # ── Per-feature correlation table (Reviewer 2's explicit request) ─────────
    corr_table = {}
    mi = mutual_info_classif(X_base, y_base, random_state=42)
    for j, fname in enumerate(C.FEATURE_NAMES):
        r = _pearson_with_label(X_base[:, j], y_base)
        corr_table[fname] = {"pearson_r": r, "r_squared": r * r, "mutual_info": float(mi[j])}
    C.save_json(corr_table, "a4_feature_label_correlation.json")
    ranked = sorted(corr_table.items(), key=lambda kv: -kv[1]["r_squared"])
    print("  Physical-feature R^2 vs label (top 5): "
          + ", ".join(f"{k}={v['r_squared']:.3f}" for k, v in ranked[:5]))

    # ── Graded proxy ladder ───────────────────────────────────────────────────
    results = []
    for r_target in R_TARGETS:
        for seed in C.SEEDS:
            rng = np.random.default_rng(seed)
            z = _proxy(y_base, r_target, rng)
            r_achieved = _pearson_with_label(z, y_base)
            X_aug = np.column_stack([X_base, z])
            names_aug = C.FEATURE_NAMES + [f"Label_Proxy_r{r_target}"]

            # R^2 rank of the proxy among all 16 features
            r2 = np.array([_pearson_with_label(X_aug[:, j], y_base) ** 2
                           for j in range(X_aug.shape[1])])
            r2_rank = int((-r2).argsort().tolist().index(15) + 1)

            split = C.seed_split(X_aug, X_raw_base, y_base, seed)
            sc = C.StandardScaler()
            Xtr_s = sc.fit_transform(split["Xtr"])
            Xte_s = sc.transform(split["Xte"])

            rf = C.RandomForestClassifier(n_estimators=200, max_depth=12,
                                          n_jobs=-1, random_state=seed)
            rf.fit(Xtr_s, split["ytr"])
            acc = C.accuracy_score(split["yte"], rf.predict(Xte_s))

            # SHAP rank of the proxy (TreeExplainer, same protocol as the paper)
            n_bg = min(500, len(Xte_s))
            bg = Xte_s[np.random.default_rng(seed).choice(len(Xte_s), n_bg, replace=False)]
            sv = shap.TreeExplainer(rf).shap_values(bg)
            sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
            mean_abs = np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1))
            shap_rank = int((-mean_abs).argsort().tolist().index(15) + 1)

            results.append({"r_target": r_target, "seed": seed,
                            "r_achieved": r_achieved, "r2_rank_of_proxy": r2_rank,
                            "shap_rank_of_proxy": shap_rank, "accuracy": acc,
                            "proxy_mean_abs_shap": float(mean_abs[15]),
                            "top_feature": names_aug[int(np.argmax(mean_abs))]})
            print(f"  r={r_target} seed={seed}: achieved r={r_achieved:.3f}, "
                  f"R^2 rank={r2_rank}/16, SHAP rank={shap_rank}/16, acc={acc:.4f}")

    C.save_json(results, "a4_graded_leakage.json")

    # Aggregate per r_target
    summary = {}
    for r_target in R_TARGETS:
        sel = [x for x in results if x["r_target"] == r_target]
        summary[str(r_target)] = {
            "r_achieved_mean": float(np.mean([x["r_achieved"] for x in sel])),
            "r2_rank_mean": float(np.mean([x["r2_rank_of_proxy"] for x in sel])),
            "shap_rank_mean": float(np.mean([x["shap_rank_of_proxy"] for x in sel])),
            "shap_rank_worst": int(max(x["shap_rank_of_proxy"] for x in sel)),
            "acc_mean": float(np.mean([x["accuracy"] for x in sel])),
            "detected_at_rank1_frac": float(np.mean([x["shap_rank_of_proxy"] == 1 for x in sel])),
        }
    C.save_json(summary, "a4_graded_leakage_summary.json")
    return results, summary
