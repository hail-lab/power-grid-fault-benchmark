"""A9 — regenerate the paper's presentation artifacts on the authoritative
(hard) benchmark: per-class metrics, confusion matrices, full-resolution SHAP
arrays, and a t-SNE projection (Reviewer 1.9 figures + updated v1 tables).

Requires the A3 checkpoints (resume-safe: run after a3 or restore its outputs).
"""
import numpy as np
import shap
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

from experiments import common as C
from experiments.a5_shap_consistency import _cnn_shap


def _eval_deep_ckpt(cls, ckpt, split):
    mdl = cls().to(C.DEVICE)
    mdl.load_state_dict(torch.load(ckpt, map_location=C.DEVICE))
    mdl.eval()
    return mdl, C.rx._eval_deep(mdl, split["Rte"], split["yte"])


def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print("A9 — PAPER PRESENTATION ARTIFACTS (seed 42 focus + per-seed preds)")
    print("=" * 70)

    ckpt_dir = C.OUT / "checkpoints"
    art = C.OUT / "paper_artifacts"
    art.mkdir(exist_ok=True)

    preds = {}
    for seed in C.SEEDS:
        split = C.seed_split(X_base, X_raw_base, y_base, seed)
        cnn, res_cnn = _eval_deep_ckpt(C.rx.FaultCNN1D, ckpt_dir / f"1D-CNN_proposed_seed{seed}.pt", split)
        tcn, res_tcn = _eval_deep_ckpt(C.TCN, ckpt_dir / f"TCN_seed{seed}.pt", split)
        preds[seed] = {"y_true": split["yte"].astype("int8"),
                       "cnn": res_cnn["y_pred"].astype("int8"),
                       "tcn": res_tcn["y_pred"].astype("int8")}
        print(f"  seed {seed}: CNN acc={res_cnn['accuracy']:.4f}  TCN acc={res_tcn['accuracy']:.4f}")

        if seed != 42:
            continue

        # ── Seed 42: full presentation set ────────────────────────────────
        sc = C.StandardScaler()
        Xtr_s = sc.fit_transform(split["Xtr"])
        Xte_s = sc.transform(split["Xte"])
        yte = split["yte"]

        # Per-class report + confusion matrices
        report = classification_report(yte, preds[42]["cnn"], target_names=C.FAULT_NAMES,
                                       digits=4, output_dict=True)
        C.save_json(report, "paper_artifacts/per_class_report_cnn_seed42.json")

        cms = {"CNN": confusion_matrix(yte, preds[42]["cnn"]).tolist(),
               "TCN": confusion_matrix(yte, preds[42]["tcn"]).tolist()}
        for name, mdl in C.sklearn_zoo(42):
            if name in ("Gradient Boosting", "XGBoost", "Random Forest"):
                mdl.fit(Xtr_s, split["ytr"])
                yp = mdl.predict(Xte_s)
                cms[name] = confusion_matrix(yte, yp).tolist()
                if name == "Random Forest":
                    rf42 = mdl
        C.save_json(cms, "paper_artifacts/confusion_matrices_seed42.json")

        # Full-resolution CNN SHAP (channel 4 + timestep 500)
        per_channel, per_timestep = _cnn_shap(cnn, split["Rte"], 42)
        np.savez(art / "cnn_shap_full_seed42.npz",
                 per_channel=per_channel, per_timestep=per_timestep)
        print("  CNN SHAP full arrays saved")

        # RF SHAP per-sample values (for beeswarm/dependence/interaction plots)
        rng = np.random.default_rng(42)
        sub = rng.choice(len(Xte_s), 500, replace=False)
        sv = shap.TreeExplainer(rf42).shap_values(Xte_s[sub])
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
        np.savez_compressed(art / "rf_shap_values_seed42.npz",
                            shap_values=sv.astype("float32"),
                            X=Xte_s[sub].astype("float32"),
                            X_unscaled=sc.inverse_transform(Xte_s[sub]).astype("float32"),
                            y=yte[sub].astype("int8"),
                            feature_names=np.array(C.FEATURE_NAMES))
        print("  RF SHAP sample values saved")

        # t-SNE of engineered test features, with CNN correctness flag
        emb = TSNE(n_components=2, random_state=42, perplexity=30,
                   init="pca").fit_transform(Xte_s)
        np.savez_compressed(art / "tsne_seed42.npz",
                            emb=emb.astype("float32"), y=yte.astype("int8"),
                            cnn_correct=(preds[42]["cnn"] == yte))
        print("  t-SNE embedding saved")

    np.savez_compressed(art / "per_seed_predictions.npz",
                        **{f"{k}_{s}": v[k] for s, v in preds.items() for k in v})
    print("  per-seed predictions saved")
    return preds
