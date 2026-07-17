"""A3 — expanded baseline comparison (answers Reviewer 1.8 and Reviewer 2.3).

Adds XGBoost, LightGBM, SVM (RBF), kNN, MLP (feature space) and a lightweight
TCN (raw waveforms) to the paper's original 7 models, under the identical
5-seed bootstrap protocol. Also records training cost (wall time, peak CUDA
memory, parameter counts) for the compute-cost table (Reviewer 1.7), and saves
per-seed deep-model checkpoints for the downstream robustness (A2) and SHAP
consistency (A5) stages.
"""
import numpy as np
import torch

from experiments import common as C


def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print(f"A3 — EXPANDED BASELINES: {len(C.SEEDS)} seeds x (9 feature models + 4 deep models)")
    print("=" * 70)

    ckpt_dir = C.OUT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    all_results, cost = {}, {}
    for seed in C.SEEDS:
        print(f"\n----- Seed {seed} -----")
        split = C.seed_split(X_base, X_raw_base, y_base, seed)
        sc = C.StandardScaler()
        Xtr_s = sc.fit_transform(split["Xtr"])
        Xte_s = sc.transform(split["Xte"])

        seed_res, seed_cost = {}, {}

        for name, mdl in C.sklearn_zoo(seed):
            mdl, res, _ = C.fit_sklearn_instrumented(name, mdl, Xtr_s, split["ytr"], Xte_s, split["yte"])
            seed_res[name] = res
            seed_cost[name] = {"train_s": res["train_s"]}
            print(f"  {name:25s}: acc={res['accuracy']:.4f}  [{res['train_s']:.0f}s]")

        for name, cls in C.DEEP_ZOO:
            mdl, res = C.fit_deep_instrumented(cls, split)
            seed_res[name] = C.strip_preds(res)
            seed_cost[name] = {k: res[k] for k in ("train_s", "n_params", "peak_cuda_mb")}
            print(f"  {name:25s}: acc={res['accuracy']:.4f}  "
                  f"[{res['train_s']:.0f}s, {res['n_params']:,} params]")
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            torch.save(mdl.state_dict(), ckpt_dir / f"{safe}_seed{seed}.pt")

        all_results[seed] = seed_res
        cost[seed] = seed_cost

    # ── Aggregate: mean ± std per model ───────────────────────────────────────
    models = list(all_results[C.SEEDS[0]].keys())
    summary = {}
    for m in models:
        accs = np.array([all_results[s][m]["accuracy"] for s in C.SEEDS])
        f1s = np.array([all_results[s][m]["f1_macro"] for s in C.SEEDS])
        aucs = np.array([all_results[s][m]["auc"] for s in C.SEEDS])
        ts = np.array([cost[s][m]["train_s"] for s in C.SEEDS])
        summary[m] = {
            "acc_mean": accs.mean(), "acc_std": accs.std(ddof=1),
            "f1_mean": f1s.mean(), "f1_std": f1s.std(ddof=1),
            "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs, ddof=1)),
            "train_s_mean": ts.mean(),
            "n_params": cost[C.SEEDS[0]][m].get("n_params"),
            "peak_cuda_mb": cost[C.SEEDS[0]][m].get("peak_cuda_mb"),
        }
        print(f"{m:25s} acc={accs.mean():.4f}±{accs.std(ddof=1):.4f}  "
              f"f1={f1s.mean():.4f}  train={ts.mean():.0f}s")

    # Per-seed paired dominance of the 1D-CNN
    cnn = np.array([all_results[s]["1D-CNN (proposed)"]["accuracy"] for s in C.SEEDS])
    dominance = {m: int(np.sum(cnn > np.array([all_results[s][m]["accuracy"] for s in C.SEEDS])))
                 for m in models if m != "1D-CNN (proposed)"}

    C.save_json({str(s): {m: C.strip_preds(r) for m, r in sr.items()}
                 for s, sr in all_results.items()}, "a3_all_seeds_results.json")
    C.save_json(summary, "a3_summary.json")
    C.save_json(cost, "a3_training_cost.json")
    C.save_json(dominance, "a3_cnn_dominance.json")
    return all_results, summary
