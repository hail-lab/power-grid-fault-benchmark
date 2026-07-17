"""Heliyon revision — master experiment runner.

Runs (in dependency order, each stage resume-safe via its output files):

  stage0 : reproduction sanity check vs published results        (~10 min)
  a3     : expanded baselines, 5 seeds x 13 models + checkpoints (~2-3 h GPU)
  a2     : robustness suite (needs a3 checkpoints)               (~40 min)
  a4     : graded-correlation leakage study                      (~30 min)
  a5     : SHAP consistency metrics (needs a3 checkpoints)       (~30 min)
  a8     : dataset export for Zenodo (optional, --with-dataset)  (~10 min)

Usage:  python run_all.py [--stages stage0,a3,a2,a4,a5] [--with-dataset]
Outputs land in outputs_revision/ next to this script.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rebuild_experiments as rx  # noqa: E402
from experiments import common as C  # noqa: E402

STAGE_DONE_MARKERS = {
    "stage0": "stage0_sanity_report.json",
    "a1": "a1_calibration_results.json",
    "a3": "a3_summary.json",
    "a2": "a2_robustness_summary.json",
    "a4": "a4_graded_leakage_summary.json",
    "a5": "a5_shap_consistency.json",
    "a8": "dataset_export/metadata.json",
    "a9": "paper_artifacts/per_class_report_cnn_seed42.json",
}


def tiny_dataset(n_per_fault=400, global_seed=0):
    """Smoke-mode dataset: original class proportions, ~2,500 events."""
    import numpy as np
    np.random.seed(global_seed)
    labels = [0] * int(n_per_fault * 1.25) + sum([[k] * n_per_fault for k in range(1, 6)], [])
    y = np.array(labels)
    X = np.zeros((len(y), len(rx.FEATURE_NAMES)))
    X_raw = np.zeros((len(y), 500, 4), dtype="float32")
    for i in range(len(y)):
        sig = rx.generate_event(y[i])
        X[i] = rx.extract_features(sig)
        for c, k in enumerate(("Va", "Vb", "Vc", "freq")):
            X_raw[i, :, c] = sig[k]
    idx = np.random.permutation(len(y))
    return X[idx], X_raw[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="stage0,a1,a3,a2,a4,a5")
    ap.add_argument("--with-dataset", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny dataset + 2 seeds; validates code, numbers are meaningless")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]
    if args.with_dataset and "a8" not in stages:
        stages.append("a8")

    t0 = time.time()
    print(f"Device: {C.DEVICE}")
    if args.smoke:
        print("SMOKE MODE: tiny dataset, 2 seeds — code validation only")
        C.SEEDS = [42, 123]
        C.OUT = HERE / "outputs_smoke"
        C.OUT.mkdir(exist_ok=True)
        X_base, X_raw_base, y_base = tiny_dataset()
    else:
        print("Generating shared 50,000-event dataset (global seed 0) ...")
        X_base, X_raw_base, y_base = rx.generate_dataset(50_000)

    for stage in stages:
        marker = C.OUT / STAGE_DONE_MARKERS[stage]
        if marker.exists() and not args.force:
            print(f"\n[SKIP] {stage} — {marker.name} already exists (use --force to re-run)")
            continue
        t1 = time.time()
        if stage == "stage0":
            from experiments import stage0_sanity
            stage0_sanity.run(X_base, X_raw_base, y_base)
        elif stage == "a1":
            from experiments import a1_calibration
            a1_calibration.run(X_base, X_raw_base, y_base)
        elif stage == "a3":
            from experiments import a3_expanded_baselines
            a3_expanded_baselines.run(X_base, X_raw_base, y_base)
        elif stage == "a2":
            from experiments import a2_robustness
            a2_robustness.run(X_base, X_raw_base, y_base)
        elif stage == "a4":
            from experiments import a4_leakage
            a4_leakage.run(X_base, X_raw_base, y_base)
        elif stage == "a5":
            from experiments import a5_shap_consistency
            a5_shap_consistency.run(X_base, X_raw_base, y_base)
        elif stage == "a8":
            from experiments import a8_dataset_export
            a8_dataset_export.run(X_base, X_raw_base, y_base)
        elif stage == "a9":
            from experiments import a9_paper_artifacts
            a9_paper_artifacts.run(X_base, X_raw_base, y_base)
        else:
            raise SystemExit(f"unknown stage: {stage}")
        print(f"[DONE] {stage} in {(time.time()-t1)/60:.1f} min")

    print(f"\nALL REQUESTED STAGES COMPLETE in {(time.time()-t0)/60:.1f} min")
    print(f"Outputs: {C.OUT}")


if __name__ == "__main__":
    main()
