"""Stage 0 — reproduction sanity check.

Re-runs the seed-42 experiment (4 sklearn baselines + 1D-CNN) on the freshly
regenerated dataset and compares against the published numbers shipped in the
bundle (published_all_seeds_results.json, copied from the repo's
outputs/tables/all_seeds_results.json).

Tolerances: sklearn models are deterministic given the same data/versions
(delta < 0.005 expected); deep models retrain on CUDA without bit determinism
(delta < 0.02 acceptable).
"""
import json
from pathlib import Path

from experiments import common as C


def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print("STAGE 0 — REPRODUCTION SANITY CHECK (seed 42)")
    print("=" * 70)

    pub_path = Path(__file__).resolve().parent.parent / "published_all_seeds_results.json"
    published = json.loads(pub_path.read_text())["42"]

    split = C.seed_split(X_base, X_raw_base, y_base, seed=42)
    sc = C.StandardScaler()
    Xtr_s = sc.fit_transform(split["Xtr"])
    Xte_s = sc.transform(split["Xte"])

    report = {}
    for name, mdl in C.sklearn_zoo(42):
        if name not in published:      # only compare the original 4 baselines
            continue
        _, res, _ = C.fit_sklearn_instrumented(name, mdl, Xtr_s, split["ytr"], Xte_s, split["yte"])
        d = res["accuracy"] - published[name]["accuracy"]
        flag = "OK" if abs(d) < 0.005 else ("WARN" if abs(d) < 0.02 else "MISMATCH")
        report[name] = {"repro_acc": res["accuracy"], "pub_acc": published[name]["accuracy"],
                        "delta": d, "flag": flag}
        print(f"  {name:25s} repro={res['accuracy']:.4f} pub={published[name]['accuracy']:.4f} "
              f"delta={d:+.4f}  {flag}")

    _, res = C.fit_deep_instrumented(C.rx.FaultCNN1D, split)
    d = res["accuracy"] - published["1D-CNN (proposed)"]["accuracy"]
    flag = "OK" if abs(d) < 0.005 else ("WARN" if abs(d) < 0.02 else "MISMATCH")
    report["1D-CNN (proposed)"] = {"repro_acc": res["accuracy"],
                                   "pub_acc": published["1D-CNN (proposed)"]["accuracy"],
                                   "delta": d, "flag": flag}
    print(f"  {'1D-CNN (proposed)':25s} repro={res['accuracy']:.4f} "
          f"pub={published['1D-CNN (proposed)']['accuracy']:.4f} delta={d:+.4f}  {flag}")

    C.save_json(report, "stage0_sanity_report.json")
    worst = max(abs(v["delta"]) for v in report.values())
    print(f"  Worst |delta| = {worst:.4f} -> "
          f"{'REPRODUCTION CONFIRMED' if worst < 0.02 else 'INVESTIGATE BEFORE PROCEEDING'}")
    return report
