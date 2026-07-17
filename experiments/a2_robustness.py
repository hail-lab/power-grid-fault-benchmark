"""A2 — robustness suite (answers Reviewer 1.5).

Evaluates the trained 1D-CNN (raw waveforms) against Gradient Boosting and
XGBoost (engineered features, recomputed from the corrupted waveforms so the
comparison is end-to-end fair) under four test-time corruption families:

  1. measurement noise   : additive Gaussian sigma in {0,2,5,10,15,20,25} V
  2. missing data        : random per-channel sample dropout p in {1,5,10,20}%,
                           linear-interpolation imputation (SCADA gap filling)
  3. sensor failure      : one full channel zeroed (each of Va, Vb, Vc, freq)
  4. calibration drift   : phase-A offset {5,10,15} V; frequency offset
                           {0.1,0.2,0.3} Hz (beyond the training envelope)

All corruptions are applied to the raw test waveforms only (models trained on
clean data), across all 5 seeds; per-seed 1D-CNN checkpoints come from A3.
"""
import numpy as np
import torch

from experiments import common as C


def _interp_missing(R, p, rng):
    """Drop a fraction p of samples independently per channel, refill by linear
    interpolation (edge values held)."""
    R = R.copy()
    n_ev, T, n_ch = R.shape
    for c in range(n_ch):
        mask = rng.random((n_ev, T)) < p
        for i in np.where(mask.any(axis=1))[0]:
            good = ~mask[i]
            if good.sum() < 2:
                continue
            R[i, mask[i], c] = np.interp(np.where(mask[i])[0], np.where(good)[0], R[i, good, c])
    return R


def _corrupt(R, kind, param, rng):
    if kind == "noise":                    # sigma volts on voltage channels
        R = R.copy()
        R[:, :, :3] += rng.normal(0, param, R[:, :, :3].shape).astype(R.dtype)
        return R
    if kind == "missing":
        return _interp_missing(R, param, rng)
    if kind == "channel_fail":             # param = channel index, stuck at 0
        R = R.copy()
        R[:, :, param] = 0.0
        return R
    if kind == "volt_drift":               # param volts added to phase A
        R = R.copy()
        R[:, :, 0] += param
        return R
    if kind == "freq_drift":               # param Hz added to frequency channel
        R = R.copy()
        R[:, :, 3] += param
        return R
    raise ValueError(kind)


def _features_from_raw(R):
    X = np.zeros((len(R), len(C.FEATURE_NAMES)))
    for i in range(len(R)):
        sig = {"Va": R[i, :, 0].astype(np.float64), "Vb": R[i, :, 1].astype(np.float64),
               "Vc": R[i, :, 2].astype(np.float64), "freq": R[i, :, 3].astype(np.float64)}
        X[i] = C.rx.extract_features(sig)
    return X


def _cnn_acc(model, R, y):
    res = C.rx._eval_deep(model, R, y)
    return res["accuracy"]


CONFIGS = (
    [("noise", s) for s in (0, 2, 5, 10, 15, 20, 25)]
    + [("missing", p) for p in (0.01, 0.05, 0.10, 0.20)]
    + [("channel_fail", c) for c in (0, 1, 2, 3)]
    + [("volt_drift", v) for v in (5, 10, 15)]
    + [("freq_drift", f) for f in (0.1, 0.2, 0.3)]
)

CHANNEL_LABEL = {0: "Va", 1: "Vb", 2: "Vc", 3: "freq"}


def run(X_base, X_raw_base, y_base, n_test=1500):
    print("\n" + "=" * 70)
    print(f"A2 — ROBUSTNESS SUITE ({len(CONFIGS)} corruption configs x {len(C.SEEDS)} seeds)")
    print("=" * 70)

    ckpt_dir = C.OUT / "checkpoints"
    rows = []
    for seed in C.SEEDS:
        print(f"\n----- Seed {seed} -----")
        rng = np.random.default_rng(seed)
        split = C.seed_split(X_base, X_raw_base, y_base, seed)

        # Subsample the test set for tractable feature re-extraction
        sub = rng.choice(len(split["Rte"]), size=min(n_test, len(split["Rte"])), replace=False)
        Rte, yte = split["Rte"][sub], split["yte"][sub]

        # Feature-space models trained on the clean training partition
        sc = C.StandardScaler()
        Xtr_s = sc.fit_transform(split["Xtr"])
        feat_models = {}
        for name, mdl in C.sklearn_zoo(seed):
            if name in ("Gradient Boosting", "XGBoost"):
                mdl.fit(Xtr_s, split["ytr"])
                feat_models[name] = mdl

        # 1D-CNN from the A3 checkpoint
        cnn = C.rx.FaultCNN1D().to(C.DEVICE)
        cnn.load_state_dict(torch.load(ckpt_dir / f"1D-CNN_proposed_seed{seed}.pt",
                                       map_location=C.DEVICE))
        cnn.eval()

        for kind, param in CONFIGS:
            kind_tag = sum(ord(ch) for ch in kind)          # stable across sessions
            Rc = _corrupt(Rte, kind, param, np.random.default_rng(seed * 1000 + kind_tag))
            Xc_s = sc.transform(_features_from_raw(Rc))
            row = {"seed": seed, "kind": kind,
                   "param": CHANNEL_LABEL[param] if kind == "channel_fail" else param,
                   "CNN": _cnn_acc(cnn, Rc, yte)}
            for name, mdl in feat_models.items():
                row[name] = C.accuracy_score(yte, mdl.predict(Xc_s))
            rows.append(row)
            print(f"  {kind:13s} {str(row['param']):>5s}: CNN={row['CNN']:.4f}  "
                  + "  ".join(f"{n}={row[n]:.4f}" for n in feat_models))

    C.save_json(rows, "a2_robustness_rows.json")

    # Aggregate mean ± std across seeds
    agg = {}
    for kind, param in CONFIGS:
        key = f"{kind}|{CHANNEL_LABEL[param] if kind == 'channel_fail' else param}"
        sel = [r for r in rows if r["kind"] == kind and str(r["param"]) ==
               str(CHANNEL_LABEL[param] if kind == "channel_fail" else param)]
        agg[key] = {m: {"mean": float(np.mean([r[m] for r in sel])),
                        "std": float(np.std([r[m] for r in sel], ddof=1))}
                    for m in sel[0] if m not in ("seed", "kind", "param")}
    C.save_json(agg, "a2_robustness_summary.json")
    return rows, agg
