"""Shared infrastructure for the Heliyon revision experiments.

Imports the authoritative pipeline (rebuild_experiments.py) and adds:
  - the expanded model zoo (XGBoost, LightGBM, SVM, kNN, MLP, TCN)
  - instrumented train/eval wrappers (wall time, peak CUDA memory)
  - the shared seed-split logic (identical to rebuild_experiments.run_bootstrap)

All new experiments reuse rebuild_experiments' generator, feature extractor,
deep-training loop and metrics so numbers stay consistent with the paper.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import rebuild_experiments as rx

SEEDS = rx.SEEDS                       # [42, 123, 7, 2024, 99]
FEATURE_NAMES = rx.FEATURE_NAMES
FAULT_NAMES = rx.FAULT_NAMES
DEVICE = rx.DEVICE

OUT = Path(__file__).resolve().parent.parent / "outputs_revision"
OUT.mkdir(exist_ok=True)


# ── Seed split (byte-identical to rebuild_experiments.run_bootstrap) ─────────
def seed_split(X_base, X_raw_base, y_base, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    idx = np.arange(len(X_base))
    idx_tmp, idx_te = train_test_split(idx, test_size=0.15,
                                       random_state=seed, stratify=y_base)
    idx_tr, idx_vl = train_test_split(idx_tmp, test_size=0.176,
                                      random_state=seed, stratify=y_base[idx_tmp])
    return dict(
        Xtr=X_base[idx_tr], Xvl=X_base[idx_vl], Xte=X_base[idx_te],
        Rtr=X_raw_base[idx_tr], Rvl=X_raw_base[idx_vl], Rte=X_raw_base[idx_te],
        ytr=y_base[idx_tr], yvl=y_base[idx_vl], yte=y_base[idx_te],
    )


# ── TCN (lightweight temporal convolutional network baseline) ────────────────
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(c_out), nn.ReLU(),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.down(x)


class TCN(nn.Module):
    """Dilated residual TCN, ~52k parameters (lightweight raw-waveform baseline)."""

    def __init__(self, in_ch=4, n_classes=6):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(in_ch, 32, k=5, dilation=1),
            TCNBlock(32, 48, k=5, dilation=2),
            TCNBlock(48, 64, k=5, dilation=4),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.blocks(x))


# ── Model zoo ─────────────────────────────────────────────────────────────────
def sklearn_zoo(seed):
    """Feature-space models. Hyperparameters mirror the paper's baselines;
    new models use widely adopted defaults of comparable capacity."""
    zoo = [
        ("Gradient Boosting", HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1, random_state=seed)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=seed)),
        ("Logistic Regression", LogisticRegression(
            max_iter=2000, C=1.0, random_state=seed, n_jobs=-1)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=12, random_state=seed)),
        ("kNN (k=5)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("MLP (128-64)", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                       early_stopping=True, random_state=seed)),
        ("SVM (RBF)", SVC(kernel="rbf", C=1.0, gamma="scale",
                          decision_function_shape="ovr", random_state=seed)),
    ]
    try:
        from xgboost import XGBClassifier
        zoo.append(("XGBoost", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            tree_method="hist", random_state=seed, n_jobs=-1,
            objective="multi:softprob", eval_metric="mlogloss")))
    except ImportError:
        print("  ! xgboost not installed — skipping")
    try:
        from lightgbm import LGBMClassifier
        zoo.append(("LightGBM", LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbosity=-1)))
    except ImportError:
        print("  ! lightgbm not installed — skipping")
    return zoo


DEEP_ZOO = [
    ("1D-CNN (proposed)", rx.FaultCNN1D),
    ("Shallow CNN", rx.ShallowCNN),
    ("BiLSTM + Attn", rx.BiLSTMAttn),
    ("TCN", TCN),
]


# ── Instrumented fit/eval ─────────────────────────────────────────────────────
def eval_sklearn(mdl, Xte_s, yte):
    yp = mdl.predict(Xte_s)
    res = {
        "accuracy": accuracy_score(yte, yp),
        "f1_macro": f1_score(yte, yp, average="macro"),
        "f1_weighted": f1_score(yte, yp, average="weighted"),
    }
    try:
        ypr = mdl.predict_proba(Xte_s)
        res["auc"] = roc_auc_score(yte, ypr, multi_class="ovr", average="macro")
    except Exception:
        res["auc"] = float("nan")   # SVC without probability calibration
    return res, yp


def fit_sklearn_instrumented(name, mdl, Xtr_s, ytr, Xte_s, yte):
    t0 = time.time()
    mdl.fit(Xtr_s, ytr)
    train_s = time.time() - t0
    res, yp = eval_sklearn(mdl, Xte_s, yte)
    res["train_s"] = train_s
    return mdl, res, yp


def fit_deep_instrumented(model_cls, split):
    """Train a deep model with the paper's protocol; record cost metrics."""
    mdl = model_cls().to(DEVICE)
    n_params = sum(p.numel() for p in mdl.parameters())
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mdl = rx._train_deep(mdl, split["Rtr"], split["ytr"], split["Rvl"], split["yvl"])
    train_s = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE.type == "cuda" else float("nan")
    res = rx._eval_deep(mdl, split["Rte"], split["yte"])
    res.update({"train_s": train_s, "n_params": int(n_params), "peak_cuda_mb": peak_mb})
    return mdl, res


def strip_preds(res):
    """JSON-safe copy of a result dict (drops y_pred, casts numpy types)."""
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in res.items() if k != "y_pred"}


def save_json(obj, name):
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"  saved {path.name}")
    return path
