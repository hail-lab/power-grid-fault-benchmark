"""A8 — dataset export for Zenodo deposit.

Writes the full 50,000-event benchmark to a compressed NPZ plus a metadata
card, so the dataset can be deposited with a citable DOI (Heliyon data policy).
The NPZ regenerates deterministically from generate_dataset(50_000, global_seed=0);
the export exists so users can cite and download the exact artifact without
running the generator.
"""
import hashlib
import json
import platform
from datetime import date

import numpy as np

from experiments import common as C


def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print("A8 — DATASET EXPORT FOR ZENODO")
    print("=" * 70)

    exp_dir = C.OUT / "dataset_export"
    exp_dir.mkdir(exist_ok=True)

    npz_path = exp_dir / "power_grid_fault_benchmark_50k.npz"
    np.savez_compressed(
        npz_path,
        X_features=X_base.astype(np.float32),
        X_raw=X_raw_base.astype(np.float32),
        y=y_base.astype(np.int8),
        feature_names=np.array(C.FEATURE_NAMES),
        class_names=np.array(C.FAULT_NAMES),
    )
    sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    size_mb = npz_path.stat().st_size / 1e6

    meta = {
        "title": "Power Grid Fault Classification Benchmark (50,000 synthetic SCADA events)",
        "version": "2.0",
        "date_generated": str(date.today()),
        "n_events": int(len(y_base)),
        "class_distribution": {C.FAULT_NAMES[k]: int(v) for k, v in
                               zip(*np.unique(y_base, return_counts=True))},
        "raw_shape": list(X_raw_base.shape),
        "sampling_rate_hz": 1000,
        "window_ms": 500,
        "channels": ["Va", "Vb", "Vc", "frequency"],
        "n_features": len(C.FEATURE_NAMES),
        "feature_names": C.FEATURE_NAMES,
        "generator": "rebuild_experiments.generate_dataset(50_000, global_seed=0)",
        "generator_numpy_version": np.__version__,
        "generator_python_version": platform.python_version(),
        "file": npz_path.name,
        "file_size_mb": round(size_mb, 1),
        "sha256": sha256,
        "license": "MIT",
        "linked_code": "https://github.com/hail-lab/power-grid-fault-benchmark",
    }
    (exp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  {npz_path.name}: {size_mb:.0f} MB, sha256={sha256[:16]}...")
    return meta
