# Power-Grid Fault Classification: Synthetic Benchmark and 1D-CNN Reference Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21416715-blue.svg)](https://doi.org/10.5281/zenodo.21416715)

Code, simulator, and reproducibility scripts for the manuscript:

> **A Reproducible Physics-Informed Synthetic Benchmark for Power System Fault
> Classification with SHAP-Based Physics-Consistency Analysis.**

---

## What this repository contains

A 50,000-event synthetic SCADA fault-classification benchmark produced by a
fully documented parametric simulator. The generator's per-class envelopes are
quantitatively calibrated against Zbus short-circuit studies on the IEEE
14/30/57/118-bus networks and a system-frequency-response model (coverage and
Kolmogorov-Smirnov statistics reported in `outputs/tables/`). Six event
classes: normal operation plus five fault types.

A compact 1D-CNN reference implementation (about 71,000 parameters, 0.3 MB)
evaluated against twelve baselines - Random Forest, Gradient Boosting,
XGBoost, LightGBM, Logistic Regression, Decision Tree, SVM, kNN, MLP,
Shallow CNN, BiLSTM with attention, and a dilated TCN - under an identical
5-seed protocol, plus a robustness suite (noise, missing data, sensor
failure, calibration drift).

A quantified SHAP physics-consistency protocol: two architecturally
independent explainers, rank-stability statistics across seeds and operating
conditions, a cross-domain check on the public CWRU bearing benchmark, and a
graded label-leakage sensitivity study.

All reported numbers are benchmark results on synthetic data; they are upper
bounds on real-world performance and are not deployment guarantees. Transfer
to utility-recorded waveforms requires separate validation.

> **Version note (v2.0).** The result tables in this repository are generated
> end-to-end by the released code at this commit and correspond to the revised
> manuscript. Tables produced with a pre-release generator parameterization
> (higher accuracies; easier fault envelopes) are archived under
> `outputs/tables/legacy_prerelease/` for the record and are superseded.

---

## Headline numbers (5-seed bootstrap on the released benchmark)

| Model               | Accuracy            | F1 (macro)          | CPU latency* |
|---------------------|---------------------|---------------------|--------------|
| TCN                 | 0.9727 ± 0.0046     | 0.9728 ± 0.0046     | 1.22 ms      |
| **1D-CNN (ours)**   | **0.9654 ± 0.0070** | **0.9651 ± 0.0071** | **0.33 ms**  |
| BiLSTM + Attention  | 0.9025 ± 0.0477     | 0.9067 ± 0.0430     | —            |
| Shallow CNN         | 0.7977 ± 0.0396     | 0.8007 ± 0.0377     | —            |
| XGBoost             | 0.7686 ± 0.0029     | 0.7672 ± 0.0030     | 0.50 ms      |
| Gradient Boosting   | 0.7671 ± 0.0041     | 0.7658 ± 0.0042     | 5.05 ms      |
| Random Forest       | 0.7369 ± 0.0048     | 0.7350 ± 0.0049     | 31.5 ms      |

Full 13-model table: `outputs/tables/a3_summary.json`. *Median single-event
latency, measured on a desktop-class 12th-gen Intel Core CPU; feature-model
latencies exclude feature extraction. The 1D-CNN tops every feature-based
baseline in all five seeds; the TCN's +0.73 pp costs 5.4x the FLOPs and 3.7x
the CPU latency.

---

## Repository layout

```
.
├── rebuild_experiments.py         Base pipeline (dataset, 7 original models, SHAP, figures)
├── run_all.py                     Revision experiment suite orchestrator (stages below)
├── experiments/
│   ├── common.py                  Shared protocol, expanded model zoo (XGB/LGBM/SVM/kNN/MLP/TCN)
│   ├── stage0_sanity.py           Reproduction check vs committed tables
│   ├── a1_calibration.py          Zbus + SFR calibration study (pandapower)
│   ├── a2_robustness.py           Noise / missing-data / sensor-failure / drift suite
│   ├── a3_expanded_baselines.py   13 models x 5 seeds + checkpoints + training costs
│   ├── a4_leakage.py              Graded label-leakage ladder + feature-label correlations
│   ├── a5_shap_consistency.py     Quantified SHAP rank-stability metrics
│   ├── a8_dataset_export.py       Dataset NPZ export (the Zenodo artifact)
│   └── a9_paper_artifacts.py      Per-class tables, confusion matrices, SHAP arrays, t-SNE
├── colab/
│   ├── colab_runner.ipynb         Run the full suite on a Colab GPU (upload bundle, Run all)
│   └── cwru_real_bearing_validation.ipynb   CWRU cross-domain check (real vibration data)
├── data/                          raw/ (CWRU .mat) and clean/ caches
├── models/
│   ├── cnn_1d.pt                  1D-CNN weights (seed 42)
│   └── checkpoints/               All deep-model weights, 4 architectures x 5 seeds
└── outputs/
    ├── figures/                   Paper figures (regenerated from this code)
    └── tables/                    Result JSON/NPZ for every experiment stage
        └── legacy_prerelease/     Superseded pre-release tables (archived)
```

---

## Reproducibility

### Environment

```bash
pip install -r requirements.txt
```

Python 3.10+; CUDA optional (deep training is ~2-3 h on a T4, longer on CPU).

### One command, everything

```bash
python run_all.py --stages stage0,a1,a3,a2,a4,a5,a9
```

`stage0` first verifies that the committed result tables reproduce on your
installation; each stage then writes its tables/artifacts to
`outputs_revision/` (compare with the committed `outputs/tables/`). A
`--smoke` flag runs every stage on a tiny dataset in ~10 minutes for code
validation. The original single-script pipeline remains available as
`python rebuild_experiments.py`.

### CWRU consistency check

Open `colab/cwru_real_bearing_validation.ipynb` and run all cells (downloads
16 public .mat files from the CWRU Bearing Data Center; place them in
`data/raw/` manually if the server is unreachable).

---

## Dataset

Archived with a citable DOI: **[10.5281/zenodo.21416715](https://doi.org/10.5281/zenodo.21416715)**
(NPZ with raw waveforms, engineered features, labels, metadata, SHA-256).
The dataset also regenerates deterministically from this repository
(`generate_dataset(50_000, global_seed=0)`).

Each event: 500 ms window, three-phase voltage (Va, Vb, Vc) + instantaneous
frequency at 1 kHz. Fully synthetic; no utility-recorded waveforms.

| Class | Description                       | Count   |
|------:|-----------------------------------|--------:|
| 0     | Normal (baseline)                 | 10,000  |
| 1     | Single line-to-ground (SLG) fault |  8,000  |
| 2     | Three-phase symmetric fault       |  8,000  |
| 3     | Transient instability             |  8,000  |
| 4     | Harmonic distortion surge         |  8,000  |
| 5     | Under-frequency / load loss       |  8,000  |

Engineered features (15 scalars, tree/linear baselines): Va/Vb/Vc RMS,
V_imbalance, V_magnitude, Freq mean/std/min/max/deviation, RoCoF mean/max,
3rd/5th harmonic (phase a), THD. The deep models ingest the raw four-channel
waveform.

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{aljaloud2026fault,
  title   = {A Reproducible Physics-Informed Synthetic Benchmark for Power
             System Fault Classification with SHAP-Based Physics-Consistency
             Analysis},
  author  = {Aljaloud, Saud},
  journal = {Heliyon},
  year    = {2026},
  note    = {Dataset: \doi{10.5281/zenodo.21416715}}
}
```
