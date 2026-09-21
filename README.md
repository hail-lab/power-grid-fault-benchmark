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
Kolmogorov-Smirnov statistics in `outputs_revision/a1_calibration_results.json`).
Six event classes: normal operation plus five fault types.

A compact 1D-CNN reference implementation (about 71,000 parameters, 0.3 MB)
evaluated against fifteen baselines under an identical 5-seed protocol: nine
feature-space classifiers (Random Forest, Gradient Boosting, XGBoost, LightGBM,
Logistic Regression, Decision Tree, SVM, kNN, MLP) and six raw-waveform models
(Shallow CNN, BiLSTM with attention, a dilated TCN, 1D adaptations of
SqueezeNet and MobileNetV2, and a cell obtained by differentiable neural
architecture search). Plus a robustness suite (noise, missing data, sensor
failure, calibration drift).

The reference model is **not** the most accurate architecture on this
benchmark - see the headline table below. It is the latency-optimal one.

A quantified SHAP physics-consistency protocol: two architecturally
independent explainers, rank-stability statistics across seeds and operating
conditions, a cross-domain check on the public CWRU bearing benchmark, and a
graded label-leakage sensitivity study.

All reported numbers are benchmark results on synthetic data; they are upper
bounds on real-world performance and are not deployment guarantees. Transfer
to utility-recorded waveforms requires separate validation.

> **Version note (v2.1).** The result tables in this repository are generated
> end-to-end by the released code at this commit and correspond to the second
> revision of the manuscript. **Every number and figure in the paper comes from
> `outputs_revision/`**, produced on one machine (Intel i7-12700, NVIDIA
> RTX 3060) so that all models are directly comparable. `outputs/tables/` holds
> the first-revision results, which were produced on cloud hardware; the nine
> feature-space models reproduce to within 0.0006, while the deep models differ
> by 0.3-2.8 pp through ordinary GPU and library non-determinism. Tables
> produced with a pre-release generator parameterization (higher accuracies;
> easier fault envelopes) are archived under
> `outputs/tables/legacy_prerelease/` and are superseded.
>
> Two corrections landed in v2.1 and are disclosed in the paper: the
> short-circuit calibration now represents synchronous machines by their
> subtransient reactance before inverting the admittance matrix (the previous
> code inverted the bare power-flow Ybus, which is near-singular), and all
> power-grid figures are now regenerated from released artifacts by
> `experiments/a11_paper_figures.py` - five of them previously had no
> generating code.

---

## Headline numbers (5-seed bootstrap on the released benchmark)

| Model               | Accuracy            | F1 (macro)          | CPU latency* |
|---------------------|---------------------|---------------------|--------------|
| NAS-1D (searched)   | 0.9891 ± 0.0018     | 0.9892 ± 0.0018     | 3.46 ms      |
| MobileNetV2-1D      | 0.9849 ± 0.0051     | 0.9850 ± 0.0052     | 2.56 ms      |
| SqueezeNet-1D       | 0.9831 ± 0.0014     | 0.9833 ± 0.0014     | 1.47 ms      |
| TCN                 | 0.9793 ± 0.0022     | 0.9793 ± 0.0022     | 1.23 ms      |
| **1D-CNN (ours)**   | **0.9681 ± 0.0053** | **0.9679 ± 0.0051** | **0.35 ms**  |
| BiLSTM + Attn       | 0.9306 ± 0.0486     | 0.9335 ± 0.0463     | 3.66 ms      |
| Shallow CNN         | 0.8183 ± 0.0609     | 0.8216 ± 0.0599     | 0.14 ms      |
| XGBoost             | 0.7692 ± 0.0038     | 0.7677 ± 0.0036     | 0.50 ms      |
| Gradient Boosting   | 0.7671 ± 0.0041     | 0.7658 ± 0.0039     | 5.05 ms      |
| Random Forest       | 0.7366 ± 0.0047     | 0.7348 ± 0.0057     | 31.5 ms      |

Full 16-model table: `outputs_revision/a3_summary.json` (plus
`a10_nas_results.json` for the searched cell). *Median single-event latency on
an Intel i7-12700; feature-model latencies exclude feature extraction.

Two readings matter here. The 1D-CNN beats every feature-based baseline in all
five seeds, so the representation gap dominates the model gap. But four
raw-waveform architectures beat the 1D-CNN, also in all five seeds - and the
latency column runs the other way, so each accuracy point costs compute.
Note that FLOPs do not predict latency: SqueezeNet-1D executes about half the
FLOPs of the 1D-CNN (6.6 vs 12.2 M) yet takes four times as long, because 1x1
and depthwise convolutions are memory-bandwidth bound. See
`outputs_revision/a6_compute_costs.json`.

---

## Repository layout

```
.
├── rebuild_experiments.py         Base pipeline (dataset, 7 original models, SHAP, figures)
├── run_all.py                     Revision experiment suite orchestrator (stages below)
├── experiments/
│   ├── common.py                  Shared protocol and the full deep model zoo
│   ├── lightweight.py             SqueezeNet-1D and MobileNetV2-1D (1D ports)
│   ├── stage0_sanity.py           Reproduction check vs committed tables
│   ├── a1_calibration.py          Zbus + SFR calibration study (pandapower)
│   ├── a2_robustness.py           Noise / missing-data / sensor-failure / drift suite
│   ├── a3_expanded_baselines.py   15 models x 5 seeds + checkpoints + training costs
│   ├── a4_leakage.py              Graded label-leakage ladder + feature-label correlations
│   ├── a5_shap_consistency.py     Quantified SHAP rank-stability metrics
│   ├── a6_compute_costs.py        Params, FLOPs, CPU/TorchScript/GPU latency per model
│   ├── a7_network_figures.py      IEEE one-line diagram, power flow, fault levels
│   ├── a8_dataset_export.py       Dataset NPZ export (the Zenodo artifact)
│   ├── a9_paper_artifacts.py      Per-class tables, confusion matrices, SHAP arrays, t-SNE
│   ├── a10_nas_search.py          DARTS cell search + weight-ranking pruning
│   └── a11_paper_figures.py       Rebuilds every power-grid figure from released artifacts
├── colab/
│   ├── colab_runner.ipynb         Run the full suite on a Colab GPU (upload bundle, Run all)
│   └── cwru_real_bearing_validation.ipynb   CWRU cross-domain check (real vibration data)
├── data/                          raw/ (CWRU .mat) and clean/ caches
├── models/
│   ├── cnn_1d.pt                  1D-CNN weights (seed 42)
│   └── checkpoints/               First-revision deep-model weights, 4 architectures x 5 seeds
├── outputs/                       FIRST-revision results (cloud hardware)
│   ├── figures/
│   └── tables/
│       └── legacy_prerelease/     Superseded pre-release tables (archived)
└── outputs_revision/              SECOND-revision results -- what the paper reports
    ├── figures/                   All paper figures, rebuilt by a11_paper_figures.py
    ├── paper_artifacts/           Confusion matrices, per-class report, SHAP arrays, t-SNE
    ├── checkpoints/               7 deep architectures x 5 seeds
    └── *.json / *.npz             One file per experiment stage
```

---

## Reproducibility

### Environment

```bash
pip install -r requirements.txt
```

Python 3.10+; CUDA optional (deep training is ~40 min for all 16 models on an RTX 3060, longer on CPU).

### One command, everything

```bash
python run_all.py --stages stage0,a1,a3,a2,a4,a5,a9
python -m experiments.a6_compute_costs
python -m experiments.a10_nas_search
python -m experiments.a11_paper_figures
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
