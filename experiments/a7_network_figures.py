"""A7 — IEEE test-network presentation (answers Reviewer 3, round 2, comments 3 and 6).

R3: "The IEEE standard test network used in the study should be clearly presented
in the manuscript" and "The power-system flow and its operational details are
missing, making the system-level analysis incomplete."

The networks were always there -- A1 runs pandapower power flow and Zbus
short-circuit studies on the IEEE 14/30/57/118-bus cases to calibrate the
generator's per-class envelopes -- but the manuscript only ever reported the
resulting coverage statistics. Nothing showed the networks, their operating
point, or their fault levels. This stage surfaces that work:

  1. a7_powerflow_summary.json  operating point and fault levels per case:
     bus/line/generator counts, total load and generation, loss, voltage profile,
     maximum line loading, and the three-phase fault level (short-circuit MVA)
     distribution computed from the same Zbus used for the sag study.
  2. ieee14_oneline.pdf         one-line diagram of the IEEE 14-bus case.
  3. ieee_calibration_coverage.pdf
     the Zbus-derived retained-voltage distributions and SFR frequency nadirs
     with the generator's per-class envelopes overlaid -- makes the coverage
     numbers in the manuscript's calibration table legible rather than asserted.

ENVIRONMENT -- read this before running.
The main project interpreter cannot run pandapower: numba 0.63.0b1 registers an
overload for np.trapz at import, NumPy 2.x removed it, and pandapower calls the
numba path unconditionally from _pd2ppc regardless of runpp(numba=False).
Use the dedicated venv, which has no numba (pandapower then falls back to
pf.no_numba):

    ../../.venv-powerflow/Scripts/python.exe experiments/a7_network_figures.py

This module therefore imports only numpy / scipy / pandapower / matplotlib and
must NOT import experiments.common (which pulls in torch and sklearn).
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:            # allow running this file directly
    sys.path.insert(0, str(REPO))

CASES = ["case14", "case30", "case57", "case118"]
DIAGRAM_CASE = "case14"

# Kept identical to a1_calibration.py -- these must not drift apart.
GEN_ENVELOPES = {
    "sag_3ph": (0.35, 0.75),                 # Class 2 retained-voltage factor
    "sag_slg": (0.72, 0.95),                 # Class 1 faulted-phase retained voltage
    "freq_drop_underfreq_hz": (0.08, 0.55),  # Class 5
    "freq_drop_3ph_hz": (0.2, 1.2),          # Class 2
}
SAG_TRIGGER_PU = 0.95                        # IEEE 1159 sag threshold


def _stamp():
    import pandapower as pp
    import scipy
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandapower": pp.__version__,
        "numba": "absent (pandapower no_numba fallback)",
    }


# ── Power-flow and short-circuit summary ──────────────────────────────────────
def powerflow_summary(case_name):
    """Operating point and three-phase fault levels for one IEEE case."""
    import pandapower as pp
    import pandapower.networks as pn

    net = getattr(pn, case_name)()
    pp.runpp(net, numba=False)

    # Fault levels come from the SAME corrected short-circuit model A1 uses --
    # machines represented by X''d shunted to ground -- rather than a second
    # implementation here. Inverting the bare power-flow Ybus gives a
    # near-singular matrix and a 45 MVA fault level on a 100 MVA base.
    from experiments.a1_calibration import XDSS_PU_SWEEP, _zbus_and_v

    fault_mva = []
    for xdss in XDSS_PU_SWEEP:
        Zbus, V0, _ = _zbus_and_v(case_name, xdss)
        z_ii = np.abs(np.diag(Zbus))
        fault_mva.append((np.abs(V0) ** 2) * net.sn_mva / np.maximum(z_ii, 1e-12))
    fault_mva = np.concatenate(fault_mva)

    gen_mw = float(net.res_gen.p_mw.sum()) if len(net.res_gen) else 0.0
    gen_mw += float(net.res_ext_grid.p_mw.sum()) if len(net.res_ext_grid) else 0.0
    load_mw = float(net.res_load.p_mw.sum()) if len(net.res_load) else 0.0

    return {
        "n_bus": int(len(net.bus)),
        "n_line": int(len(net.line)),
        "n_trafo": int(len(net.trafo)),
        "n_gen": int(len(net.gen)) + int(len(net.ext_grid)),
        "n_load": int(len(net.load)),
        "s_base_mva": float(net.sn_mva),
        "base_kv_levels": sorted({float(v) for v in net.bus.vn_kv}),
        "total_load_mw": load_mw,
        "total_load_mvar": float(net.res_load.q_mvar.sum()) if len(net.res_load) else 0.0,
        "total_gen_mw": gen_mw,
        "losses_mw": gen_mw - load_mw,
        "vm_pu_min": float(net.res_bus.vm_pu.min()),
        "vm_pu_max": float(net.res_bus.vm_pu.max()),
        "va_deg_min": float(net.res_bus.va_degree.min()),
        "va_deg_max": float(net.res_bus.va_degree.max()),
        "line_loading_max_pct": float(net.res_line.loading_percent.max()),
        "line_loading_mean_pct": float(net.res_line.loading_percent.mean()),
        "fault_level_mva_min": float(fault_mva.min()),
        "fault_level_mva_median": float(np.median(fault_mva)),
        "fault_level_mva_max": float(fault_mva.max()),
    }


# ── One-line diagram ──────────────────────────────────────────────────────────
def oneline_diagram(case_name, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandapower as pp
    import pandapower.networks as pn
    import pandapower.plotting as plot

    net = getattr(pn, case_name)()
    pp.runpp(net, numba=False)

    # pandapower 3.x moved bus coordinates from net.bus_geodata (removed) to a
    # GeoJSON string column net.bus.geo. Generate them if the case carries none.
    if "geo" not in net.bus.columns or net.bus.geo.isna().all():
        plot.create_generic_coordinates(net, respect_switches=True)

    def _xy(bus):
        return json.loads(net.bus.geo.at[bus])["coordinates"]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    collections = [
        plot.create_bus_collection(net, net.bus.index, size=0.06,
                                   color="#1f2a44", zorder=3),
        plot.create_line_collection(net, net.line.index, color="#6b7a99",
                                    linewidths=1.2, use_bus_geodata=True, zorder=1),
    ]
    if len(net.trafo):
        # Without an explicit size the trafo symbol is scaled to the coordinate
        # extent and swamps the diagram.
        collections.append(
            plot.create_trafo_collection(net, net.trafo.index, color="#b5651d",
                                         size=0.05, linewidths=1.3, zorder=2))
    plot.draw_collections(collections, ax=ax)

    # Label buses, and mark generator and load buses distinctly.
    gen_buses = set(net.gen.bus) | set(net.ext_grid.bus)
    load_buses = set(net.load.bus)
    for b in net.bus.index:
        x, y = _xy(b)
        ax.annotate(str(b + 1), (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="#1f2a44")
        if b in gen_buses:
            ax.plot(x, y, marker="o", ms=11, mfc="none", mec="#c0392b",
                    mew=1.6, zorder=4)
        elif b in load_buses:
            ax.plot(x, y, marker="v", ms=6, color="#2e7d32", zorder=4)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=9, mfc="none", mec="#c0392b",
               label="generator / slack bus"),
        Line2D([], [], marker="v", ls="none", ms=6, color="#2e7d32", label="load bus"),
        Line2D([], [], color="#6b7a99", lw=1.2, label="line"),
    ] + ([Line2D([], [], color="#b5651d", lw=1.6, label="transformer")]
         if len(net.trafo) else []),
        loc="best", fontsize=7, frameon=False)

    ax.set_axis_off()
    # No in-figure title: the LaTeX caption carries it, and duplicating it wastes
    # column height and reads as an error in a typeset paper.
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


# ── Calibration coverage figure ───────────────────────────────────────────────
def coverage_figure(npz_path, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    panels = [
        ("sags_3ph", "sag_3ph",
         "Three-phase fault\nretained voltage (pu)", "Class 2 envelope"),
        ("sags_slg", "sag_slg",
         "SLG fault, faulted phase\nretained voltage (pu)", "Class 1 envelope"),
        ("nadirs", "freq_drop_underfreq_hz",
         "SFR frequency nadir (Hz)", "Class 5 envelope"),
        ("nadirs", "freq_drop_3ph_hz",
         "SFR frequency nadir (Hz)", "Class 2 envelope"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    for ax, (key, env_key, xlabel, env_label) in zip(axes, panels):
        vals = d[key]
        lo, hi = GEN_ENVELOPES[env_key]
        ax.hist(vals, bins=60, color="#6b7a99", edgecolor="none", density=True)
        ax.axvspan(lo, hi, color="#c0392b", alpha=0.18, lw=0)
        ax.axvline(lo, color="#c0392b", lw=1.0)
        ax.axvline(hi, color="#c0392b", lw=1.0)
        coverage = float(np.mean((vals >= lo) & (vals <= hi)))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(f"{env_label}\ncoverage = {coverage:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("density (network study)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


# ── Driver ────────────────────────────────────────────────────────────────────
def run(out_dir=None, npz_path=None, fig_dir=None):
    out_dir = Path(out_dir or REPO / "outputs_revision")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir or REPO / "outputs_revision" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Default to the round-2 (corrected short-circuit model) distributions.
    npz_path = Path(npz_path or REPO / "outputs_revision" /
                    "a1_calibration_distributions.npz")

    print("\n" + "=" * 70)
    print("A7 — IEEE TEST-NETWORK PRESENTATION")
    print("=" * 70)

    summary = {"_provenance": _stamp(), "cases": {}}
    for case in CASES:
        s = powerflow_summary(case)
        summary["cases"][case] = s
        print(f"  {case:8s} {s['n_bus']:3d} bus / {s['n_line']:3d} line / "
              f"{s['n_trafo']:2d} trafo | load {s['total_load_mw']:7.1f} MW | "
              f"loss {s['losses_mw']:6.2f} MW | "
              f"vm {s['vm_pu_min']:.3f}-{s['vm_pu_max']:.3f} pu | "
              f"maxload {s['line_loading_max_pct']:6.1f}% | "
              f"Ssc {s['fault_level_mva_min']:7.0f}-{s['fault_level_mva_max']:8.0f} MVA")

    path = out_dir / "a7_powerflow_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {path.name}")

    p = oneline_diagram(DIAGRAM_CASE, fig_dir / "ieee14_oneline.pdf")
    print(f"  saved {p.name}")

    if npz_path.exists():
        p = coverage_figure(npz_path, fig_dir / "ieee_calibration_coverage.pdf")
        print(f"  saved {p.name}")
    else:
        print(f"  ! {npz_path} not found — run a1_calibration first")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()
    run(out_dir=args.out, npz_path=args.npz, fig_dir=args.figures)
