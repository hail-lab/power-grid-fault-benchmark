"""A1 — quantitative calibration study (answers Reviewer 2.2 and Reviewer 1.1).

Grounds the parametric generator's per-class boundary conditions in power-flow
and short-circuit calculations on the IEEE 14/30/57/118-bus MATPOWER cases
(via pandapower), plus a standard system-frequency-response (SFR) model:

  0. Network model. Synchronous machines are represented by their subtransient
     reactance X''d shunted to ground at the machine bus before Ybus is
     inverted. This was corrected in round 2 (Reviewer 3, comments 3 and 6):
     round 1 inverted the bare power-flow Ybus, which has no path to ground at
     generator buses, leaving the matrix near-singular. On case14 that gave
     |Z_ii| = 2.216-2.464 pu at every bus and a 45 MVA fault level on a 100 MVA
     base. X''d is swept over {0.15, 0.20, 0.30} pu on the machine base and the
     sensitivity is reported in "xdss_sensitivity".

  1. Three-phase fault voltage sags: classic Zbus short-circuit analysis.
     For every (fault bus f, observation bus i) pair and fault impedances
     Zf in {0, 0.5, 1, 2.5, 5} ohm:  V_i_fault = V_i0 - Z_if / (Z_ff + Zf) * V_f0.
     The distribution of retained voltages at observation buses (conditioned
     on a recordable sag, retained <= 0.95 pu per IEEE 1159 trigger) is
     compared against the generator's Class-2 sag envelope U(0.35, 0.75).
  2. Single-line-to-ground sags: sequence-network formula with Z2 = Z1 and
     Z0 = k * Z1, k swept over {1.0, 2.0, 3.0} (MATPOWER cases carry no
     zero-sequence data; the assumption range is standard for overhead
     transmission and is documented in the manuscript). Compared against the
     generator's Class-1 envelope U(0.72, 0.95).
  3. Frequency events: SFR model (swing equation with primary droop),
     d(dw)/dt = (dP_mech - dP_loss - D*dw) / (2H), governor droop R = 5%,
     Tg = 8 s, over H in [2,8] s, load damping D in [0.5,2], generation loss
     dP in [2,10]%. Frequency-nadir distribution compared against the
     generator's Class-5 U(0.08, 0.55) Hz and Class-2 U(0.2, 1.2) Hz drops.

Metrics reported per comparison: envelope coverage (fraction of simulated
values inside the generator range), two-sample KS statistic between the
simulated distribution and the generator's (uniform) envelope, and quantiles.
"""
import numpy as np
from scipy.stats import ks_2samp

# NOTE: experiments.common is imported lazily inside run(), not here. It pulls in
# torch and scikit-learn, and this module must stay importable from the
# pandapower-only venv (see the ENVIRONMENT note in a7_network_figures.py) so
# that A7 can reuse the corrected short-circuit model rather than reimplement it.

CASES = ["case14", "case30", "case57", "case118"]
# 3-phase faults are near-bolted; SLG faults ground through arc + tower-footing
# resistance (typical 1-40 ohm). Without distinct fault resistances the SLG
# formula degenerates to the 3ph one under the uniform Z0 = k*Z1 assumption
# (the k factor cancels at Zf = 0).
FAULT_Z_3PH_OHM = [0.0, 0.5, 1.0, 2.5, 5.0]
FAULT_Z_SLG_OHM = [1.0, 5.0, 10.0, 20.0, 40.0]
Z0_FACTORS = [1.0, 2.0, 3.0]
# Machine subtransient reactance on the machine base, swept like Z0_FACTORS.
# 0.15-0.30 pu spans salient-pole hydro through round-rotor thermal units.
XDSS_PU_SWEEP = [0.15, 0.20, 0.30]
PF_ASSUMED = 0.85                # machine rating inferred as max_p_mw / PF
SAG_TRIGGER_PU = 0.95            # IEEE 1159 sag threshold
GEN_ENVELOPES = {                # the generator's per-class parameter ranges
    "sag_3ph": (0.35, 0.75),     # Class 2 retained-voltage factor
    "sag_slg": (0.72, 0.95),     # Class 1 retained-voltage factor (faulted phase)
    "freq_drop_underfreq_hz": (0.08, 0.55),   # Class 5
    "freq_drop_3ph_hz": (0.2, 1.2),           # Class 2
}


def _machine_shunt_admittance(net, xdss_pu):
    """Subtransient admittances of the synchronous machines, per ppc bus index.

    A short-circuit Zbus is NOT the inverse of the power-flow Ybus. The
    power-flow matrix has no path to ground at the generator buses, so the only
    grounding is line charging plus the odd shunt; the matrix is near-singular
    and its inverse is dominated by the null space. On case14 that produced
    |Z_ii| of 2.216-2.464 pu at EVERY bus -- essentially uniform, and giving a
    45 MVA fault level on a 100 MVA base, which is not a transmission network.
    Every machine must be represented by its subtransient reactance shunted to
    ground before the matrix is inverted.

    MATPOWER cases carry no machine reactance and no machine rating (sn_mva is
    NaN throughout), so the rating is taken as max_p_mw / PF_ASSUMED and X''d is
    converted from the machine base to the system base:

        X_sys = X''d * S_base_sys / S_machine

    X''d itself is swept (XDSS_PU_SWEEP) exactly as Z0/Z1 is swept for the SLG
    study, and the sensitivity is reported rather than hidden in a point value.
    """
    n = net._ppc["bus"].shape[0]
    y = np.zeros(n, dtype=complex)
    lookup = net._pd2ppc_lookups["bus"]
    for df in (net.gen, net.ext_grid):
        if not len(df):
            continue
        for _, row in df.iterrows():
            b = int(lookup[int(row["bus"])])
            p_max = float(row.get("max_p_mw", np.nan))
            s_mach = (p_max / PF_ASSUMED) if np.isfinite(p_max) and p_max > 0 \
                else float(net.sn_mva)
            x_sys = xdss_pu * float(net.sn_mva) / s_mach
            y[b] += 1.0 / (1j * x_sys)
    return y


def _zbus_and_v(case_name, xdss_pu):
    """Short-circuit Zbus, pre-fault voltages, and per-bus impedance base.

    xdss_pu is the machine subtransient reactance on the machine's own base.
    Pass None to reproduce the round-1 (power-flow Ybus) behaviour.
    """
    import pandapower as pp
    import pandapower.networks as pn
    net = getattr(pn, case_name)()
    pp.runpp(net, numba=False)
    Ybus = np.asarray(net._ppc["internal"]["Ybus"].todense()).copy()
    if xdss_pu is not None:
        Ybus[np.diag_indices_from(Ybus)] += _machine_shunt_admittance(net, xdss_pu)
    V0 = np.asarray(net._ppc["internal"]["V"]).ravel()
    base_kv = net._ppc["bus"][:, 9]          # BASE_KV column
    zbase_ohm = base_kv ** 2 / net.sn_mva    # per-bus impedance base
    return np.linalg.inv(Ybus), V0, zbase_ohm


def _retained_3ph(Z, V0, zbase, zf_ohm):
    """Retained voltage magnitudes at all observation buses for faults at all
    buses (bolted or impedance zf)."""
    n = len(V0)
    out = []
    for f in range(n):
        zf_pu = zf_ohm / zbase[f]
        If = V0[f] / (Z[f, f] + zf_pu)
        Vi = V0 - Z[:, f] * If
        retained = np.abs(Vi) / np.maximum(np.abs(V0), 1e-9)
        out.append(retained)
    return np.concatenate(out)


def _retained_slg(Z, V0, zbase, zf_ohm, k0):
    """Faulted-phase retained voltage under SLG faults (Z2 = Z1, Z0 = k0*Z1)."""
    n = len(V0)
    out = []
    for f in range(n):
        zf_pu = zf_ohm / zbase[f]
        Iseq = V0[f] / (Z[f, f] * (2 + k0) + 3 * zf_pu)
        Va = V0 - (2 + k0) * Z[:, f] * Iseq
        retained = np.abs(Va) / np.maximum(np.abs(V0), 1e-9)
        out.append(retained)
    return np.concatenate(out)


def _sfr_nadir(H, D, dP, R=0.05, Tg=8.0, f0=60.0, t_end=30.0, dt=0.01):
    """Frequency nadir (Hz) from the SFR model with first-order governor."""
    dw, pm = 0.0, 0.0                     # per-unit speed deviation, governor output
    nadir = 0.0
    for _ in range(int(t_end / dt)):
        dpm = (-dw / R - pm) / Tg          # droop-controlled primary response
        ddw = (pm - dP - D * dw) / (2 * H)
        pm += dpm * dt
        dw += ddw * dt
        nadir = min(nadir, dw)
    return -nadir * f0


def _compare(sim_values, envelope, rng):
    lo, hi = envelope
    sim = np.asarray(sim_values)
    coverage = float(np.mean((sim >= lo) & (sim <= hi)))
    uniform_sample = rng.uniform(lo, hi, size=min(len(sim), 5000))
    ks = ks_2samp(sim, uniform_sample)
    return {
        "n_sim": int(len(sim)),
        "envelope": [lo, hi],
        "coverage_of_envelope": coverage,
        "ks_statistic_vs_envelope": float(ks.statistic),
        "sim_quantiles": {q: float(np.quantile(sim, float(q)))
                          for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
    }


def run(X_base=None, X_raw_base=None, y_base=None):
    from experiments import common as C

    print("\n" + "=" * 70)
    print("A1 — PANDAPOWER CALIBRATION STUDY")
    print("=" * 70)

    rng = np.random.default_rng(42)
    sags_3ph, sags_slg = [], []
    per_case = {}

    for case in CASES:
        c3, cs = [], []
        fault_mva = []
        for xdss in XDSS_PU_SWEEP:
            Z, V0, zbase = _zbus_and_v(case, xdss)
            for zf in FAULT_Z_3PH_OHM:
                c3.append(_retained_3ph(Z, V0, zbase, zf))
            for zf in FAULT_Z_SLG_OHM:
                for k0 in Z0_FACTORS:
                    cs.append(_retained_slg(Z, V0, zbase, zf, k0))
            # bolted three-phase fault level, for the network table (A7)
            z_ii = np.abs(np.diag(Z))
            fault_mva.append((np.abs(V0) ** 2) * 100.0 / np.maximum(z_ii, 1e-12))
        c3, cs = np.concatenate(c3), np.concatenate(cs)
        fault_mva = np.concatenate(fault_mva)
        # keep recordable sags only (IEEE 1159 trigger), clip numerical noise
        c3 = np.clip(c3[c3 <= SAG_TRIGGER_PU], 0, 1.2)
        cs = np.clip(cs[cs <= SAG_TRIGGER_PU], 0, 1.2)
        sags_3ph.append(c3)
        sags_slg.append(cs)
        per_case[case] = {"n_buses": len(V0),
                          "sag3ph_median": float(np.median(c3)),
                          "sagslg_median": float(np.median(cs)),
                          "fault_level_mva_min": float(fault_mva.min()),
                          "fault_level_mva_median": float(np.median(fault_mva)),
                          "fault_level_mva_max": float(fault_mva.max())}
        print(f"  {case}: {len(V0)} buses, 3ph sag median={np.median(c3):.3f} pu, "
              f"SLG faulted-phase median={np.median(cs):.3f} pu, "
              f"Ssc median={np.median(fault_mva):.0f} MVA")

    # Per-X''d sensitivity: the assumption must be shown not to drive the result.
    xdss_sensitivity = {}
    for xdss in XDSS_PU_SWEEP:
        v3, vs = [], []
        for case in CASES:
            Z, V0, zbase = _zbus_and_v(case, xdss)
            for zf in FAULT_Z_3PH_OHM:
                v3.append(_retained_3ph(Z, V0, zbase, zf))
            for zf in FAULT_Z_SLG_OHM:
                for k0 in Z0_FACTORS:
                    vs.append(_retained_slg(Z, V0, zbase, zf, k0))
        v3, vs = np.concatenate(v3), np.concatenate(vs)
        v3 = np.clip(v3[v3 <= SAG_TRIGGER_PU], 0, 1.2)
        vs = np.clip(vs[vs <= SAG_TRIGGER_PU], 0, 1.2)
        lo3, hi3 = GEN_ENVELOPES["sag_3ph"]
        lo1, hi1 = GEN_ENVELOPES["sag_slg"]
        xdss_sensitivity[f"{xdss:.2f}"] = {
            "sag_3ph_coverage": float(np.mean((v3 >= lo3) & (v3 <= hi3))),
            "sag_slg_coverage": float(np.mean((vs >= lo1) & (vs <= hi1))),
            "sag_3ph_median": float(np.median(v3)),
            "sag_slg_median": float(np.median(vs)),
        }
        print(f"  X''d={xdss:.2f} pu: 3ph coverage="
              f"{xdss_sensitivity[f'{xdss:.2f}']['sag_3ph_coverage']:.3f}, "
              f"SLG coverage={xdss_sensitivity[f'{xdss:.2f}']['sag_slg_coverage']:.3f}")

    sags_3ph = np.concatenate(sags_3ph)
    sags_slg = np.concatenate(sags_slg)

    # SFR frequency nadirs
    nadirs = []
    for H in np.linspace(2, 8, 7):
        for D in np.linspace(0.5, 2.0, 4):
            for dP in np.linspace(0.02, 0.10, 9):
                nadirs.append(_sfr_nadir(H, D, dP))
    nadirs = np.array(nadirs)
    print(f"  SFR nadirs: median={np.median(nadirs):.3f} Hz, "
          f"range=[{nadirs.min():.3f}, {nadirs.max():.3f}] Hz")

    results = {
        "per_case": per_case,
        "sag_3ph_vs_class2_envelope": _compare(sags_3ph, GEN_ENVELOPES["sag_3ph"], rng),
        "sag_slg_vs_class1_envelope": _compare(sags_slg, GEN_ENVELOPES["sag_slg"], rng),
        "freq_nadir_vs_class5_envelope": _compare(nadirs, GEN_ENVELOPES["freq_drop_underfreq_hz"], rng),
        "freq_nadir_vs_class2_envelope": _compare(nadirs, GEN_ENVELOPES["freq_drop_3ph_hz"], rng),
        "xdss_sensitivity": xdss_sensitivity,
        "assumptions": {
            "z0_over_z1": Z0_FACTORS,
            "fault_z_3ph_ohm": FAULT_Z_3PH_OHM, "fault_z_slg_ohm": FAULT_Z_SLG_OHM,
            "sag_trigger_pu": SAG_TRIGGER_PU,
            "xdss_pu_sweep": XDSS_PU_SWEEP,
            "machine_pf_assumed": PF_ASSUMED,
            "machine_representation": (
                "Synchronous machines represented by subtransient reactance "
                "shunted to ground at the machine bus before inverting Ybus. "
                "Round 1 inverted the power-flow Ybus, which has no path to "
                "ground at generator buses; that matrix is near-singular and "
                "understates network strength (case14 |Z_ii| ~2.3 pu at every "
                "bus, 45 MVA fault level on a 100 MVA base). Machine ratings "
                "are inferred as max_p_mw / PF because MATPOWER cases carry no "
                "sn_mva."
            ),
            "sfr": {"H_s": [2, 8], "D_pu": [0.5, 2.0], "dP_pu": [0.02, 0.10],
                    "droop_R": 0.05, "Tg_s": 8.0},
        },
    }

    # Raw distributions for figure generation (A7)
    np.savez(C.OUT / "a1_calibration_distributions.npz",
             sags_3ph=sags_3ph, sags_slg=sags_slg, nadirs=nadirs)
    C.save_json(results, "a1_calibration_results.json")
    return results
