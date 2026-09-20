"""A6 — inference cost for every deep baseline (Reviewer 1.7, Reviewer 2 round 2).

Round 1 reported parameters, CPU/GPU latency and FLOPs for the proposed 1D-CNN
and the TCN only, from an ad-hoc script that was never committed; the numbers
lived in outputs/tables/a6_local_costs.json with no code behind them. Reviewer 2
now asks for a comparison against lightweight DNN architectures, and the whole
point of SqueezeNet and MobileNetV2 is cost, not accuracy -- so every deep model
in DEEP_ZOO is measured here, from one script, on one machine.

Reported per model:
  params                parameter count
  flops_m               multiply-accumulates x 2, batch size 1, in millions
  cpu_ms                eager-mode CPU latency, batch 1
  torchscript_cpu_ms    after torch.jit.script/trace
  gpu_ms                CUDA latency, batch 1 (synchronised)

FLOPs convention: 2 x MAC, which reproduces the round-1 figure of 12.2 MFLOPs
for the proposed CNN (6,114,304 MACs). Conv1d and Linear dominate; BatchNorm,
activations and pooling are not counted, as is conventional.

Batch-1 GPU latency is reported with the round-1 caveat retained: at this model
size the measurement is dominated by launch overhead, not compute, so it is not
a meaningful speed ranking -- CPU latency is the deployment-relevant number for
edge protection hardware.

Usage:  python -m experiments.a6_compute_costs
"""

import json
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from experiments import common as C

WARMUP = 20
REPEATS = 200
INPUT_SHAPE = (1, 4, 500)


# ── FLOPs ─────────────────────────────────────────────────────────────────────
def count_flops(model, shape=INPUT_SHAPE):
    """Multiply-accumulates x 2 for one forward pass at batch size 1."""
    macs = [0]
    handles = []

    def conv_hook(mod, inp, out):
        # out: (B, C_out, L_out); each output element costs C_in/groups * k MACs
        k = mod.kernel_size[0]
        c_in = mod.in_channels // mod.groups
        macs[0] += out.shape[-1] * mod.out_channels * c_in * k

    def linear_hook(mod, inp, out):
        macs[0] += mod.in_features * mod.out_features

    def lstm_hook(mod, inp, out):
        # 4 gates, input and hidden contributions, per direction per timestep
        seq = inp[0].shape[1] if mod.batch_first else inp[0].shape[0]
        dirs = 2 if mod.bidirectional else 1
        per_step = 4 * (mod.input_size * mod.hidden_size
                        + mod.hidden_size * mod.hidden_size)
        macs[0] += seq * dirs * per_step * mod.num_layers

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.LSTM):
            handles.append(m.register_forward_hook(lstm_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*shape))
    for h in handles:
        h.remove()
    return 2 * macs[0]


# ── Latency ───────────────────────────────────────────────────────────────────
def _time(fn, x, device):
    for _ in range(WARMUP):
        fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000.0)     # median, not mean: robust to jitter


def measure(model_cls):
    out = {}
    cpu = torch.device("cpu")

    m = model_cls().to(cpu).eval()
    out["params"] = int(sum(p.numel() for p in m.parameters()))
    out["flops_m"] = round(count_flops(m) / 1e6, 1)

    x = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        out["cpu_ms"] = _time(lambda z: m(z), x, cpu)

        try:
            sm = torch.jit.trace(m, x)
            sm = torch.jit.optimize_for_inference(sm)
            out["torchscript_cpu_ms"] = _time(lambda z: sm(z), x, cpu)
        except Exception as e:                       # noqa: BLE001
            out["torchscript_cpu_ms"] = None
            out["torchscript_error"] = f"{type(e).__name__}: {e}"

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            mg = model_cls().to(dev).eval()
            xg = x.to(dev)
            out["gpu_ms"] = _time(lambda z: mg(z), xg, dev)
        else:
            out["gpu_ms"] = None
    return out


def run(X_base=None, X_raw_base=None, y_base=None):
    print("\n" + "=" * 70)
    print("A6 — INFERENCE COST (batch 1)")
    print("=" * 70)

    results = {
        "_provenance": {
            "host": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "warmup": WARMUP, "repeats": REPEATS, "statistic": "median",
            "flops_convention": "2 x MAC, Conv1d + Linear + LSTM, batch 1",
        },
        "models": {},
    }

    print(f"{'model':<22}{'params':>10}{'MFLOPs':>9}{'CPU ms':>9}"
          f"{'TS ms':>9}{'GPU ms':>9}")
    print("-" * 68)
    for name, cls in C.DEEP_ZOO:
        r = measure(cls)
        results["models"][name] = r
        ts = f"{r['torchscript_cpu_ms']:.3f}" if r["torchscript_cpu_ms"] else "  n/a"
        gpu = f"{r['gpu_ms']:.3f}" if r["gpu_ms"] else "  n/a"
        print(f"{name:<22}{r['params']:>10,}{r['flops_m']:>9.1f}"
              f"{r['cpu_ms']:>9.3f}{ts:>9}{gpu:>9}")

    C.save_json(results, "a6_compute_costs.json")
    return results


if __name__ == "__main__":
    run()
