"""A10 — NAS-designed lightweight classifier (answers Reviewer 2, round 2, comment 1).

R2 asks for comparison against "NAS-based lightweight model structure search" and
points at Ruan, Han, Yan and Guehmann, "Light convolutional neural network by
neural architecture search and model pruning for bearing fault diagnosis and
remaining useful life prediction", Scientific Reports 13 (2023),
doi:10.1038/s41598-023-31532-9. That paper's method is two-stage:

  1. a cell-based CNN whose cell is found by DARTS after continuous relaxation,
     with few stacked cells to keep the network small;
  2. further reduction by weights-ranking-based pruning of connections.

Both stages are reproduced here on this benchmark's 4-channel x 500-sample input.

Method
------
Search space (per edge, 8 candidate operations, the DARTS set adapted to 1D):
    none, skip_connect, sep_conv_3, sep_conv_5, dil_conv_3, dil_conv_5,
    max_pool_3, avg_pool_3

A cell is a DAG with two input nodes (the outputs of the two preceding cells) and
NODES intermediate nodes; every intermediate node sums a mixed operation over all
its predecessors. Architecture weights alpha are relaxed with a softmax and
optimised by FIRST-ORDER DARTS -- w is updated on one half of the training split
and alpha on the other, alternately, without the second-order gradient. First
order is used deliberately: it is roughly 3x cheaper and the original DARTS paper
reports it reaches comparable architectures, and the search budget is the honest
weak point of NAS in a benchmark paper, so it is reported rather than hidden.

CRITICAL: the search sees only the TRAINING partition, split 50/50. The test
partition is never touched during search, so the reported accuracy is not
contaminated by architecture selection.

The discrete cell is then derived (per node: the two strongest incoming edges,
each taking its strongest non-none operation), stacked NCELLS deep, and retrained
FROM SCRATCH on all five seeds under exactly the protocol used by every other
deep baseline (common.fit_deep_instrumented -> rebuild_experiments._train_deep).

Finally, global magnitude pruning removes 10%, 25% and 50% of the convolutional
weights in turn and the model is re-evaluated WITHOUT fine-tuning, mirroring the
retention experiment in the cited paper. A curve is reported rather than the
single 50% point, because the derived model here is an order of magnitude smaller
than an ImageNet-scale network and its retention behaviour is the informative
quantity; at 50% with no fine-tuning a model this compact degrades sharply, and
that is reported rather than tuned around.

Usage:  python -m experiments.a10_nas_search
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import common as C
import rebuild_experiments as rx

# ── Search configuration ──────────────────────────────────────────────────────
PRIMITIVES = ["none", "skip_connect", "sep_conv_3", "sep_conv_5",
              "dil_conv_3", "dil_conv_5", "max_pool_3", "avg_pool_3"]
NODES = 4                 # intermediate nodes per cell
SEARCH_C = 16             # channels during search (small: the cell, not the size)
SEARCH_LAYERS = 2
SEARCH_EPOCHS = 15
SEARCH_BATCH = 256
NCELLS = 2                # "limited stacking cells to reduce network size"
# Channels when the derived cell is retrained. C=48 gives 60,870 parameters,
# which sits between SqueezeNet-1D (57,798) and the TCN (66,934) and below the
# proposed CNN (70,950): the architectures are then compared at a comparable
# parameter budget rather than at whatever size the search happened to produce.
EVAL_C = 48
# The cited paper reports accuracy retained after removing 50% of connections.
# Reported here as a curve rather than a single point: this model is an order of
# magnitude smaller than an ImageNet-scale network, so the retention behaviour is
# the interesting quantity, not one number.
PRUNE_FRACTIONS = [0.10, 0.25, 0.50]
SEARCH_SEED = 0


# ── Operations ────────────────────────────────────────────────────────────────
def _op(name, c, stride):
    if name == "none":
        return Zero(stride)
    if name == "skip_connect":
        return nn.Identity() if stride == 1 else FactorizedReduce(c, c)
    if name == "max_pool_3":
        return nn.MaxPool1d(3, stride=stride, padding=1)
    if name == "avg_pool_3":
        return nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False)
    if name.startswith("sep_conv"):
        k = int(name.split("_")[-1])
        return SepConv(c, c, k, stride)
    if name.startswith("dil_conv"):
        k = int(name.split("_")[-1])
        return DilConv(c, c, k, stride, dilation=2)
    raise ValueError(name)


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return self.bn(self.conv(F.relu(x)))


class SepConv(nn.Module):
    """Depthwise-separable conv, ReLU-Conv-BN ordering as in DARTS."""

    def __init__(self, c_in, c_out, k, stride):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(c_in, c_in, k, stride=stride, padding=k // 2,
                      groups=c_in, bias=False),
            nn.Conv1d(c_in, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, c_in, c_out, k, stride, dilation):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(c_in, c_in, k, stride=stride, padding=pad,
                      dilation=dilation, groups=c_in, bias=False),
            nn.Conv1d(c_in, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x):
        return self.op(x)


class MixedOp(nn.Module):
    def __init__(self, c, stride):
        super().__init__()
        self.ops = nn.ModuleList([_op(p, c, stride) for p in PRIMITIVES])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# ── Search network ────────────────────────────────────────────────────────────
class SearchCell(nn.Module):
    def __init__(self, c, reduction):
        super().__init__()
        self.reduction = reduction
        self.preprocess0 = nn.Conv1d(c, c, 1, bias=False)
        self.preprocess1 = nn.Conv1d(c, c, 1, bias=False)
        self.ops = nn.ModuleList()
        for i in range(NODES):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                self.ops.append(MixedOp(c, stride))

    def forward(self, s0, s1, weights):
        states = [self.preprocess0(s0), self.preprocess1(s1)]
        offset = 0
        for _ in range(NODES):
            cur = sum(self.ops[offset + j](h, weights[offset + j])
                      for j, h in enumerate(states))
            offset += len(states)
            states.append(cur)
        return torch.cat(states[-NODES:], dim=1)


class SearchNetwork(nn.Module):
    def __init__(self, in_ch=4, n_classes=6, c=SEARCH_C, layers=SEARCH_LAYERS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c))
        self.cells = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for i in range(layers):
            cell = SearchCell(c, reduction=(i == layers - 1))
            self.cells.append(cell)
            # squeeze the NODES-way concat back to c channels
            self.reduce.append(nn.Sequential(
                nn.Conv1d(c * NODES, c, 1, bias=False), nn.BatchNorm1d(c)))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(c, n_classes))
        n_edges = sum(2 + i for i in range(NODES))
        self.alpha = nn.Parameter(1e-3 * torch.randn(n_edges, len(PRIMITIVES)))

    def arch_parameters(self):
        return [self.alpha]

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if n != "alpha"]

    def forward(self, x):
        w = F.softmax(self.alpha, dim=-1)
        s0 = s1 = self.stem(x)
        for cell, red in zip(self.cells, self.reduce):
            out = red(cell(s0, s1, w))
            if out.shape[-1] != s1.shape[-1]:
                s1 = F.adaptive_avg_pool1d(s1, out.shape[-1])
                s0 = F.adaptive_avg_pool1d(s0, out.shape[-1])
            s0, s1 = s1, out
        return self.head(s1)

    def genotype(self):
        """Per node: the two strongest incoming edges, each with its best op."""
        w = F.softmax(self.alpha, dim=-1).detach().cpu().numpy()
        none_idx = PRIMITIVES.index("none")
        gene, offset = [], 0
        for i in range(NODES):
            n_in = 2 + i
            edges = []
            for j in range(n_in):
                row = w[offset + j].copy()
                row[none_idx] = -1.0                 # never select 'none'
                k = int(row.argmax())
                edges.append((float(row[k]), j, PRIMITIVES[k]))
            edges.sort(reverse=True)
            gene.append([(op, j) for _, j, op in edges[:2]])
            offset += n_in
        return gene


# ── Derived (discrete) network ────────────────────────────────────────────────
class FixedCell(nn.Module):
    def __init__(self, c, genotype, reduction):
        super().__init__()
        self.genotype = genotype
        self.preprocess0 = nn.Conv1d(c, c, 1, bias=False)
        self.preprocess1 = nn.Conv1d(c, c, 1, bias=False)
        self.ops = nn.ModuleList()
        for node in genotype:
            for op_name, j in node:
                stride = 2 if reduction and j < 2 else 1
                self.ops.append(_op(op_name, c, stride))

    def forward(self, s0, s1):
        states = [self.preprocess0(s0), self.preprocess1(s1)]
        idx = 0
        for node in self.genotype:
            parts = []
            for _, j in node:
                h = self.ops[idx](states[j])
                idx += 1
                parts.append(h)
            n = min(p.shape[-1] for p in parts)
            parts = [p if p.shape[-1] == n else F.adaptive_avg_pool1d(p, n)
                     for p in parts]
            states = [s if s.shape[-1] == n else F.adaptive_avg_pool1d(s, n)
                      for s in states]
            states.append(sum(parts))
        return torch.cat(states[-NODES:], dim=1)


def make_nas_model(genotype, c=EVAL_C, ncells=NCELLS, in_ch=4, n_classes=6):
    """Factory returning a zero-argument class, matching the DEEP_ZOO interface."""

    class NASNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, c, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(c))
            self.cells = nn.ModuleList()
            self.reduce = nn.ModuleList()
            for i in range(ncells):
                self.cells.append(FixedCell(c, genotype, reduction=(i == ncells - 1)))
                self.reduce.append(nn.Sequential(
                    nn.Conv1d(c * NODES, c, 1, bias=False), nn.BatchNorm1d(c)))
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Dropout(0.2), nn.Linear(c, n_classes))

        def forward(self, x):
            s0 = s1 = self.stem(x)
            for cell, red in zip(self.cells, self.reduce):
                out = red(cell(s0, s1))
                if out.shape[-1] != s1.shape[-1]:
                    s1 = F.adaptive_avg_pool1d(s1, out.shape[-1])
                    s0 = F.adaptive_avg_pool1d(s0, out.shape[-1])
                s0, s1 = s1, out
            return self.head(s1)

    NASNet.__name__ = "NASNet1D"
    return NASNet


# ── Search loop ───────────────────────────────────────────────────────────────
def search(split, epochs=None, verbose=True):
    # Resolved at call time, not bound as a default, so a smoke test can lower it.
    epochs = epochs or SEARCH_EPOCHS
    torch.manual_seed(SEARCH_SEED)
    np.random.seed(SEARCH_SEED)

    X = torch.from_numpy(rx._normalize_raw(split["Rtr"])).permute(0, 2, 1).float()
    y = torch.from_numpy(split["ytr"].copy()).long()

    # Split the TRAINING partition in half: w on one half, alpha on the other.
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEARCH_SEED))
    half = n // 2
    iw, ia = perm[:half], perm[half:]
    Xw, yw = X[iw].to(C.DEVICE), y[iw].to(C.DEVICE)
    Xa, ya = X[ia].to(C.DEVICE), y[ia].to(C.DEVICE)

    model = SearchNetwork().to(C.DEVICE)
    opt_w = torch.optim.SGD(model.weight_parameters(), lr=0.025,
                            momentum=0.9, weight_decay=3e-4)
    opt_a = torch.optim.Adam(model.arch_parameters(), lr=3e-4,
                             betas=(0.5, 0.999), weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_w, epochs)
    crit = nn.CrossEntropyLoss()

    nb = max(1, half // SEARCH_BATCH)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        pw = torch.randperm(len(yw), device=C.DEVICE)
        pa = torch.randperm(len(ya), device=C.DEVICE)
        for b in range(nb):
            sa = pa[b * SEARCH_BATCH:(b + 1) * SEARCH_BATCH]
            if len(sa):
                opt_a.zero_grad()
                crit(model(Xa[sa]), ya[sa]).backward()
                opt_a.step()
            sw = pw[b * SEARCH_BATCH:(b + 1) * SEARCH_BATCH]
            if len(sw):
                opt_w.zero_grad()
                crit(model(Xw[sw]), yw[sw]).backward()
                nn.utils.clip_grad_norm_(model.weight_parameters(), 5.0)
                opt_w.step()
        sched.step()
        if verbose:
            model.eval()
            with torch.no_grad():
                acc = (model(Xa[:4096]).argmax(1) == ya[:4096]).float().mean().item()
            print(f"    epoch {ep + 1:2d}/{epochs}  alpha-split acc={acc:.4f}")
    search_s = time.time() - t0
    return model.genotype(), search_s


# ── Pruning ───────────────────────────────────────────────────────────────────
def magnitude_prune(model, fraction):
    """Global magnitude pruning of Conv1d weights (weights-ranking pruning).

    Returns (n_removed, saved_state) so the caller can restore the dense weights
    and sweep several fractions from the same trained model.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    if not convs:
        return 0, None
    saved = [m.weight.detach().clone() for m in convs]
    allw = torch.cat([m.weight.detach().abs().flatten() for m in convs])
    k = int(fraction * allw.numel())
    if k < 1:
        return 0, saved
    thresh = torch.kthvalue(allw, k).values
    removed = 0
    with torch.no_grad():
        for m in convs:
            mask = m.weight.abs() > thresh
            removed += int((~mask).sum())
            m.weight.mul_(mask)
    return removed, saved


def restore(model, saved):
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        for m, w in zip(convs, saved):
            m.weight.copy_(w)


# ── Driver ────────────────────────────────────────────────────────────────────
def run(X_base, X_raw_base, y_base):
    print("\n" + "=" * 70)
    print("A10 — NAS CELL SEARCH + PRUNING")
    print("=" * 70)

    # Search once, on the first seed's training partition only.
    split0 = C.seed_split(X_base, X_raw_base, y_base, C.SEEDS[0])
    print(f"  searching (first-order DARTS, {SEARCH_EPOCHS} epochs, "
          f"{NODES} nodes, {len(PRIMITIVES)} ops)")
    genotype, search_s = search(split0)
    print(f"  search completed in {search_s / 60:.1f} min")
    print("  derived cell:")
    for i, node in enumerate(genotype):
        print(f"    node {i}: " + ", ".join(f"{op}<-{j}" for op, j in node))

    NASNet = make_nas_model(genotype)
    n_params = sum(p.numel() for p in NASNet().parameters())
    print(f"  derived model: {n_params:,} parameters, {NCELLS} cells, C={EVAL_C}")

    # Retrain from scratch on every seed, identical protocol to all other models.
    accs, f1s, train_s = [], [], []
    pruned = {f"{f:.2f}": [] for f in PRUNE_FRACTIONS}
    for seed in C.SEEDS:
        split = C.seed_split(X_base, X_raw_base, y_base, seed)
        mdl, res = C.fit_deep_instrumented(NASNet, split)
        accs.append(res["accuracy"])
        f1s.append(res["f1_macro"])
        train_s.append(res["train_s"])

        msg = []
        for frac in PRUNE_FRACTIONS:
            removed, saved = magnitude_prune(mdl, frac)
            pres = rx._eval_deep(mdl, split["Rte"], split["yte"])
            pruned[f"{frac:.2f}"].append(pres["accuracy"])
            msg.append(f"{frac:.0%}->{pres['accuracy']:.4f}")
            restore(mdl, saved)
        print(f"  seed {seed}: acc={res['accuracy']:.4f}  "
              f"prune[{', '.join(msg)}]  [{res['train_s']:.0f}s]")

        ckpt = C.OUT / "checkpoints"
        ckpt.mkdir(exist_ok=True)
        torch.save(mdl.state_dict(), ckpt / f"NAS-1D_seed{seed}.pt")

    accs, f1s = np.array(accs), np.array(f1s)
    prune_summary = {}
    for frac, vals in pruned.items():
        v = np.array(vals)
        prune_summary[frac] = {
            "accuracy_mean": float(v.mean()),
            "accuracy_std": float(v.std(ddof=1)),
            "retention": float(v.mean() / accs.mean()),
        }
    results = {
        "genotype": [[list(e) for e in node] for node in genotype],
        "search": {
            "method": "first-order DARTS, continuous relaxation",
            "primitives": PRIMITIVES, "nodes": NODES,
            "epochs": SEARCH_EPOCHS, "channels": SEARCH_C,
            "search_wall_s": search_s,
            "data": "training partition of seed "
                    f"{C.SEEDS[0]} only, split 50/50 for w and alpha; "
                    "the test partition is never used during search",
        },
        "model": {"n_params": int(n_params), "n_cells": NCELLS, "channels": EVAL_C},
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std(ddof=1)),
        "f1_macro_mean": float(f1s.mean()), "f1_macro_std": float(f1s.std(ddof=1)),
        "train_s_mean": float(np.mean(train_s)),
        "pruned": prune_summary,
        "pruning_note": ("global magnitude pruning of Conv1d weights, "
                         "NO fine-tuning after pruning"),
        "per_seed": {str(s): {"accuracy": float(a)} for s, a in zip(C.SEEDS, accs)},
    }
    print(f"\n  NAS-1D: acc={accs.mean():.4f}+-{accs.std(ddof=1):.4f}  "
          f"({n_params:,} params, search {search_s / 60:.1f} min)")
    for frac, s in prune_summary.items():
        print(f"    pruned {float(frac):.0%}: acc={s['accuracy_mean']:.4f}"
              f"+-{s['accuracy_std']:.4f}  retention={s['retention']:.3f}")

    C.save_json(results, "a10_nas_results.json")
    return results


if __name__ == "__main__":
    import rebuild_experiments as _rx
    print("Generating shared 50,000-event dataset (global seed 0) ...")
    Xb, Xr, yb = _rx.generate_dataset(50_000)
    run(Xb, Xr, yb)
