"""Lightweight DNN baselines — 1D ports of SqueezeNet and MobileNetV2.

Answers Reviewer 2 (round 2, comment 1): compare against typical lightweight DNN
models such as SqueezeNet and MobileNet rather than classic machine-learning
methods. Round 1 read "lightweight models" as XGBoost/LightGBM/SVM/kNN/MLP/TCN;
the reviewer meant lightweight neural architectures.

Both originals are 2D image classifiers and cannot ingest this benchmark's input,
which is a 4-channel x 500-sample multivariate waveform (three phase voltages plus
frequency). The defining structural ideas port to 1D one-for-one and are kept
faithful:

  SqueezeNet   Fire module: a 1x1 "squeeze" convolution that reduces channel
               count, feeding two parallel "expand" convolutions (1x1 and 3x3)
               whose outputs are concatenated. Classifier is a 1x1 convolution
               plus global average pooling -- no fully connected layer.
               (Iandola et al., 2016)

  MobileNetV2  Inverted residual with linear bottleneck: 1x1 pointwise expansion
               by factor t, depthwise convolution (groups = channels), 1x1
               pointwise projection with NO activation, and a residual connection
               when stride is 1 and channel count is unchanged. ReLU6 throughout.
               (Sandler et al., 2018)

Sizing note: the expansion factors and channel widths below are the originals'.
Depth is reduced because the input is 500 samples on 4 channels, not
224x224x3 -- an unreduced ImageNet stack would downsample the sequence to nothing.
Neither model is tuned to match the proposed CNN's parameter count; they are
reported at whatever size their own design rules produce, which is the only
comparison that means anything.

Interface matches rebuild_experiments.FaultCNN1D exactly, so these are drop-in
entries for common.DEEP_ZOO and are picked up unchanged by the A2 robustness and
A5 SHAP stages, which load checkpoints by filename.

Smoke test:  python -m experiments.lightweight
"""

import torch
import torch.nn as nn


# ── SqueezeNet-1D ─────────────────────────────────────────────────────────────
class Fire1D(nn.Module):
    """Fire module: squeeze 1x1 -> (expand 1x1 || expand 3x3), concatenated."""

    def __init__(self, c_in, c_squeeze, c_expand_1x1, c_expand_3x3):
        super().__init__()
        self.squeeze = nn.Conv1d(c_in, c_squeeze, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm1d(c_squeeze)
        self.expand_1x1 = nn.Conv1d(c_squeeze, c_expand_1x1, kernel_size=1)
        self.expand_3x3 = nn.Conv1d(c_squeeze, c_expand_3x3, kernel_size=3, padding=1)
        self.expand_bn = nn.BatchNorm1d(c_expand_1x1 + c_expand_3x3)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        s = self.act(self.squeeze_bn(self.squeeze(x)))
        e = torch.cat([self.expand_1x1(s), self.expand_3x3(s)], dim=1)
        return self.act(self.expand_bn(e))


class SqueezeNet1D(nn.Module):
    """SqueezeNet v1.1 adapted to 4-channel, 500-sample waveforms.

    Input : (batch, 4, 500)
    Output: (batch, 6)
    """

    def __init__(self, in_channels=4, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(48), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),      # 500 -> 250 -> 125
            Fire1D(48, 16, 32, 32),
            Fire1D(64, 16, 32, 32),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),      # 125 -> 63
            Fire1D(64, 32, 64, 64),
            Fire1D(128, 32, 64, 64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),      # 63 -> 32
            Fire1D(128, 48, 96, 96),
        )
        # SqueezeNet's classifier is a 1x1 conv + global pool, not a linear layer.
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(192, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── MobileNetV2-1D ────────────────────────────────────────────────────────────
class InvertedResidual1D(nn.Module):
    """Expand 1x1 -> depthwise conv -> project 1x1 (linear), with residual."""

    def __init__(self, c_in, c_out, stride, expand_ratio, kernel_size=3):
        super().__init__()
        self.use_residual = stride == 1 and c_in == c_out
        hidden = int(round(c_in * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv1d(c_in, hidden, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden), nn.ReLU6(inplace=True),
            ]
        layers += [
            # depthwise: one filter per channel
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU6(inplace=True),
            # linear bottleneck: projection carries NO activation
            nn.Conv1d(hidden, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class MobileNetV21D(nn.Module):
    """MobileNetV2 adapted to 4-channel, 500-sample waveforms.

    Input : (batch, 4, 500)
    Output: (batch, 6)

    Config rows are (expand_ratio t, out_channels c, repeats n, stride s), the
    original's notation, truncated to four stages for a 500-sample sequence.
    """

    CONFIG = [
        # t, c,  n, s
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 2, 2),
        (6, 64, 2, 2),
    ]

    def __init__(self, in_channels=4, num_classes=6, width_mult=1.0):
        super().__init__()
        c = int(32 * width_mult)
        layers = [
            nn.Conv1d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c), nn.ReLU6(inplace=True),
        ]
        for t, c_out, n, s in self.CONFIG:
            c_out = int(c_out * width_mult)
            for i in range(n):
                layers.append(InvertedResidual1D(c, c_out,
                                                 stride=s if i == 0 else 1,
                                                 expand_ratio=t))
                c = c_out
        c_last = int(256 * width_mult)
        layers += [
            nn.Conv1d(c, c_last, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_last), nn.ReLU6(inplace=True),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(c_last, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Smoke test ────────────────────────────────────────────────────────────────
def _count(model):
    return sum(p.numel() for p in model.parameters())


def _smoke():
    import rebuild_experiments as rx

    x = torch.randn(2, 4, 500)
    print(f"{'model':<22}{'params':>10}  {'output':>12}   forward")
    print("-" * 60)
    reference = [("1D-CNN (proposed)", rx.FaultCNN1D),
                 ("Shallow CNN", rx.ShallowCNN)]
    new = [("SqueezeNet-1D", SqueezeNet1D),
           ("MobileNetV2-1D", MobileNetV21D)]
    for name, cls in reference + new:
        m = cls().eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 6), f"{name}: bad output shape {tuple(y.shape)}"
        assert torch.isfinite(y).all(), f"{name}: non-finite logits"
        print(f"{name:<22}{_count(m):>10,}  {str(tuple(y.shape)):>12}   ok")
    print("\nall models accept (B, 4, 500) and emit (B, 6)")


if __name__ == "__main__":
    _smoke()
