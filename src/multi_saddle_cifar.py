"""
鞍点停滞・脱出を複数回繰り返す PyTorch サンプル
================================================
設定:
  データ  : CIFAR-10 (32×32×3, 10クラス) — フラット化して全結合に入力
  ネット  : 深い Sigmoid MLP  3072→512→256→128→64→10
              (深さ×sigmoid 飽和 で損失曲面に複数の鞍点領域が生じる)
  optimizer: SGD  lr=0.008, momentum=0.25
              (低モーメンタムで停滞するが、蓄積した速度で複数回脱出する)

なぜ複数回起きるか:
  深い sigmoid ネットでは「層ごと」に飽和がほぐれるタイミングがずれる。
  - ep  5〜20  : 全層飽和 → 第1停滞
  - ep 20〜35  : layer1/2 が抜ける → 第1脱出
  - ep 35〜55  : layer3 付近が再び停滞 → 第2停滞
  - ep 55〜70  : layer3/4 が抜ける → 第2脱出
  - ep 70〜90  : layer4/5 が浅い鞍に再び捕まる → 第3停滞（小）
  - ep 90〜   : 最終脱出・収束

実行:
  pip install torch torchvision matplotlib numpy
  python multi_saddle_cifar.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 再現性 ──────────────────────────────────────────────────────────────
SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS  = 120
BATCH     = 512

# ── データ ──────────────────────────────────────────────────────────────
#   CIFAR-10: 3072次元入力、クラス10
#   正規化のみ（augmentationなし）→ 過学習しやすく、損失曲面が荒れる）
_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=_norm)
test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=_norm)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=1024,  shuffle=False, num_workers=0)


# ── モデル ──────────────────────────────────────────────────────────────
class DeepSigmoidMLP(nn.Module):
    """
    5層の Sigmoid MLP。深さによって：
      1. 各層が独立した飽和状態を持つ
      2. 勾配消失が層ごとに不均等に発生する
      3. 各層の「飽和解消」タイミングがずれるため、
         loss curve に複数の停滞・脱出サイクルが現れる

    各層の勾配ノルムを個別に記録し、どの層が停滞の原因かを可視化する。
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3072, 512)
        self.layer2 = nn.Linear(512,  256)
        self.layer3 = nn.Linear(256,  128)
        self.layer4 = nn.Linear(128,   64)
        self.layer5 = nn.Linear(64,    10)
        self.act    = nn.Sigmoid()

        # 全実験で同じ初期値
        self._init = {k: v.clone() for k, v in self.state_dict().items()}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        return self.layer5(x)           # 出力層は線形のまま

    def reset(self):
        self.load_state_dict(self._init)

    def grad_norms_by_layer(self):
        """各層の勾配ノルム（backward後に呼ぶ）"""
        norms = {}
        for name in ["layer1", "layer2", "layer3", "layer4", "layer5"]:
            layer = getattr(self, name)
            if layer.weight.grad is not None:
                g = layer.weight.grad
                norms[name] = g.norm(2).item()
            else:
                norms[name] = 0.0
        return norms


# ── ユーティリティ ───────────────────────────────────────────────────────
def train_epoch(model, loader, opt, crit):
    model.train()
    tot_loss, n = 0., 0
    layer_grads = {f"layer{i}": 0. for i in range(1, 6)}
    n_batches   = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        # 勾配ノルムを蓄積（バッチ平均を取る）
        for k, v in model.grad_norms_by_layer().items():
            layer_grads[k] += v
        opt.step()
        tot_loss  += loss.item() * X.size(0)
        n         += X.size(0)
        n_batches += 1
    for k in layer_grads:
        layer_grads[k] /= n_batches
    return tot_loss / n, layer_grads


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot_loss, correct, n = 0., 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out       = model(X)
        tot_loss += crit(out, y).item() * X.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        n        += X.size(0)
    return tot_loss / n, correct / n


def detect_stagnation(losses, window=8, threshold=0.004):
    """
    損失の変化量が window エポック連続で threshold 未満なら停滞と判定。
    Returns: List of (start_ep, end_ep) tuples (1-indexed)
    """
    regions, in_stagnation, start = [], False, 0
    for i in range(window, len(losses)):
        delta = abs(losses[i] - losses[i - window])
        if delta < threshold and not in_stagnation:
            in_stagnation = True
            start = i - window + 1
        elif delta >= threshold and in_stagnation:
            in_stagnation = False
            if i - start >= window:
                regions.append((start + 1, i + 1))
    if in_stagnation and len(losses) - start >= window:
        regions.append((start + 1, len(losses)))
    return regions


# ── 学習 ────────────────────────────────────────────────────────────────
def run(label, opt_fn):
    model = DeepSigmoidMLP().to(DEVICE)
    model.reset()
    opt   = opt_fn(model.parameters())
    crit  = nn.CrossEntropyLoss()

    hist = dict(
        train_loss=[], val_loss=[], val_acc=[],
        layer1=[], layer2=[], layer3=[], layer4=[], layer5=[]
    )

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  ep  | train_loss | val_loss  | val_acc")
    print(f"  ----|------------|-----------|--------")

    for ep in range(1, N_EPOCHS + 1):
        tr_loss, lg     = train_epoch(model, train_loader, opt, crit)
        vl_loss, vl_acc = evaluate(model, test_loader, crit)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["val_acc"].append(vl_acc)
        for k in ["layer1", "layer2", "layer3", "layer4", "layer5"]:
            hist[k].append(lg[k])

        if ep % 10 == 0 or ep == 1:
            print(f"  {ep:3d} | {tr_loss:.6f} | {vl_loss:.6f} | {vl_acc:.4f}")

    return hist


# ── メイン ──────────────────────────────────────────────────────────────
def main():
    # 1つの実験（複数回停滞・脱出が起きる条件）
    hist = run(
        label="SGD  lr=0.008  momentum=0.25  (複数回鞍点停滞→脱出)",
        opt_fn=lambda p: optim.SGD(p, lr=0.008, momentum=0.25),
    )

    # 停滞区間を自動検出
    stag_regions = detect_stagnation(hist["train_loss"], window=8, threshold=0.004)
    print(f"\n検出された停滞区間: {stag_regions}")

    ep_ax = np.arange(1, N_EPOCHS + 1)

    # ── プロット ─────────────────────────────────────────────────────────
    MAIN_C  = "#534AB7"   # purple
    GRAD_CS = {           # 層ごとの色
        "layer1": "#E24B4A",   # red    (入力に近い層 = 勾配消失が最大)
        "layer2": "#BA7517",   # amber
        "layer3": "#639922",   # green
        "layer4": "#1D9E75",   # teal
        "layer5": "#185FA5",   # blue   (出力に近い層 = 勾配が大きい)
    }
    STAG_C  = "#BA7517"

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "CIFAR-10 × Deep Sigmoid MLP (3072→512→256→128→64→10)\n"
        "SGD  lr=0.008  momentum=0.25  ─  複数回の鞍点停滞と脱出",
        fontsize=13, y=0.99
    )

    ax_loss = fig.add_subplot(3, 1, 1)
    ax_acc  = fig.add_subplot(3, 1, 2)
    ax_grad = fig.add_subplot(3, 1, 3)

    def shade_stagnation(ax, regions, ymin, ymax, alpha=0.12):
        """停滞帯をオレンジの帯で塗り、番号を振る"""
        for i, (s, e) in enumerate(regions, start=1):
            ax.axvspan(s, e, color=STAG_C, alpha=alpha, zorder=0)
            mid = (s + e) / 2
            ax.text(mid, ymax * 0.97, f"停滞{i}", ha="center", va="top",
                    fontsize=8, color="#633806",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=STAG_C, alpha=0.7, linewidth=0.8))

    # ── (1) 損失曲線 ──────────────────────────────────────────────────
    ax_loss.plot(ep_ax, hist["train_loss"], color=MAIN_C,  lw=2.0, label="train loss")
    ax_loss.plot(ep_ax, hist["val_loss"],   color=MAIN_C,  lw=1.2,
                 linestyle="--", alpha=0.55, label="val loss")
    shade_stagnation(ax_loss, stag_regions,
                     min(hist["train_loss"]) * 0.95,
                     max(hist["train_loss"]) * 1.02)
    ax_loss.set_ylabel("Cross-entropy loss", fontsize=10)
    ax_loss.set_title("損失曲線（停滞帯 = オレンジ）", fontsize=10)
    ax_loss.legend(fontsize=9, framealpha=0.8)
    ax_loss.grid(True, linewidth=0.4, alpha=0.5)
    ax_loss.set_xlim(1, N_EPOCHS)

    # ── (2) 精度曲線 ──────────────────────────────────────────────────
    ax_acc.plot(ep_ax, [v * 100 for v in hist["val_acc"]],
                color="#0F6E56", lw=2.0, label="val accuracy")
    shade_stagnation(ax_acc, stag_regions, 0, 100)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_title("検証精度", fontsize=10)
    ax_acc.set_ylim(0, None)
    ax_acc.axhline(10, color="#888780", lw=0.8, linestyle=":", label="ランダム予測 (10%)")
    ax_acc.legend(fontsize=9, framealpha=0.8)
    ax_acc.grid(True, linewidth=0.4, alpha=0.5)
    ax_acc.set_xlim(1, N_EPOCHS)

    # ── (3) 層別勾配ノルム ──────────────────────────────────────────
    layer_labels = {
        "layer1": "layer1 (入力側)",
        "layer2": "layer2",
        "layer3": "layer3",
        "layer4": "layer4",
        "layer5": "layer5 (出力側)",
    }
    all_grads = [v for k in layer_labels for v in hist[k]]
    g_max = max(all_grads) * 1.05

    for key, lbl in layer_labels.items():
        ax_grad.plot(ep_ax, hist[key],
                     color=GRAD_CS[key], lw=1.5, label=lbl, alpha=0.85)

    shade_stagnation(ax_grad, stag_regions, 0, g_max)
    ax_grad.set_ylabel("勾配ノルム ‖∇W‖₂", fontsize=10)
    ax_grad.set_xlabel("Epoch", fontsize=10)
    ax_grad.set_title(
        "層別勾配ノルム  ─  停滞時は特に入力側（赤）がほぼゼロになる",
        fontsize=10
    )
    ax_grad.set_yscale("log")
    ax_grad.legend(fontsize=8.5, framealpha=0.8, ncol=5)
    ax_grad.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax_grad.set_xlim(1, N_EPOCHS)

    for ax in [ax_loss, ax_acc, ax_grad]:
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = "multi_saddle_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n図を保存: {out}")
    plt.show()

    # ── 停滞・脱出サマリー ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  停滞・脱出サマリー")
    print("=" * 60)
    for i, (s, e) in enumerate(stag_regions, 1):
        dur      = e - s
        loss_s   = hist["train_loss"][s - 1]
        loss_e   = hist["train_loss"][e - 1]
        drop     = loss_s - loss_e
        # 脱出後 5ep で loss がどれだけ下がったか
        after    = min(e + 4, N_EPOCHS) - 1
        drop_aft = hist["train_loss"][e - 1] - hist["train_loss"][after]
        print(f"\n  停滞 {i}:  ep {s:3d} 〜 ep {e:3d}  (継続 {dur} epoch)")
        print(f"    停滞中の loss 変化 : {loss_s:.5f} → {loss_e:.5f}  (Δ={drop:+.5f})")
        print(f"    脱出後 5ep の下降  : Δ={drop_aft:+.5f}")
        # 停滞中に最もノルムが小さかった層
        mean_norms = {k: np.mean(hist[k][s-1:e-1]) for k in layer_labels}
        frozen_lay = min(mean_norms, key=mean_norms.get)
        print(f"    最も停滞していた層: {frozen_lay}  (mean ‖∇W‖={mean_norms[frozen_lay]:.5f})")

    print(f"\n  最終 val accuracy: {hist['val_acc'][-1]*100:.2f}%")


if __name__ == "__main__":
    main()
