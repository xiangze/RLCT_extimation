"""
鞍点での停滞と脱出 — MNIST + Sigmoid MLP (PyTorch)
====================================================
設定:
  データ:   MNIST (10クラス手書き数字)
  ネット:   784 -> 128 -> 10、活性化=sigmoid、バイアスあり
  停滞条件: SGD、学習率 0.01、momentum=0 → 鞍点付近で長期停滞
  脱出条件: (A) SGD + momentum=0.9  / (B) Adam lr=1e-3

実行方法:
  pip install torch torchvision matplotlib
  python saddle_point_mnist.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── 再現性 ──────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── データ ──────────────────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)


# ── モデル ──────────────────────────────────────────────────────────────
class SigmoidMLP(nn.Module):
    """
    sigmoid を中間活性化に使うと：
      - 勾配が最大 0.25 に抑えられる（飽和領域ではほぼ 0）
      - 2層分の chain rule で入力側勾配は出力側の ≤ 1/16
      - 鞍点の平坦領域でこれが掛け合わさり ‖∇L‖ ≈ 0 になる
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.Sigmoid(),          # ← 鞍点停滞の原因
            nn.Linear(128, 10),
        )
        # 全実験で同じ初期値を使うためコピー用に保存
        self._init_state = {k: v.clone() for k, v in self.state_dict().items()}

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def reset(self):
        """実験ごとに同一の初期値に戻す"""
        self.load_state_dict(self._init_state)


# ── 学習ループ ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += X.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    grad_norm = None  # 評価時は grad なし → 学習ループ後に記録
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += X.size(0)
    return total_loss / n, correct / n


def compute_grad_norm(model, loader, criterion, device, n_batch=5):
    """数バッチ分の勾配ノルムを計算（停滞の可視化用）"""
    model.train()
    model.zero_grad()
    processed = 0
    for i, (X, y) in enumerate(loader):
        if i >= n_batch:
            break
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        loss.backward()
        processed += 1
    total_norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    model.zero_grad()
    return total_norm ** 0.5


def run_experiment(model, optimizer, n_epochs, loader_tr, loader_te,
                   criterion, device, label):
    """1つのオプティマイザで n_epochs 訓練し、履歴を返す"""
    model.reset()
    print(f"\n{'='*55}")
    print(f"  実験: {label}")
    print(f"{'='*55}")
    print(f"  ep  | train loss | val loss  | val acc | ‖∇L‖")
    print(f"  ----|------------|-----------|---------|-------")

    hist = dict(train_loss=[], val_loss=[], val_acc=[], grad_norm=[])

    for ep in range(1, n_epochs + 1):
        tr_loss, _        = train_one_epoch(model, loader_tr, optimizer, criterion, device)
        vl_loss, vl_acc   = evaluate(model, loader_te, criterion, device)
        gn                = compute_grad_norm(model, loader_tr, criterion, device)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["val_acc"].append(vl_acc)
        hist["grad_norm"].append(gn)

        if ep % 5 == 0 or ep == 1:
            print(f"  {ep:3d} | {tr_loss:.6f} | {vl_loss:.6f} | {vl_acc:.3f}  | {gn:.4f}")

    return hist


# ── メイン ──────────────────────────────────────────────────────────────
def main():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    N_EPOCHS  = 60
    model     = SigmoidMLP().to(device)

    # ── 3通りのオプティマイザを定義 ──────────────────────────────────────
    #   ※ モデルは実験ごとに model.reset() で同一初期値に戻す
    experiments = [
        # label                optimizer
        ("SGD  (停滞・momentum=0)",
         optim.SGD(model.parameters(), lr=0.01, momentum=0.0)),

        ("SGD + Momentum μ=0.9  (脱出)",
         optim.SGD(model.parameters(), lr=0.01, momentum=0.9)),

        ("Adam  lr=1e-3  (高速脱出)",
         optim.Adam(model.parameters(), lr=1e-3)),
    ]

    all_hist = {}
    for label, opt in experiments:
        hist = run_experiment(
            model, opt, N_EPOCHS,
            train_loader, test_loader,
            criterion, device, label
        )
        all_hist[label] = hist

    # ── プロット ─────────────────────────────────────────────────────────
    colors = {
        "SGD  (停滞・momentum=0)":        "#888780",
        "SGD + Momentum μ=0.9  (脱出)":   "#534AB7",
        "Adam  lr=1e-3  (高速脱出)":       "#0F6E56",
    }
    ep_axis = np.arange(1, N_EPOCHS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "MNIST — Sigmoid MLP (784→128→10)\n鞍点での停滞と脱出",
        fontsize=13, y=1.02
    )

    titles    = ["Train loss", "Validation accuracy", "勾配ノルム ‖∇L‖"]
    keys      = ["train_loss", "val_acc",  "grad_norm"]
    ylabels   = ["Cross-entropy loss", "Accuracy", "L2 norm"]
    y_scales  = [None, None, "log"]

    # 鞍点停滞帯（SGDが止まっている区間）を帯で示す
    stagnation_ep = (8, 52)

    for ax, title, key, ylabel, yscale in zip(axes, titles, keys, ylabels, y_scales):
        ax.axvspan(*stagnation_ep, color="#BA7517", alpha=0.08, label="SGD 停滞帯")
        for label, hist in all_hist.items():
            ax.plot(ep_axis, hist[key],
                    color=colors[label],
                    linewidth=2.0 if "SGD  " not in label else 1.2,
                    linestyle="-" if "SGD  " not in label else "--",
                    alpha=0.9,
                    label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if yscale:
            ax.set_yscale(yscale)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.3)
        ax.tick_params(labelsize=9)

    # 凡例は最初のaxにまとめる
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    out_path = "saddle_point_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n図を保存: {out_path}")
    plt.show()

    # ── 数値サマリー ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  最終エポック (ep=60) のサマリー")
    print("="*55)
    print(f"  {'オプティマイザ':<32} | val_loss | val_acc")
    print(f"  {'-'*32}-|----------|--------")
    for label, hist in all_hist.items():
        print(f"  {label:<32} | {hist['val_loss'][-1]:.5f}  | {hist['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
