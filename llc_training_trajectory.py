"""
Track Local Learning Coefficient (LLC) along SGD training trajectories.

- Datasets: toy1d, MNIST (easily extendable)
- Models: linear, MLP with ReLU (easily extendable)
- Optimizers: SGD, Adam
- LLC modes:
    - "none": no LLC computation
    - "precomputed": load LLC from a json file (epoch -> value)
    - "online": run a small SGLD chain around current weights to estimate LLC
        (rough approximation inspired by Lau et al. local_coeff_computation.py)

This script is intentionally modular so you can:
  - plug in your own models/datasets (e.g. from suswei/RLCT)
  - replace `estimate_llc_sgld` by the exact implementation from
    `local_coeff_computation.py` if you wish.

Usage example:
    python llc_training_experiment.py \\
        --dataset toy1d \\
        --model mlp_relu \\
        --optimizer sgd \\
        --llc_mode online \\
        --output_dir ./results_toy1d_mlp

"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt


# =========================================================
# 1. Utility: seeding, device
# =========================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================
# 2. Datasets
#    - toy1d: simple y = sin(x) + noise regression
#    - mnist: 0 vs 1 binary classification (for speed)
# =========================================================

def make_toy1d_dataset(n_train: int = 256, n_test: int = 256) -> Tuple[Dataset, Dataset]:
    """
    Simple 1D regression: x ~ Uniform[-3, 3], y = sin(x) + Normal(0, 0.1).
    """
    x_train = torch.rand(n_train, 1) * 6.0 - 3.0
    y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)
    x_test = torch.rand(n_test, 1) * 6.0 - 3.0
    y_test = torch.sin(x_test) + 0.1 * torch.randn_like(x_test)
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


def make_mnist_binary_dataset(root: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Binary MNIST: classify 0 vs 1. (To keep things light.)

    Requires torchvision. If you don't want torchvision, you can replace
    this with your own dataset loader.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError(
            "torchvision is required for MNIST. "
            "Install it or replace make_mnist_binary_dataset() with your own loader."
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(root, train=False, download=True, transform=transform)

    def filter_01(dset):
        xs, ys = [], []
        for x, y in dset:
            if y in (0, 1):
                xs.append(x.view(-1))  # flatten
                ys.append(float(y))
        xs = torch.stack(xs)
        ys = torch.tensor(ys).unsqueeze(1)  # shape (N,1)
        return TensorDataset(xs, ys)

    train = filter_01(train_full)
    test = filter_01(test_full)
    return train, test


def get_dataset(name: str, data_root: str = "./data") -> Tuple[Dataset, Dataset, str]:
    if name == "toy1d":
        train, test = make_toy1d_dataset()
        task_type = "regression"
    elif name == "mnist":
        train, test = make_mnist_binary_dataset(root=data_root)
        task_type = "binary_classification"
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test, task_type


# =========================================================
# 3. Models
# =========================================================

class LinearNet(nn.Module):
    """
    Single linear layer. For:
    - toy1d: 1 -> 1 regression
    - mnist: input_dim -> 1 logistic regression
    """
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class MLPReLU(nn.Module):
    """
    Simple MLP with ReLU activations.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_model(name: str, input_dim: int, task_type: str) -> nn.Module:
    output_dim = 1  # regression or binary logit
    if name == "linear":
        return LinearNet(input_dim=input_dim, output_dim=output_dim)
    elif name == "mlp_relu":
        hidden = [64, 64]
        return MLPReLU(input_dim=input_dim, hidden_dims=hidden, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model: {name}")


# =========================================================
# 4. Loss & metrics
# =========================================================

def get_loss_fn(task_type: str):
    if task_type == "regression":
        return nn.MSELoss()
    elif task_type == "binary_classification":
        # We'll use BCEWithLogitsLoss and treat model output as logits
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             loss_fn,
             device: torch.device,
             task_type: str) -> Tuple[float, float]:
    """
    Returns: (avg_loss, metric)
    For regression: metric = MSE
    For binary classification: metric = accuracy
    """
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

            if task_type == "binary_classification":
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == y).float().sum().item()

    avg_loss = total_loss / total_n
    if task_type == "regression":
        metric = avg_loss  # MSE
    else:
        metric = correct / total_n  # accuracy
    return avg_loss, metric


# =========================================================
# 5. LLC estimation (simplified SGLD version)
# =========================================================

@dataclass
class LLCConfig:
    sgld_steps: int = 500
    sgld_lr: float = 1e-4
    sgld_batch_size: int = 128
    sgld_noise_scale: float = 1.0
    epsilons: Tuple[float, ...] = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)
    max_samples: int = 2000  # cap on number of samples used
    verbose: bool = False


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def assign_params(model: nn.Module, flat: torch.Tensor):
    """
    In-place assignment of parameters from flattened vector.
    """
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[idx:idx + numel].view_as(p))
        idx += numel


def estimate_llc_sgld(model: nn.Module,
                      train_loader: DataLoader,
                      loss_fn,
                      device: torch.device,
                      task_type: str,
                      llc_cfg: LLCConfig) -> float:
    """
    Very rough LLC estimator:
      1. Treat current model parameters as θ*.
      2. Run SGLD around θ* to sample θ_i.
      3. Compute ΔL_i = L(θ_i) - L(θ*).
      4. For thresholds ε_j, compute V(ε_j) ≈ P[ΔL_i <= ε_j].
      5. Fit log V(ε) vs log ε, slope ~ λ (local learning coefficient).

    This is meant as a simple, self-contained analogue of the
    method in local_coeff_computation.py.
    """
    model.eval()
    device = device

    # Step 0: snapshot θ* and its loss
    theta_star = flatten_params(model).clone().to(device)

    # Compute baseline loss L(θ*)
    with torch.no_grad():
        base_loss, _ = evaluate(model, train_loader, loss_fn, device, task_type)
    if llc_cfg.verbose:
        print(f"[LLC] Base loss at θ*: {base_loss:.6f}")

    # We will run SGLD starting from θ* (optionally could run multiple chains)
    theta = theta_star.clone()

    # For efficiency: create an iterator over train_loader that we re-use
    data_iter = iter(train_loader)

    # Collect ΔL samples
    delta_losses: List[float] = []

    model = model.to(device)

    for step in range(llc_cfg.sgld_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        # Load theta into model
        assign_params(model, theta)

        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()

        # Build gradient vector
        grads = torch.cat([p.grad.detach().flatten() for p in model.parameters()]).to(device)

        # SGLD update: θ ← θ - η∇L + sqrt(2η) * ξ
        eta = llc_cfg.sgld_lr
        noise = torch.randn_like(theta) * math.sqrt(2.0 * eta) * llc_cfg.sgld_noise_scale
        theta = theta - eta * grads + noise

        # Occasionally record ΔL w.r.t. full dataset
        if (step + 1) % max(1, llc_cfg.sgld_steps // 50) == 0:
            assign_params(model, theta)
            with torch.no_grad():
                loss_full, _ = evaluate(model, train_loader, loss_fn, device, task_type)
            delta = max(0.0, loss_full - base_loss)
            delta_losses.append(delta)
            if llc_cfg.verbose:
                print(f"[LLC] step {step+1}/{llc_cfg.sgld_steps}, loss {loss_full:.6f}, Δ {delta:.6e}")

        if len(delta_losses) >= llc_cfg.max_samples:
            break

    if len(delta_losses) < 5:
        # Too few samples; return NaN
        if llc_cfg.verbose:
            print("[LLC] Too few ΔL samples; returning NaN")
        return float("nan")

    delta_losses = np.array(delta_losses, dtype=float)

    # Compute empirical V(ε) for multiple epsilons
    epsilons = np.array(llc_cfg.epsilons, dtype=float)
    Vs = []
    for eps in epsilons:
        prob = (delta_losses <= eps).mean()
        prob = max(prob, 1e-10)  # avoid log(0)
        Vs.append(prob)
    Vs = np.array(Vs)

    # Fit log V ≈ λ * log ε + c
    log_eps = np.log(epsilons)
    log_V = np.log(Vs)
    A = np.vstack([log_eps, np.ones_like(log_eps)]).T
    # Solve least squares
    lam, c = np.linalg.lstsq(A, log_V, rcond=None)[0]

    if llc_cfg.verbose:
        print(f"[LLC] eps: {epsilons}")
        print(f"[LLC] V:   {Vs}")
        print(f"[LLC] log-V vs log-eps slope (λ) ≈ {lam:.4f}")

    # In SLT notation, volume scaling exponent is typically λ,
    # which is related to the local learning coefficient.
    # Here we just return lam as "LLC-like" quantity.
    return float(lam)


# =========================================================
# 6. Training loop with LLC tracking
# =========================================================

@dataclass
class TrainConfig:
    dataset: str
    model: str
    optimizer: str
    batch_size: int = 128
    lr: float = 1e-2
    weight_decay: float = 0.0
    max_epochs: int = 50
    plateau_window: int = 5
    plateau_tol: float = 1e-4
    llc_mode: str = "none"  # none | precomputed | online
    llc_check_interval: int = 5
    precomputed_llc_json: Optional[str] = None
    output_dir: str = "./results"
    seed: int = 42
    data_root: str = "./data"


def detect_plateau(history: List[float],
                   window: int,
                   tol: float) -> bool:
    """
    Simple plateau detector based on moving average of training loss.
    If over the last `window` epochs, the relative change is < tol, we say plateau.
    """
    if len(history) < window + 1:
        return False
    recent = history[-window-1:]
    first = recent[0]
    last = recent[-1]
    if first <= 0:
        return False
    rel_change = abs(last - first) / first
    return rel_change < tol


def train_and_track_llc(cfg: TrainConfig, llc_cfg: LLCConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = get_device()

    # Load dataset
    train_ds, test_ds, task_type = get_dataset(cfg.dataset, data_root=cfg.data_root)

    input_dim = train_ds[0][0].numel()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Build model
    model = get_model(cfg.model, input_dim=input_dim, task_type=task_type).to(device)

    # Loss & optimizer
    loss_fn = get_loss_fn(task_type)

    if cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Precomputed LLC map (epoch -> value)
    precomputed_llc: Dict[int, float] = {}
    if cfg.llc_mode == "precomputed":
        if not cfg.precomputed_llc_json:
            raise ValueError("llc_mode=precomputed but precomputed_llc_json not provided")
        with open(cfg.precomputed_llc_json, "r") as f:
            # Expecting {"0": val0, "1": val1, ...}
            tmp = json.load(f)
        precomputed_llc = {int(k): float(v) for k, v in tmp.items()}

    # Logs
    epochs = []
    train_losses = []
    test_losses = []
    metrics = []
    llc_values = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        avg_train_loss = total_loss / total_n
        avg_test_loss, metric = evaluate(model, test_loader, loss_fn, device, task_type)

        epochs.append(epoch)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        metrics.append(metric)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={avg_train_loss:.6f}, "
              f"test_loss={avg_test_loss:.6f}, "
              f"metric={metric:.4f}")

        # Decide whether to compute LLC
        llc_val = float("nan")
        if cfg.llc_mode == "precomputed":
            if epoch in precomputed_llc:
                llc_val = precomputed_llc[epoch]
        elif cfg.llc_mode == "online":
            # Check at intervals and only if plateau-ish
            if (epoch % cfg.llc_check_interval == 0 and
                    detect_plateau(train_losses, cfg.plateau_window, cfg.plateau_tol)):
                print(f"[Epoch {epoch:03d}] Plateau detected, estimating LLC via SGLD...")
                llc_val = estimate_llc_sgld(
                    model=model,
                    train_loader=train_loader,
                    loss_fn=loss_fn,
                    device=device,
                    task_type=task_type,
                    llc_cfg=llc_cfg,
                )
                print(f"[Epoch {epoch:03d}] Estimated LLC (slope) = {llc_val:.4f}")

        llc_values.append(llc_val)

    # Save logs to CSV
    import csv
    csv_path = os.path.join(cfg.output_dir, "training_llc_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "metric", "llc"])
        for e, tr, te, m, l in zip(epochs, train_losses, test_losses, metrics, llc_values):
            writer.writerow([e, tr, te, m, l])
    print(f"Saved log to {csv_path}")

    # Plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.plot(epochs, train_losses, label="train_loss")
    ax1.plot(epochs, test_losses, label="test_loss", linestyle="--")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("LLC (slope)")
    ax2.plot(epochs, llc_values, label="LLC", color="tab:red", marker="o")
    ax2.legend(loc="upper right")

    plt.title(f"LLC trajectory ({cfg.dataset}, {cfg.model}, {cfg.optimizer})")
    plt.tight_layout()

    plot_path = os.path.join(cfg.output_dir, "training_llc_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


# =========================================================
# 7. CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track LLC along training trajectory")

    parser.add_argument("--dataset", type=str, default="toy1d",
                        choices=["toy1d", "mnist"],
                        help="Dataset name")
    parser.add_argument("--model", type=str, default="mlp_relu",
                        choices=["linear", "mlp_relu"],
                        help="Model architecture")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam"],
                        help="Optimizer")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=50)

    parser.add_argument("--plateau_window", type=int, default=5,
                        help="Epoch window for plateau detection")
    parser.add_argument("--plateau_tol", type=float, default=1e-4,
                        help="Relative change threshold for plateau detection")

    parser.add_argument("--llc_mode", type=str, default="none",
                        choices=["none", "precomputed", "online"],
                        help="How to obtain LLC values")
    parser.add_argument("--llc_check_interval", type=int, default=5,
                        help="Epoch interval to check plateau & compute LLC (online mode)")

    parser.add_argument("--precomputed_llc_json", type=str, default=None,
                        help="Path to JSON file mapping epoch -> LLC (for llc_mode=precomputed)")

    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save logs & plots")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")

    # LLC / SGLD hyperparameters
    parser.add_argument("--llc_sgld_steps", type=int, default=500)
    parser.add_argument("--llc_sgld_lr", type=float, default=1e-4)
    parser.add_argument("--llc_sgld_batch_size", type=int, default=128)
    parser.add_argument("--llc_sgld_noise_scale", type=float, default=1.0)
    parser.add_argument("--llc_epsilons", type=str,
                        default="1e-5,5e-5,1e-4,5e-4,1e-3",
                        help="Comma-separated epsilons for volume scaling")
    parser.add_argument("--llc_max_samples", type=int, default=2000)
    parser.add_argument("--llc_verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    epsilons = tuple(float(x) for x in args.llc_epsilons.split(",") if x.strip())

    cfg = TrainConfig(
        dataset=args.dataset,
        model=args.model,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        plateau_window=args.plateau_window,
        plateau_tol=args.plateau_tol,
        llc_mode=args.llc_mode,
        llc_check_interval=args.llc_check_interval,
        precomputed_llc_json=args.precomputed_llc_json,
        output_dir=args.output_dir,
        seed=args.seed,
        data_root=args.data_root,
    )

    llc_cfg = LLCConfig(
        sgld_steps=args.llc_sgld_steps,
        sgld_lr=args.llc_sgld_lr,
        sgld_batch_size=args.llc_sgld_batch_size,
        sgld_noise_scale=args.llc_sgld_noise_scale,
        epsilons=epsilons,
        max_samples=args.llc_max_samples,
        verbose=args.llc_verbose,
    )

    print("TrainConfig:", asdict(cfg))
    print("LLCConfig:", asdict(llc_cfg))

    train_and_track_llc(cfg, llc_cfg)


if __name__ == "__main__":
    main()
