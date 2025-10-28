#!/usr/bin/env python3
"""
Estimate the Local Learning Coefficient (LLC, aka learning coefficient / RLCT)
for a small softmax neural network while varying the "softmax coefficient" α
(temperature inverse) using power posteriors and SGLD, inspired by:
  - suswei/RLCT `pyro_example.py` (tempered posterior + expected NLL vs 1/β)
  - Lau et al. (2023, 2024) on LLC for deep linear networks

Method sketch (standard in SLT numerics):
  Z_n(β) = ∫ p(D|w)^β φ(w) dw,  with φ a prior.
  Then d/dβ log Z_n(β) = E_β[ log p(D|w) ].
  As n → ∞ (realizable case), log Z_n(β) ≈ β n ℓ_0 + λ log(nβ) + O(1).
  Hence E_β[log p(D|w)] ≈ n ℓ_0 + λ/β, so
        E_β[NLL(w)] ≈ a + (-λ) * (1/β)     with NLL = −log p(D|w).
Thus, plotting E_β[NLL] vs 1/β should be ≈-linear, and slope = −λ.

This script:
  - builds a tiny MLP classifier with softmax logits scaled by α;
  - fits a MAP solution to warm-start sampling;
  - runs SGLD to approximate samples from p_β(w) ∝ p(D|w)^β φ(w);
  - estimates λ by linear regression of E_β[NLL] against 1/β;
  - repeats across α grid and plots λ(α).

NOTE
  * Results depend on the prior scale (σ_prior), dataset, and sampler settings.
  * SGLD here is simple full-batch Langevin; for higher fidelity consider HMC/NUTS
    (e.g. Pyro/NumPyro) and more β points.
  * Keep datasets small so runs finish quickly.

Usage (examples):
  python llc_softmax_sgld.py --alphas 0.5 1.0 2.0 \
      --betas 0.1 0.25 0.5 --map-steps 300 --sgld-steps 800

Outputs:
  - results JSON with per-α curve & fitted λ
  - per-α plot: Eβ[NLL] vs 1/β with linear fit & slope → λ
  - λ vs α summary plot
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------------------- Dataset ---------------------

def make_gaussian_blobs(n_per_class: int = 120, centers=None, std=0.55, k=3, seed=42):
    rng = np.random.RandomState(seed)
    if centers is None:
        angles = np.linspace(0, 2*np.pi, k, endpoint=False)
        centers = np.stack([2.5*np.cos(angles), 2.5*np.sin(angles)], axis=1)
    X, y = [], []
    for i in range(k):
        cov = np.eye(2) * (std ** 2)
        Xi = rng.multivariate_normal(mean=centers[i], cov=cov, size=n_per_class)
        yi = np.full(n_per_class, i, dtype=int)
        X.append(Xi)
        y.append(yi)
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])

# --------------------- Model ----------------------

class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        # He init
        nn.init.kaiming_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        h = F.relu(self.lin1(x))
        logits = self.lin2(h)
        return alpha * logits  # softmax coefficient α multiplies logits

# ------------------- Objectives -------------------

def nll_sum(model: nn.Module, X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    logits = model(X, alpha=alpha)
    return F.cross_entropy(logits, y, reduction='sum')

def l2_prior_penalty(model: nn.Module, sigma: float) -> torch.Tensor:
    s = torch.tensor(0.0, device=X.device)
    coeff = 0.5 / (sigma ** 2)
    for p in model.parameters():
        s = s + coeff * (p**2).sum()
    return s

# -------------------- MAP fit ---------------------

def fit_map(model, X, y, alpha: float, sigma_prior: float = 5.0, lr: float = 5e-3, steps: int = 500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nll_sum(model, X, y, alpha) + l2_prior_penalty(model, sigma_prior)
        loss.backward()
        opt.step()
    return model

# --------------------- SGLD -----------------------

@dataclass
class SGLDConfig:
    beta: float = 0.2
    step_size: float = 5e-5
    steps: int = 1200
    burnin: int = 600
    sample_every: int = 5
    sigma_prior: float = 5.0
    alpha: float = 1.0
    step_decay: float = 0.9997

def sgld_sample(model: nn.Module, X: torch.Tensor, y: torch.Tensor, cfg: SGLDConfig) -> Dict[str, float]:
    # copy so we don't mutate caller's model
    mdl = SmallMLP(X.shape[1], model.lin1.out_features, model.lin2.out_features)
    mdl.load_state_dict(model.state_dict())
    mdl.train()

    collected_nll = []
    step_size = cfg.step_size

    for t in range(cfg.steps):
        for p in mdl.parameters():
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        U = cfg.beta * nll_sum(mdl, X, y, cfg.alpha) + l2_prior_penalty(mdl, cfg.sigma_prior)
        U.backward()
        with torch.no_grad():
            for p in mdl.parameters():
                # Langevin step: w ← w − η ∇U + √(2η) ξ
                p.add_(-step_size * p.grad)
                p.add_(torch.randn_like(p) * math.sqrt(2.0 * step_size))
        step_size *= cfg.step_decay
        if t >= cfg.burnin and ((t - cfg.burnin) % cfg.sample_every == 0):
            with torch.no_grad():
                collected_nll.append(float(nll_sum(mdl, X, y, cfg.alpha).cpu()))

    mean_nll = float(np.mean(collected_nll)) if collected_nll else float('nan')
    std_nll  = float(np.std(collected_nll))  if collected_nll else float('nan')
    return {"beta": cfg.beta, "mean_nll": mean_nll, "std_nll": std_nll, "num_samples": len(collected_nll)}

# -------------- Lambda estimation -----------------

def estimate_lambda(betas: List[float], mean_nlls: List[float]) -> Tuple[float, Tuple[float, float]]:
    x = np.array([1.0/b for b in betas], dtype=np.float64)
    y = np.array(mean_nlls, dtype=np.float64)
    A = np.vstack([np.ones_like(x), x]).T
    a_hat, b_hat = np.linalg.lstsq(A, y, rcond=None)[0]
    lam = -b_hat
    return float(lam), (float(a_hat), float(b_hat))

# --------------------- Plots ----------------------
def plot_curve(alpha: float, betas: List[float], mean_nlls: List[float], a_hat: float, b_hat: float, out_png: str):
    x = np.array([1.0/b for b in betas], dtype=np.float64)
    y = np.array(mean_nlls, dtype=np.float64)
    yhat = a_hat + b_hat * x

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, label='Eβ[NLL] samples')
    plt.plot(x, yhat, label=f'fit: λ ≈ {-b_hat:.3f}')
    plt.xlabel('1 / β')
    plt.ylabel('Eβ[ total NLL ]')
    plt.title(f'Eβ[NLL] vs 1/β (alpha={alpha})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_png); plt.close()


def plot_lambda_vs_alpha(alphas: List[float], lambdas: List[float], out_png: str):
    idx = np.argsort(np.array(alphas))
    A = np.array(alphas)[idx]
    L = np.array(lambdas)[idx]
    plt.figure(figsize=(6,4))
    plt.plot(A, L, marker='o')
    plt.xlabel('softmax coefficient α')
    plt.ylabel('estimated LLC λ')
    plt.title('Estimated LLC vs softmax coefficient')
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

# --------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-per-class', type=int, default=120)
    parser.add_argument('--std', type=float, default=0.55)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--betas', type=float, nargs='+', default=[0.1, 0.25, 0.5])
    parser.add_argument('--sigma-prior', type=float, default=5.0)
    parser.add_argument('--map-steps', type=int, default=400)
    parser.add_argument('--sgld-steps', type=int, default=900)
    parser.add_argument('--burnin-frac', type=float, default=0.6)
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--step-size', type=float, default=5e-5)
    parser.add_argument('--step-decay', type=float, default=0.9997)
    parser.add_argument('--outdir', type=str, default='out_llc_softmax')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # data
    X, y = make_gaussian_blobs(n_per_class=args.n_per_class, std=args.std, k=3, seed=42)
    in_dim = X.shape[1]; out_dim = int(y.max().item() + 1)

    # storage
    summary = {}
    alpha_list, lambda_list = [] , []

    for alpha in args.alphas:
        base = SmallMLP(in_dim, args.hidden, out_dim)
        fit_map(base, X, y, alpha=alpha, sigma_prior=args.sigma_prior, steps=args.map_steps)

        curve = {}
        for beta in args.betas:
            cfg = SGLDConfig(
                beta=beta,
                step_size=args.step_size,
                steps=args.sgld_steps,
                burnin=int(args.sgld_steps * args.burnin_frac),
                sample_every=args.sample_every,
                sigma_prior=args.sigma_prior,
                alpha=alpha,
                step_decay=args.step_decay,
            )
            stat = sgld_sample(base, X, y, cfg)
            curve[float(beta)] = stat

        betas_sorted = sorted(curve.keys())
        mean_nlls = [curve[b]['mean_nll'] for b in betas_sorted]
        lam, (a_hat, b_hat) = estimate_lambda(betas_sorted, mean_nlls)

        # plots per alpha
        png_curve = os.path.join(args.outdir, f'curve_alpha_{str(alpha).replace(".","p")}.png')
        plot_curve(alpha, betas_sorted, mean_nlls, a_hat, b_hat, png_curve)

        summary[str(alpha)] = {
            'alpha': alpha,
            'betas': betas_sorted,
            'curve_mean_nll': mean_nlls,
            'linfit': {'a': a_hat, 'b': b_hat},
            'lambda_hat': lam,
            'curve_png': os.path.basename(png_curve),
        }
        alpha_list.append(alpha); lambda_list.append(lam)
        print(f"alpha={alpha:>4}: lambda_hat ≈ {lam:.3f} | fit y≈{a_hat:.2f} + ({b_hat:.2f})*(1/β)")

    # λ vs α plot
    png_lambda = os.path.join(args.outdir, 'lambda_vs_alpha.png')
    plot_lambda_vs_alpha(alpha_list, lambda_list, png_lambda)

    # JSON dump
    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'alphas': alpha_list, 'lambdas': lambda_list}, f, indent=2)

    print("\nSaved:")
    print(" -", png_lambda)
    for a in summary.values():
        print(" -", os.path.join(args.outdir, a['curve_png']))
    print(" -", os.path.join(args.outdir, 'results.json'))

if __name__ == '__main__':
    # Make X, y visible to functions that capture it (for l2 prior helper)
    X, y = make_gaussian_blobs()
    main()
