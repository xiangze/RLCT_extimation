#!/usr/bin/env python3
"""
Estimate the Local Learning Coefficient (LLC, aka learning coefficient / RLCT)
for a small softmax neural network while varying the "softmax coefficient" α
(temperature inverse) using **power posteriors** and either **SGLD** or
**Pyro NUTS/HMC** (switchable via `--sampler {sgld,nuts,compare}`), inspired by:
  - suswei/RLCT `pyro_example.py`（温度付きポステリオリで Eβ[NLL] vs 1/β 線形）
  - Lau et al. (2023, 2024) on LLC for deep linear networks

This version adds:
  • **Step-size grid** for SGLD and **Richardson 外挿 (η→0)** per β（`--sgld-stepsize-grid` ＋ `--sgld-estimator`）
  • **複数シード複数反復**で SGLD の **分散/不確かさ** 推定（`--sgld-replicates`）
  • **加重最小二乗 (WLS)** による λ 推定（各 β の不確かさで重み付け）
  • **比較モード (`--sampler compare`)**：同一 α,β で **NUTS vs SGLD** を並走し
    Δ(β)=Eβ[NLL]_SGLD − Eβ[NLL]_NUTS を JSON に保存（外挿の較正用）
  • **スポット検証**：SGLD 実行時に特定 β を NUTS で確認（`--validate-betas`）

Method (standard in SLT numerics):
  Z_n(β) = ∫ p(D|w)^β φ(w) dw,  with φ a prior.
  Then d/dβ log Z_n(β) = E_β[ log p(D|w) ].
  As n → ∞ (realizable), log Z_n(β) ≈ β n ℓ₀ + λ log(nβ) + O(1).
  Hence E_β[log p(D|w)] ≈ nℓ₀ + λ/β, so with NLL = −log p(D|w):
      E_β[NLL] ≈ a + (−λ) * (1/β).   ⇒ slope = −λ.
We estimate λ by (weighted) linear regression of Eβ[NLL] vs 1/β.

Outputs (under `--outdir`):
  - per-α plot: `curve_alpha_<α>.png` (Eβ[NLL] vs 1/β & fitted line, with λ)
  - summary plot: `lambda_vs_alpha.png` for λ(α)
  - `results.json` with raw numbers（SGLD グリッド/外挿、NUTS ESS、WLS 詳細、Δ など）

Usage examples:
  # SGLD（リチャードソン外挿で η→0 推定）
  python llc_softmax_rlct.py --sampler sgld --alphas 0.5 1.0 2.0 --betas 0.1 0.25 0.5 \
    --sgld-steps 900 --sgld-stepsize-grid 5e-5 2.5e-5 --sgld-replicates 4 --sgld-estimator richardson

  # NUTS/HMC（厳密寄り）
  python llc_softmax_rlct.py --sampler nuts --alphas 0.5 1.0 --betas 0.15 0.3 \
    --nuts-warmup 800 --nuts-samples 1000 --chains 1

  # 比較モード（較正データの作成）
  python llc_softmax_rlct.py --sampler compare --alphas 1.0 --betas 0.15 0.3 0.5 \
    --sgld-stepsize-grid 5e-5 2.5e-5 --sgld-replicates 3 --nuts-warmup 600 --nuts-samples 800

Note: NUTS/HMC mode requires `pyro-ppl` installed.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Optional Pyro import (only needed for --sampler nuts/compare or --validate-betas)
try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer.mcmc import NUTS, MCMC
    _HAVE_PYRO = True
except Exception:
    _HAVE_PYRO = False

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
    def __init__(self, in_dim: int, hidden: int, out_dim: int, activation: str = "relu"):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.act = nn.Identity() if activation == "identity" else nn.ReLU()
        # He init
        nn.init.kaiming_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        h = self.act(self.lin1(x))
        logits = self.lin2(h)
        return alpha * logits  # softmax coefficient α multiplies logits

# ------------------- Pack/Unpack for NUTS -------------------

@dataclass
class ParamShapes:
    in_dim: int
    hidden: int
    out_dim: int

    @property
    def total(self) -> int:
        return self.hidden*self.in_dim + self.hidden + self.out_dim*self.hidden + self.out_dim

    def unpack(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """w: (..., D) → (W1, b1, W2, b2) with shapes matching lin layers."""
        D1 = self.hidden*self.in_dim
        D2 = D1 + self.hidden
        D3 = D2 + self.out_dim*self.hidden
        W1 = w[..., :D1].reshape(-1, self.hidden, self.in_dim)
        b1 = w[..., D1:D2].reshape(-1, self.hidden)
        W2 = w[..., D2:D3].reshape(-1, self.out_dim, self.hidden)
        b2 = w[..., D3:].reshape(-1, self.out_dim)
        return W1, b1, W2, b2


def logits_from_w(X: torch.Tensor, w: torch.Tensor, shapes: ParamShapes, alpha: float, activation: str) -> torch.Tensor:
    """Compute logits for all samples in X given flat params w (batchable)."""
    W1, b1, W2, b2 = shapes.unpack(w)
    H = (X @ W1.transpose(-1, -2)) + b1.unsqueeze(-2)   # (S, N, hidden)
    if activation == "relu":
        H = F.relu(H)
    logits = (H @ W2.transpose(-1, -2)) + b2.unsqueeze(-2)  # (S, N, out_dim)
    return alpha * logits

# ------------------- Objectives -------------------

def nll_sum_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    logits = model(X, alpha=alpha)
    return F.cross_entropy(logits, y, reduction='sum')

# L2 prior for raw parameter tensors

def l2_prior_penalty_params(params: List[torch.Tensor], sigma: float) -> torch.Tensor:
    coeff = 0.5 / (sigma ** 2)
    return sum(coeff * (p**2).sum() for p in params)

# -------------------- MAP fit (warm start for SGLD) ---------------------

def fit_map(model, X, y, alpha: float, sigma_prior: float = 5.0, lr: float = 5e-3, steps: int = 500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nll_sum_model(model, X, y, alpha) + l2_prior_penalty_params(list(model.parameters()), sigma_prior)
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
    seed: Optional[int] = None


def sgld_sample(model: nn.Module, X: torch.Tensor, y: torch.Tensor, cfg: SGLDConfig) -> Dict[str, float]:
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    mdl = SmallMLP(X.shape[1], model.lin1.out_features, model.lin2.out_features, activation="relu")
    mdl.load_state_dict(model.state_dict())
    mdl.train()

    collected_nll = []
    step_size = cfg.step_size

    for t in range(cfg.steps):
        for p in mdl.parameters():
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        U = cfg.beta * nll_sum_model(mdl, X, y, cfg.alpha) + l2_prior_penalty_params(list(mdl.parameters()), cfg.sigma_prior)
        U.backward()
        with torch.no_grad():
            for p in mdl.parameters():
                p.add_(-step_size * p.grad)
                p.add_(torch.randn_like(p) * math.sqrt(2.0 * step_size))
        step_size *= cfg.step_decay
        if t >= cfg.burnin and ((t - cfg.burnin) % cfg.sample_every == 0):
            with torch.no_grad():
                collected_nll.append(float(nll_sum_model(mdl, X, y, cfg.alpha).cpu()))

    mean_nll = float(np.mean(collected_nll)) if collected_nll else float('nan')
    std_nll  = float(np.std(collected_nll))  if collected_nll else float('nan')
    return {"beta": cfg.beta, "mean_nll": mean_nll, "std_nll": std_nll, "num_samples": len(collected_nll)}

# ---- SGLD grid + replicates + (optionally) Richardson extrapolation ----

def sgld_grid_estimates(base_model: nn.Module, X: torch.Tensor, y: torch.Tensor, alpha: float, betas: List[float],
                        step_sizes: List[float], replicates: int, steps: int, burnin_frac: float, sample_every: int,
                        sigma_prior: float, step_decay: float, seed0: int,
                        estimator: str = "richardson"):
    """
    Returns per-β dict with raw grid stats and an aggregate estimate using `estimator` in {"smallest","richardson","linfit"}.
    For each β and each η in `step_sizes`, run `replicates` independent SGLD chains and aggregate.
    """
    results = {}
    step_sizes = sorted(step_sizes, reverse=False)
    for beta in betas:
        per_eta = {}
        for i_eta, eta in enumerate(step_sizes):
            means = []
            for r in range(replicates):
                cfg = SGLDConfig(
                    beta=beta, step_size=eta, steps=steps, burnin=int(steps*burnin_frac), sample_every=sample_every,
                    sigma_prior=sigma_prior, alpha=alpha, step_decay=step_decay, seed=seed0 + 7919*i_eta + 97*r
                )
                stat = sgld_sample(base_model, X, y, cfg)
                means.append(stat["mean_nll"])
            per_eta[eta] = {
                "replicate_means": means,
                "mean_of_means": float(np.mean(means)),
                "std_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                "n_repl": replicates
            }

        # Aggregate per β according to estimator
        chosen_mean = None
        chosen_var = None
        aux = {"used": None}

        if estimator == "smallest":
            eta = step_sizes[0]
            m = per_eta[eta]["mean_of_means"]
            v = (per_eta[eta]["std_of_means"] ** 2) if per_eta[eta]["std_of_means"] is not None else None
            chosen_mean, chosen_var = m, v
            aux["used"] = {"type": "smallest", "eta": eta}

        elif estimator == "richardson":
            # find a pair (η, η/2)
            pair_used = None
            for j in range(1, len(step_sizes)):
                if abs(step_sizes[j-1] - 2*step_sizes[j])/step_sizes[j] < 1e-6:
                    eta_big = step_sizes[j-1]; eta_small = step_sizes[j]
                    m_big = per_eta[eta_big]["mean_of_means"]
                    m_small = per_eta[eta_small]["mean_of_means"]
                    m0 = 2*m_small - m_big
                    # variance approx: Var(2X - Y) = 4 Var(X) + Var(Y) （独立近似）
                    v_small = (per_eta[eta_small]["std_of_means"] ** 2) if per_eta[eta_small]["std_of_means"] is not None else None
                    v_big   = (per_eta[eta_big]["std_of_means"] ** 2)   if per_eta[eta_big]["std_of_means"]   is not None else None
                    if v_small is not None and v_big is not None:
                        v0 = 4*v_small + v_big
                    else:
                        v0 = None
                    chosen_mean, chosen_var = float(m0), (None if v0 is None else float(v0))
                    pair_used = (eta_small, eta_big)
                    break
            if chosen_mean is None:
                # fallback
                eta = step_sizes[0]
                chosen_mean = per_eta[eta]["mean_of_means"]
                chosen_var  = (per_eta[eta]["std_of_means"] ** 2) if per_eta[eta]["std_of_means"] is not None else None
                aux["used"] = {"type": "richardson_fallback_smallest", "eta": eta}
            else:
                aux["used"] = {"type": "richardson", "pair": pair_used}

        elif estimator == "linfit":
            # fit m(η) ≈ m0 + c1 η  （≥2点が必要）
            xs = np.array(step_sizes, dtype=float)
            ys = np.array([per_eta[eta]["mean_of_means"] for eta in step_sizes], dtype=float)
            if len(xs) >= 2:
                A = np.vstack([np.ones_like(xs), xs]).T
                m0, c1 = np.linalg.lstsq(A, ys, rcond=None)[0]
                chosen_mean = float(m0)
                # 粗い分散近似：観測の分散をそのまま使う or None
                chosen_var = float(np.var(ys, ddof=1)) if len(xs) > 2 else None
                aux["used"] = {"type": "linfit", "c1": float(c1)}
            else:
                eta = step_sizes[0]
                chosen_mean = per_eta[eta]["mean_of_means"]
                chosen_var  = (per_eta[eta]["std_of_means"] ** 2) if per_eta[eta]["std_of_means"] is not None else None
                aux["used"] = {"type": "linfit_fallback_smallest", "eta": eta}
        else:
            raise ValueError("estimator must be one of {smallest, richardson, linfit}")

        results[float(beta)] = {
            "per_eta": per_eta,
            "aggregate_mean": chosen_mean,
            "aggregate_var": chosen_var,
            "aggregate_info": aux["used"]
        }
    return results

# --------------------- NUTS/HMC (Pyro) -----------------------

@dataclass
class NUTSConfig:
    warmup: int = 600
    samples: int = 600
    chains: int = 1
    max_tree_depth: int = 8
    seed: int = 0


def _compute_ess_from_series(x: np.ndarray) -> float:
    """Very simple ESS via initial-positive-sequence of autocorrelations (Geyer-ish)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n <= 3:
        return float(n)
    var = np.var(x, ddof=0)
    if var == 0:
        return float(n)
    # autocovariances via FFT would be faster; here do simple upto max_lag
    max_lag = min(1000, n-1)
    rho = []
    for k in range(1, max_lag+1):
        c = np.dot(x[:-k], x[k:]) / n
        rho_k = c / var
        rho.append(rho_k)
        if rho_k < 0:
            break
    tau = 1 + 2*sum(rho) if rho else 1.0
    ess = n / tau if tau > 0 else float(n)
    return float(max(1.0, ess))


def nuts_mean_nll(X: torch.Tensor, y: torch.Tensor, shapes: ParamShapes, alpha: float, beta: float,
                  sigma_prior: float, activation: str, cfg: NUTSConfig) -> Dict[str, float]:
    if not _HAVE_PYRO:
        raise RuntimeError("Pyro is not installed. Please `pip install pyro-ppl`." )

    pyro.set_rng_seed(cfg.seed)

    def model_pyro():
        w = pyro.sample("w", dist.Normal(0.0, sigma_prior).expand([shapes.total]).to_event(1))
        logits = logits_from_w(X, w, shapes, alpha, activation=activation)[0]  # (N, C)
        with pyro.poutine.scale(scale=beta):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    nuts_kernel = NUTS(model_pyro, max_tree_depth=cfg.max_tree_depth)
    mcmc = MCMC(nuts_kernel, warmup_steps=cfg.warmup, num_samples=cfg.samples, num_chains=cfg.chains)
    mcmc.run()
    post = mcmc.get_samples(group_by_chain=False)
    w_samps = post["w"]  # (S, D)

    # Compute total-NLL for each posterior draw
    with torch.no_grad():
        logits = logits_from_w(X, w_samps, shapes, alpha, activation=activation)  # (S, N, C)
        log_probs = torch.log_softmax(logits, dim=-1)
        n = X.shape[0]
        idx = torch.arange(n)
        lp = log_probs[..., idx, y]  # (S, N)
        total_nll = -lp.sum(dim=-1).cpu().numpy()  # (S,)
        mean_nll = float(total_nll.mean())
        std_nll  = float(total_nll.std(ddof=0))
        ess      = _compute_ess_from_series(total_nll)
        var_of_mean = (std_nll**2) / max(1.0, ess)

    return {"beta": float(beta), "mean_nll": mean_nll, "std_nll": std_nll, "ess": ess, "var_of_mean": var_of_mean,
            "num_draws": int(w_samps.shape[0])}

# -------------- Lambda estimation (weighted) -----------------

def wls_lambda(betas: List[float], means: List[float], variances: Optional[List[Optional[float]]] = None):
    x = np.array([1.0/b for b in betas], dtype=np.float64)
    y = np.array(means, dtype=np.float64)
    X = np.vstack([np.ones_like(x), x]).T
    if variances is None or any(v is None or not np.isfinite(v) or v <= 0 for v in variances):
        # OLS fallback
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a_hat, b_hat = coef
        # homoskedastic residual se
        resid = y - X @ coef
        dof = max(1, len(y) - 2)
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * np.linalg.inv(X.T @ X)
    else:
        w = 1.0/np.array(variances, dtype=np.float64)
        W = np.diag(w)
        XtW = X.T @ W
        cov = np.linalg.inv(XtW @ X)
        coef = cov @ (XtW @ y)
        a_hat, b_hat = coef
        resid = y - X @ coef
        # sandwich (Huber-White) with W
        S = np.diag(resid**2 * w)
        cov = cov @ (X.T @ S @ X) @ cov
    se_a = float(np.sqrt(cov[0,0])); se_b = float(np.sqrt(cov[1,1]))
    lam = -float(b_hat); se_lam = float(se_b)
    return {"lambda_hat": lam, "se_lambda": se_lam, "intercept": float(a_hat), "se_intercept": se_a,
            "coef_raw": {"a": float(a_hat), "b": float(b_hat)}, "cov": [[float(cov[0,0]), float(cov[0,1])],[float(cov[1,0]), float(cov[1,1])]]}

# --------------------- Plots ----------------------

def plot_curve(alpha: float, betas: List[float], mean_nlls: List[float], variances: Optional[List[Optional[float]]],
               a_hat: float, b_hat: float, out_png: str):
    x = np.array([1.0/b for b in betas], dtype=np.float64)
    y = np.array(mean_nlls, dtype=np.float64)
    yhat = a_hat + b_hat * x

    plt.figure(figsize=(6,4))
    if variances is not None and all(v is not None and np.isfinite(v) and v>0 for v in variances):
        yerr = np.sqrt(np.array(variances, dtype=float))
        plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label='Eβ[NLL] ± SE')
    else:
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
    parser.add_argument('--activation', type=str, default='relu', choices=['relu','identity'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--betas', type=float, nargs='+', default=[0.1, 0.25, 0.5])
    parser.add_argument('--sigma-prior', type=float, default=5.0)
    parser.add_argument('--outdir', type=str, default='out_llc_softmax')

    # Sampler selection
    parser.add_argument('--sampler', type=str, default='sgld', choices=['sgld','nuts','compare'])

    # SGLD params
    parser.add_argument('--map-steps', type=int, default=400)
    parser.add_argument('--sgld-steps', type=int, default=900)
    parser.add_argument('--burnin-frac', type=float, default=0.6)
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--step-decay', type=float, default=0.9997)
    parser.add_argument('--sgld-stepsize-grid', type=float, nargs='+', default=[5e-5, 2.5e-5])
    parser.add_argument('--sgld-replicates', type=int, default=3)
    parser.add_argument('--sgld-estimator', type=str, default='richardson', choices=['smallest','richardson','linfit'])
    parser.add_argument('--validate-betas', type=float, nargs='*', default=None, help='only with sgld: run nuts at these β to spot-check')

    # NUTS params
    parser.add_argument('--nuts-warmup', type=int, default=600)
    parser.add_argument('--nuts-samples', type=int, default=600)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument('--max-tree-depth', type=int, default=8)

    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # data
    X, y = make_gaussian_blobs(n_per_class=args.n_per_class, std=args.std, k=3, seed=42)
    in_dim = X.shape[1]; out_dim = int(y.max().item() + 1)
    shapes = ParamShapes(in_dim=in_dim, hidden=args.hidden, out_dim=out_dim)

    summary = {}
    alpha_list, lambda_list = [] , []

    def nuts_curve(alpha: float):
        if not _HAVE_PYRO:
            raise SystemExit("Pyro is not available. Install with `pip install pyro-ppl`.")
        curve = {}
        for beta in args.betas:
            stat = nuts_mean_nll(
                X=X, y=y, shapes=shapes, alpha=alpha, beta=beta,
                sigma_prior=args.sigma_prior, activation=args.activation,
                cfg=NUTSConfig(warmup=args.nuts_warmup, samples=args.nuts_samples, chains=args.chains,
                               max_tree_depth=args.max_tree_depth, seed=args.seed))
            curve[float(beta)] = stat
        return curve

    def sgld_curve(alpha: float):
        base = SmallMLP(in_dim, args.hidden, out_dim, activation=args.activation)
        fit_map(base, X, y, alpha=alpha, sigma_prior=args.sigma_prior, steps=args.map_steps)
        grid = sgld_grid_estimates(base_model=base, X=X, y=y, alpha=alpha, betas=args.betas,
                                   step_sizes=args.sgld_stepsize_grid, replicates=args.sgld_replicates,
                                   steps=args.sgld_steps, burnin_frac=args.burnin_frac, sample_every=args.sample_every,
                                   sigma_prior=args.sigma_prior, step_decay=args.step_decay, seed0=args.seed,
                                   estimator=args.sgld_estimator)
        # Build an aggregate curve from grid results
        curve = {}
        for beta in args.betas:
            g = grid[float(beta)]
            curve[float(beta)] = {
                "beta": float(beta),
                "mean_nll": g["aggregate_mean"],
                # Use variance of mean if available; otherwise None
                "var_of_mean": g["aggregate_var"],
                "per_eta": g["per_eta"],
                "aggregate_info": g["aggregate_info"],
            }
        return curve

    if args.sampler == 'sgld':
        for alpha in args.alphas:
            curve = sgld_curve(alpha)
            betas_sorted = sorted(curve.keys())
            means = [curve[b]['mean_nll'] for b in betas_sorted]
            variances = [curve[b].get('var_of_mean', None) for b in betas_sorted]
            fit = wls_lambda(betas_sorted, means, variances)

            png_curve = os.path.join(args.outdir, f'curve_alpha_{str(alpha).replace(".","p")}.png')
            a_hat, b_hat = fit['intercept'], -fit['lambda_hat']
            plot_curve(alpha, betas_sorted, means, variances, a_hat, b_hat, png_curve)

            summary[str(alpha)] = {
                'alpha': alpha,
                'mode': 'sgld',
                'betas': betas_sorted,
                'curve_mean_nll': means,
                'variances': variances,
                'wls': fit,
                'curve_png': os.path.basename(png_curve),
            }
            alpha_list.append(alpha); lambda_list.append(fit['lambda_hat'])
            print(f"[SGLD] alpha={alpha:>4}: λ ≈ {fit['lambda_hat']:.3f} (± {fit['se_lambda']:.3f})")

            # spot validation via NUTS for specified betas
            if args.validate_betas and _HAVE_PYRO:
                val = {}
                for b in args.validate_betas:
                    stat = nuts_mean_nll(X, y, shapes, alpha, b, args.sigma_prior, args.activation,
                                         NUTSConfig(warmup=args.nuts_warmup, samples=args.nuts_samples, chains=args.chains,
                                                    max_tree_depth=args.max_tree_depth, seed=args.seed))
                    val[float(b)] = stat
                summary[str(alpha)]['spot_validation'] = val

    elif args.sampler == 'nuts':
        if not _HAVE_PYRO:
            raise SystemExit("Pyro is not available. Install with `pip install pyro-ppl`." )
        for alpha in args.alphas:
            curve = nuts_curve(alpha)
            betas_sorted = sorted(curve.keys())
            means = [curve[b]['mean_nll'] for b in betas_sorted]
            variances = [curve[b]['var_of_mean'] for b in betas_sorted]
            fit = wls_lambda(betas_sorted, means, variances)

            png_curve = os.path.join(args.outdir, f'curve_alpha_{str(alpha).replace(".","p")}.png')
            a_hat, b_hat = fit['intercept'], -fit['lambda_hat']
            plot_curve(alpha, betas_sorted, means, variances, a_hat, b_hat, png_curve)

            summary[str(alpha)] = {
                'alpha': alpha,
                'mode': 'nuts',
                'betas': betas_sorted,
                'curve_mean_nll': means,
                'variances': variances,
                'wls': fit,
                'curve_png': os.path.basename(png_curve),
            }
            alpha_list.append(alpha); lambda_list.append(fit['lambda_hat'])
            print(f"[NUTS] alpha={alpha:>4}: λ ≈ {fit['lambda_hat']:.3f} (± {fit['se_lambda']:.3f})")

    else:  # compare
        if not _HAVE_PYRO:
            raise SystemExit("Pyro is not available. Install with `pip install pyro-ppl`." )
        for alpha in args.alphas:
            curve_nuts = nuts_curve(alpha)
            curve_sgld = sgld_curve(alpha)
            betas_sorted = sorted(set(curve_nuts.keys()) & set(curve_sgld.keys()))

            means_n = [curve_nuts[b]['mean_nll'] for b in betas_sorted]
            vars_n  = [curve_nuts[b]['var_of_mean'] for b in betas_sorted]
            means_s = [curve_sgld[b]['mean_nll'] for b in betas_sorted]
            vars_s  = [curve_sgld[b].get('var_of_mean', None) for b in betas_sorted]

            # fits
            fit_n = wls_lambda(betas_sorted, means_n, vars_n)
            fit_s = wls_lambda(betas_sorted, means_s, vars_s if all(v is not None for v in vars_s) else None)

            # delta per beta
            dels, del_vars = [], []
            for i,b in enumerate(betas_sorted):
                d = means_s[i] - means_n[i]
                v = (vars_s[i] if vars_s[i] is not None else 0.0) + (vars_n[i] if vars_n[i] is not None else 0.0)
                dels.append(d); del_vars.append(v if v>0 else None)

            # plots
            png_curve = os.path.join(args.outdir, f'curve_alpha_{str(alpha).replace(".","p")}.png')
            a_hat, b_hat = fit_s['intercept'], -fit_s['lambda_hat']
            plot_curve(alpha, betas_sorted, means_s, vars_s, a_hat, b_hat, png_curve)

            # summary
            summary[str(alpha)] = {
                'alpha': alpha,
                'mode': 'compare',
                'betas': betas_sorted,
                'nuts': {'means': means_n, 'vars': vars_n, 'wls': fit_n},
                'sgld': {'means': means_s, 'vars': vars_s, 'wls': fit_s},
                'delta': {'means': dels, 'vars': del_vars},
                'curve_png': os.path.basename(png_curve),
            }
            alpha_list.append(alpha); lambda_list.append(fit_s['lambda_hat'])
            print(f"[CMP ] alpha={alpha:>4}: λ_sgld ≈ {fit_s['lambda_hat']:.3f}  vs  λ_nuts ≈ {fit_n['lambda_hat']:.3f}")

    # λ(α)
    png_lambda = os.path.join(args.outdir, 'lambda_vs_alpha.png')
    plot_lambda_vs_alpha(alpha_list, lambda_list, png_lambda)

    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'alphas': alpha_list, 'lambdas': lambda_list}, f, indent=2)

    print("\nSaved:")
    print(" -", png_lambda)
    for a in summary.values():
        print(" -", os.path.join(args.outdir, a['curve_png']))
    print(" -", os.path.join(args.outdir, 'results.json'))

if __name__ == '__main__':
    main()
