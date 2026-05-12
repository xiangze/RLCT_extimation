#!/usr/bin/env python3
"""
Estimate the Local Learning Coefficient (LLC, aka RLCT) for a small softmax
neural network while varying the "softmax coefficient" α (temperature inverse).

Method sketch (standard in SLT numerics):
  Z_n(β) = ∫ p(D|w)^β φ(w) dw
  E_β[NLL(w)] ≈ a + (−λ) * (1/β)
  → plot E_β[NLL] vs 1/β, slope = −λ.

Usage:
  python estimate_softmaxDNN.py --alphas 0.5 1.0 2.0 --betas 0.1 0.25 0.5
"""

from __future__ import annotations
import math
import os
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from argparse_dataclass import ArgumentParser

import models


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #
@dataclass 
class Dataset:
    X:torch.Tensor = field(default_factory=lambda: torch.zeros((3, 3)))
    Y:torch.Tensor = field(default_factory=lambda: torch.zeros((3, 3)))

class GaussianBlobDataset(Dataset):
    """2-D Gaussian blob classification dataset."""

    def __init__(self, n_per_class: int = 120, k: int = 3, std: float = 0.55, seed: int = 42):
        rng = np.random.RandomState(seed)
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        centers = np.stack([2.5 * np.cos(angles), 2.5 * np.sin(angles)], axis=1)

        X_list, y_list = [], []
        for i in range(k):
            Xi = rng.multivariate_normal(centers[i], np.eye(2) * std ** 2, size=n_per_class)
            X_list.append(Xi)
            y_list.append(np.full(n_per_class, i, dtype=np.int64))

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list)
        perm = rng.permutation(len(X))

        self.X = torch.from_numpy(X[perm])
        self.y = torch.from_numpy(y[perm])
        self.in_dim = X.shape[1]
        self.out_dim = k


# ------------------------------------------------------------------ #
#  Posterior energy (NLL + prior)                                      #
# ------------------------------------------------------------------ #

class PosteriorEnergy:
    """Computes NLL and L2 prior penalty for a given model."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, sigma_prior: float = 5.0):
        self.X = X
        self.y = y
        self.sigma_prior = sigma_prior

    def nll_sum(self, model: nn.Module, alpha: float) -> torch.Tensor:
        try:
            logits = model(self.X, alpha=alpha)
        except TypeError:
            logits = model(self.X)
        return F.cross_entropy(logits, self.y, reduction="sum")

    def prior_penalty(self, model: nn.Module) -> torch.Tensor:
        coeff = 0.5 / self.sigma_prior ** 2
        return sum(coeff * (p ** 2).sum() for p in model.parameters())

    def potential(self, model: nn.Module, alpha: float, beta: float = 1.0) -> torch.Tensor:
        """U(w) = β · NLL + prior  (used as the Langevin potential)."""
        return beta * self.nll_sum(model, alpha) + self.prior_penalty(model)


# ------------------------------------------------------------------ #
#  MAP warm-start                                                      #
# ------------------------------------------------------------------ #

class MAPSolver:
    """Finds a MAP estimate via Adam to warm-start SGLD."""

    def __init__(self, energy: PosteriorEnergy, lr: float = 5e-3, steps: int = 500):
        self.energy = energy
        self.lr = lr
        self.steps = steps

    def fit(self, model: nn.Module, alpha: float) -> nn.Module:
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            loss = self.energy.potential(model, alpha, beta=1.0)
            loss.backward()
            opt.step()
        return model


# ------------------------------------------------------------------ #
#  SGLD sampler                                                        #
# ------------------------------------------------------------------ #

@dataclass
class SGLDConfig:
    beta: float = 0.2
    step_size: float = 5e-5
    steps: int = 1200
    burnin: int = 600
    sample_every: int = 5
    step_decay: float = 0.9997


class SGLDSampler:
    """Stochastic Gradient Langevin Dynamics sampler."""

    def __init__(self, energy: PosteriorEnergy, cfg: SGLDConfig):
        self.energy = energy
        self.cfg = cfg

    def sample(self, model: nn.Module, alpha: float) -> Dict:
        """Run SGLD and return statistics of collected NLL samples."""
        mdl = copy.deepcopy(model)
        mdl.train()

        collected_nll: List[float] = []
        step_size = self.cfg.step_size

        for t in range(self.cfg.steps):
            for p in mdl.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            U = self.energy.potential(mdl, alpha, self.cfg.beta)
            U.backward()

            with torch.no_grad():
                for p in mdl.parameters():
                    p.add_(-step_size * p.grad)
                    p.add_(torch.randn_like(p) * math.sqrt(2.0 * step_size))

            step_size *= self.cfg.step_decay

            if t >= self.cfg.burnin and (t - self.cfg.burnin) % self.cfg.sample_every == 0:
                with torch.no_grad():
                    collected_nll.append(float(self.energy.nll_sum(mdl, alpha).cpu()))

        mean_nll = float(np.mean(collected_nll)) if collected_nll else float("nan")
        std_nll  = float(np.std(collected_nll))  if collected_nll else float("nan")
        return {"beta": self.cfg.beta, "mean_nll": mean_nll, "std_nll": std_nll,
                "num_samples": len(collected_nll)}


# ------------------------------------------------------------------ #
#  LLC estimator                                                       #
# ------------------------------------------------------------------ #

class LLCEstimator:
    """
    Estimates the LLC (λ) for a range of (α, β) values.

    Workflow per α:
      1. MAP warm-start
      2. SGLD at each β  →  E_β[NLL]
      3. Linear regression of E_β[NLL] vs 1/β  →  slope = −λ
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset, #GaussianBlobDataset,
        energy: PosteriorEnergy,
        map_solver: MAPSolver,
        sgld_cfg_template: SGLDConfig,
        betas: List[float],
        outdir: Path,
    ):
        self.model = model
        self.dataset = dataset
        self.energy = energy
        self.map_solver = map_solver
        self.sgld_cfg_template = sgld_cfg_template
        self.betas = sorted(betas)
        self.outdir = Path(outdir)
        self.alpha_list= []
        self.lambda_list= []
        self.summary =  {}

    @staticmethod
    def _fit_lambda(betas: List[float], mean_nlls: List[float]) -> Tuple[float, float, float]:
        """OLS of E_β[NLL] ~ a + b*(1/β); returns λ=-b, a, b."""
        x = np.array([1.0 / b for b in betas], dtype=np.float64)
        y = np.array(mean_nlls, dtype=np.float64)
        A = np.vstack([np.ones_like(x), x]).T
        a_hat, b_hat = np.linalg.lstsq(A, y, rcond=None)[0]
        return -float(b_hat), float(a_hat), float(b_hat)

    def run1(self,alpha,plot):
        # 1. MAP warm-start (mutates model in-place; intentional)
        self.map_solver.fit(self.model, alpha)
        # 2. SGLD across β values
        curve: Dict[float, dict] = {}
        for beta in self.betas:
            cfg = SGLDConfig(
                beta=beta,
                step_size=self.sgld_cfg_template.step_size,
                steps=self.sgld_cfg_template.steps,
                burnin=self.sgld_cfg_template.burnin,
                sample_every=self.sgld_cfg_template.sample_every,
                step_decay=self.sgld_cfg_template.step_decay,
            )
            stat = SGLDSampler(self.energy, cfg).sample(self.model, alpha)
            curve[beta] = stat

        # 3. Fit λ
        mean_nlls = [curve[b]["mean_nll"] for b in self.betas]
        lam, a_hat, b_hat = self._fit_lambda(self.betas, mean_nlls)

        # 4. Optional per-α plot
        png_path = ""
        if plot:
            png_path = str(self.outdir / f"curve_alpha_{str(alpha).replace('.','p')}.png")
            self._plot_curve(alpha, mean_nlls, a_hat, b_hat, png_path)

        self.summary[str(alpha)] = {
            "alpha": alpha, "betas": self.betas, "curve_mean_nll": mean_nlls,
            "linfit": {"a": a_hat, "b": b_hat}, "lambda_hat": lam,
            "curve_png": os.path.basename(png_path),
        }
        self.alpha_list.append(alpha)
        self.lambda_list.append(lam)

        print(f"alpha={alpha:>4}: λ̂ ≈ {lam:.3f}  (fit: {a_hat:.2f} + ({b_hat:.2f})·(1/β))")

        return self.alpha_list, self.lambda_list, self.summary

    def run(self, alphas: List[float], plot: bool = True) -> Tuple[List[float], List[float], dict]:
        for alpha in alphas:
            self.run1(alpha,plot)
        return self.alpha_list, self.lambda_list, self.summary

    # ---- plots ----

    def _plot_curve(self, alpha: float, mean_nlls: List[float], a_hat: float, b_hat: float, out_png: str):
        x = np.array([1.0 / b for b in self.betas])
        y = np.array(mean_nlls)
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, label="Eβ[NLL] samples")
        plt.plot(x, a_hat + b_hat * x, label=f"fit: λ ≈ {-b_hat:.3f}")
        plt.xlabel("1 / β"); plt.ylabel("Eβ[ total NLL ]")
        plt.title(f"Eβ[NLL] vs 1/β  (α={alpha})")
        plt.legend(loc="best"); plt.tight_layout()
        plt.savefig(out_png); plt.close()

    def plot_lambda_vs_alpha(self, alpha_list: List[float], lambda_list: List[float], out_png: str):
        idx = np.argsort(alpha_list)
        A, L = np.array(alpha_list)[idx], np.array(lambda_list)[idx]
        plt.figure(figsize=(6, 4))
        plt.plot(A, L, marker="o")
        plt.xlabel("softmax coefficient α"); plt.ylabel("estimated LLC λ")
        plt.title("Estimated LLC vs softmax coefficient")
        plt.tight_layout(); plt.savefig(out_png); plt.close()

class LLCEstimator_simple(LLCEstimator):
    def __init__(self,args, model,dataset, betas: List[float],outdir: Path):
        # Build components
        self.betas=args.betas
        self.outdir=args.outdir
        energy     = PosteriorEnergy(dataset.X, dataset.Y, sigma_prior=args.sigma_prior)
        map_solver = MAPSolver(energy, steps=args.map_steps)
        sgld_tmpl  = SGLDConfig(
            step_size=args.step_size,
            steps=args.sgld_steps,
            burnin=int(args.sgld_steps * args.burnin_frac),
            sample_every=args.sample_every,
            step_decay=args.step_decay)
        super().__init__(model,dataset,energy,map_solver,sgld_tmpl,betas,outdir)
        
# ------------------------------------------------------------------ #
#  Config & main                                                       #
# ------------------------------------------------------------------ #

@dataclass
class LLCConfigs:
    seed: int = 0
    n_per_class: int = 120
    std: float = 0.55
    hidden: int = 16
    alphas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    betas: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    sigma_prior: float = 5.0
    map_steps: int = 400
    sgld_steps: int = 900
    burnin_frac: float = 0.6
    sample_every: int = 5
    step_size: float = 5e-5
    step_decay: float = 0.9997
    outdir: Path = Path("out_llc_softmax")
    device: str = "cuda"
    plateau_thresh:int =1
    plateau_window:int= 1


def main():
    parser = ArgumentParser(LLCConfigs)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Build dataset
    dataset = GaussianBlobDataset(n_per_class=args.n_per_class, std=args.std, seed=42)
    X, y = dataset.X, dataset.y

    # Build model
    model = models.FlexibleCNN(
        in_channels=dataset.in_dim,
        num_classes=dataset.out_dim,
        base_channels=32,
        num_layers=args.hidden,
        use_resnet=False,
        dropout_rate=0.1,
        use_unet=False,
        use_layernorm=False,
        task="classification",
    )

    # Build components
    energy     = PosteriorEnergy(X, y, sigma_prior=args.sigma_prior)
    map_solver = MAPSolver(energy, steps=args.map_steps)
    sgld_tmpl  = SGLDConfig(
        step_size=args.step_size,
        steps=args.sgld_steps,
        burnin=int(args.sgld_steps * args.burnin_frac),
        sample_every=args.sample_every,
        step_decay=args.step_decay,
    )

    estimator = LLCEstimator(
        model=model,
        dataset=dataset,
        energy=energy,
        map_solver=map_solver,
        sgld_cfg_template=sgld_tmpl,
        betas=args.betas,
        outdir=args.outdir,
    )

    # Run
    alpha_list, lambda_list, summary = estimator.run(args.alphas)

    # λ vs α summary plot
    png_lambda = os.path.join(args.outdir, "lambda_vs_alpha.png")
    estimator.plot_lambda_vs_alpha(alpha_list, lambda_list, png_lambda)

    # JSON dump
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump({"summary": summary, "alphas": alpha_list, "lambdas": lambda_list}, f, indent=2)

    print("\nSaved:")
    print(" -", png_lambda)
    for a in summary.values():
        if a["curve_png"]:
            print(" -", os.path.join(args.outdir, a["curve_png"]))
    print(" -", os.path.join(args.outdir, "results.json"))


if __name__ == "__main__":
    main()
