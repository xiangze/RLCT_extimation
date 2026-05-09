"""
================================================================================
DNN学習における対称性・鞍点・過学習抑止の数値実験スイート
================================================================================

主張:
  「不変多様体でつながれた鞍点を経由した段階的対称性学習が、 局所最適解への閉じ込めを防ぎ、過学習を抑止する」

実験構成:
  Exp A  深線形NN — 段階的特異値獲得（H1・H2）
  Exp B  グロッキング — 対称性獲得と汎化の相転移（H3）  ← GPU推奨
  
実行方法:
  python symmetry_overfitting_experiments.py --exp A    # 単一実験
  python symmetry_overfitting_experiments.py --exp all  # 全実験（時間かかる）

"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import RLCT_helper
import estimate_softmaxDNN 
import llc_training_trajectory as traj
from estimate_softmaxDNN  import LLCConfigs 

# 日本語フォント
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
except:
    pass

torch.manual_seed(42)
np.random.seed(42)

def banner(s,n=60):
    print("\n" + "="*n)
    print(s)
    print("="*n)

# ==============================================================================
# 実験 A: 深線形NN — 段階的特異値獲得 (Saddle-to-Saddle)
# ==============================================================================
# 仮説 H1: 鞍点漸近 → 低ランク解への暗黙的正則化
# 仮説 H2: 特異値が一本ずつ順に活性化する
#
# 先行研究:
#   Saxe et al. (2014) — 深線形NNの厳密解、段階的特異値獲得を予測
#   Arora et al. (2019) — 低ランクバイアスの証明
#   Zhang et al. (2025) — 時間スケール分離が複雑度を制御
#
# 何を見る:
#   - 積行列 W_L...W_1 の特異値σ_iの時系列
#   - 有効ランク eff_rank = (Σσ)²/Σσ²
#   - 深さdepthが大きいほどプラトーが長く、ランク圧縮が強い
#
# 「過学習抑止」との関係:
#   低ランク解 = 少ない有効パラメータ = 暗黙的正則化 = 過学習しにくい
# ==============================================================================

def make_low_rank_data(n=300, d=10, rank=2, noise=0.02, seed=42):
    """真の写像がrank行列であるデータを生成（過剰パラメータ設定）"""
    torch.manual_seed(seed)
    U = torch.randn(d, rank)
    V = torch.randn(rank, d)
    W_true = U @ V / rank**0.5
    X = torch.randn(n, d)
    Y = X @ W_true.T + noise * torch.randn(n, d)
    n_tr = int(n * 0.7)
    return (X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:], W_true)

class DeepLinearNet(nn.Module):
    def __init__(self, d, depth=3, init_scale=0.01):
        """
        init_scale が小さい → 鞍点（原点付近）から出発
        これが段階的ランク学習の必要条件
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=False) for _ in range(depth)
        ])
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=init_scale)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def product_matrix(self):
        W = self.layers[0].weight
        for l in self.layers[1:]:
            W = l.weight @ W
        return W.detach()

    def effective_rank(self):
        sv = torch.linalg.svdvals(self.product_matrix()).numpy()
        sv = sv[sv > 1e-8]
        if len(sv) == 0:
            return 0.0
        return float((sv.sum()**2) / (sv**2).sum())


def run_exp_A(args,cfg:LLCConfigs, d=10, depths=(2, 3, 4), rank=2, n_steps=12000,
              lr=0.005, log_every=100,calcRLCT=True):
    """
    実験A: 深さを変えて段階的特異値獲得を観測

    Returns: dict of {depth: history}
    """
    X_tr, Y_tr, X_te, Y_te, W_true = make_low_rank_data(d=d, rank=rank)
    results = {}
    
    in_dim=args.in_dim
    out_dim=args.out_dim

    for depth in depths:
        model = DeepLinearNet(d, depth=depth)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        H = dict(step=[], train_loss=[], test_loss=[],
                 singular_values=[], eff_rank=[],llc=[])
        
        llc_val = float("nan")
        te_losses=[]
        for step in range(n_steps):
            opt.zero_grad()
            loss = loss_fn(model(X_tr), Y_tr)
            loss.backward()
            opt.step()

            if step % log_every == 0:
                with torch.no_grad():
                    te_loss = loss_fn(model(X_te), Y_te).item()
                    te_losses.append(te_loss)
                sv = torch.linalg.svdvals(model.product_matrix()).numpy()
                H['step'].append(step)
                H['train_loss'].append(loss.item())
                H['test_loss'].append(te_loss)
                H['singular_values'].append(sv.copy())
                H['eff_rank'].append(model.effective_rank())
                if(calcRLCT and 
                    traj.detect_plateau(te_losses, cfg.plateau_window, cfg.plateau_thresh)):
                        alpha_list,lambda_list,summary = estimate_softmaxDNN.get_lambda_from_alphabeta(model,args,in_dim,out_dim)
                        llc_val=[alpha_list,lambda_list]
                        H["llc"].append(llc_val)
                        print(f"[step {step:03d}] Estimated LLC (slope) = {llc_val:.4f}")
                results[depth] = H

        final = H['singular_values'][-1]
        print(f"  depth={depth}: eff_rank={H['eff_rank'][-1]:.2f}  "
              f"sv=[{final[0]:.3f}, {final[1]:.3f}, {final[2]:.4f}...]  "
              f"tr_loss={H['train_loss'][-1]:.5f}")

    return results

def plot_exp_A(results, depths, rank, outfile):
    colors = {2: '#E8593C', 3: '#4A90D9', 4: '#1D9E75'}
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.32)
    fig.suptitle(
        'Exp A: Deep Linear Network — Staged Singular Value Acquisition\n'
        f'(true rank={rank}, implicit regularization via saddle-to-saddle dynamics)',
        fontsize=11)

    # (a) 特異値時系列 depth=3
    ax1 = fig.add_subplot(gs[0, :2])
    H3 = results[3]
    steps = np.array(H3['step'])
    svs = np.array(H3['singular_values'])
    sv_colors = ['#E8593C', '#4A90D9', '#1D9E75', '#EF9F27', '#888']
    for i in range(min(5, svs.shape[1])):
        ax1.plot(steps, svs[:, i], color=sv_colors[i], lw=2,
                 label=f'sigma_{i+1}')
    ax1.axhline(0, color='k', lw=0.5, ls='--')
    # 活性化タイミングを縦線で示す
    for i, thr in enumerate([0.3, 0.1]):
        active = np.where(svs[:, i] > thr)[0]
        if len(active) > 0:
            ax1.axvline(steps[active[0]], color=sv_colors[i],
                        ls=':', lw=1.5, alpha=0.7)
            ax1.text(steps[active[0]] + 200, svs[:, i].max() * 0.4,
                     f'sigma_{i+1} activated', color=sv_colors[i], fontsize=8)
    ax1.set(xlabel='step', ylabel='singular value',
            title='(a) Staged singular value activation  [depth=3, rank*=2]\n'
                  'Each plateau = one saddle point; each jump = saddle escape')
    ax1.legend(fontsize=9)

    # (b) 有効ランク比較
    ax2 = fig.add_subplot(gs[0, 2])
    for d in depths:
        H = results[d]
        ax2.plot(H['step'], H['eff_rank'],
                 color=colors[d], lw=2, label=f'depth={d}')
    ax2.axhline(rank, color='k', ls='--', lw=1, label=f'true rank={rank}')
    ax2.set(xlabel='step', ylabel='effective rank',
            title='(b) Effective rank convergence\n(deeper = stronger low-rank bias)')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(rank + 2, 4))

    # (c) 訓練損失（プラトー形状）
    ax3 = fig.add_subplot(gs[1, 0])
    for d in depths:
        H = results[d]
        ax3.semilogy(H['step'], H['train_loss'],
                     color=colors[d], lw=2, label=f'depth={d}')
    ax3.set(xlabel='step', ylabel='train loss (log)',
            title='(c) Loss curve — plateau structure\n(longer plateau = stronger regularization)')
    ax3.legend(fontsize=8)

    # (d) テスト vs 訓練損失
    ax4 = fig.add_subplot(gs[1, 1])
    d_ref = 3
    H = results[d_ref]
    ax4.semilogy(H['step'], H['train_loss'],
                 color='#E8593C', lw=2, label='train loss')
    ax4.semilogy(H['step'], H['test_loss'],
                 color='#4A90D9', lw=2, ls='--', label='test loss')
    ax4.set(xlabel='step', ylabel='loss (log)',
            title=f'(d) Train vs Test loss  [depth={d_ref}]\n'
                  '(implicit regularization prevents overfitting)')
    ax4.legend(fontsize=8)

    # (e) 有効ランク vs テスト損失（低ランク = 良い汎化）
    ax5 = fig.add_subplot(gs[1, 2])
    for d in depths:
        H = results[d]
        sc = ax5.scatter(H['eff_rank'], H['test_loss'],
                         c=H['step'], cmap='viridis', s=10, alpha=0.5,
                         label=f'depth={d}')
    ax5.set(xlabel='effective rank', ylabel='test loss',
            title='(e) Effective rank vs test loss\n(lower rank → better generalization)')
    plt.colorbar(sc, ax=ax5, label='step')

    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# ==============================================================================
# 実験 B: グロッキング — 対称性獲得と汎化の相転移
# ==============================================================================
# 仮説 H3: 対称性学習が段階的に起き、獲得が汎化の相転移を引き起こす
#
# 先行研究:
#   Power et al. (2022) — グロッキング発見
#   Intrinsic Task Symmetry (2026) — 対称性獲得と汎化の因果関係
#   Grokking complexity dynamics (2025) — 複雑度の相転移
#
# 重要: グロッキングには通常50,000〜200,000ステップが必要
#       GPUを使用し、n_epochs を十分大きく設定すること
#
# weight_decay の役割:
#   WD=0:   記憶化に永続的に閉じ込まれる（局所最適解）
#   WD=中:  グロッキング（記憶化の後、遅延して汎化）
#   WD=大:  最初から汎化（対称性を素早く獲得）
#
# Fourier集中度の意味:
#   剰余算の真の解はFourier基底（周期関数）で表現される
#   → 汎化時に埋め込みベクトルがFourier成分に集中する
#   → これが「対称性獲得」の定量的指標
# ==============================================================================

class GrokMLP(nn.Module):
    """グロッキング実験用MLP（標準設定）"""
    def __init__(self, p, d_emb=64, d_hidden=256):
        super().__init__()
        self.embed = nn.Embedding(p, d_emb)
        self.net = nn.Sequential(
            nn.Linear(d_emb * 2, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),  nn.ReLU(),
            nn.Linear(d_hidden, p)
        )

    def forward(self, a, b):
        return self.net(torch.cat([self.embed(a), self.embed(b)], -1))

    def fourier_concentration(self, p, top_k=4):
        """埋め込みのFourierスペクトル集中度（対称性指標）"""
        E = self.embed.weight.detach().numpy()[:p]
        Ef = np.fft.fft(E, axis=0)
        pw = np.abs(Ef)**2
        tot = pw.sum(0) + 1e-10
        return float((np.sort(pw, 0)[::-1][:top_k] / tot).sum(0).mean())


def make_grok_data(p=23, train_frac=0.4, seed=0):
    torch.manual_seed(seed)
    pairs = torch.tensor([(a, b, (a+b) % p)
                          for a in range(p) for b in range(p)],
                         dtype=torch.long)
    idx = torch.randperm(len(pairs))
    n_tr = int(len(pairs) * train_frac)
    return pairs[idx[:n_tr]], pairs[idx[n_tr:]]


def run_exp_B(p=23, train_frac=0.4,
              weight_decays=(0.0, 1.0, 5.0),
              n_epochs=100000,   # 十分長く！GPUで〜10分
              lr=1e-3, log_every=500):
    """
    実験B: weight_decayを変えてグロッキングを観測

    NOTE: n_epochs=100000 を推奨。短いと汎化が現れないことがある。
    Returns: dict of {wd_label: history}
    """
    tr_data, te_data = make_grok_data(p=p, train_frac=train_frac)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    results = {}
    for wd in weight_decays:
        label = f'WD={wd}'
        model = GrokMLP(p).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        tr_d = tr_data.to(device)
        te_d = te_data.to(device)
        a_tr, b_tr, c_tr = tr_d[:,0], tr_d[:,1], tr_d[:,2]
        a_te, b_te, c_te = te_d[:,0], te_d[:,1], te_d[:,2]

        H = dict(epoch=[], tr_acc=[], te_acc=[], fourier=[])

        for ep in range(n_epochs):
            model.train()
            loss = F.cross_entropy(model(a_tr, b_tr), c_tr)
            opt.zero_grad(); loss.backward(); opt.step()

            if ep % log_every == 0:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(a_tr,b_tr).argmax(-1)==c_tr).float().mean().item()
                    te_acc = (model(a_te,b_te).argmax(-1)==c_te).float().mean().item()
                fc = model.fourier_concentration(p)
                H['epoch'].append(ep)
                H['tr_acc'].append(tr_acc)
                H['te_acc'].append(te_acc)
                H['fourier'].append(fc)

                if ep % (log_every * 10) == 0:
                    print(f"  {label} ep={ep:6d}: "
                          f"tr={tr_acc:.3f} te={te_acc:.3f} fc={fc:.3f}")

        results[label] = H

    return results


def plot_exp_B(results, outfile):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        'Exp B: Grokking — Symmetry Acquisition Causes Generalization Phase Transition\n'
        '(mod-23 addition, Fourier concentration = symmetry indicator)',
        fontsize=10)

    colors_B = ['#E8593C', '#4A90D9', '#1D9E75', '#EF9F27']
    labels = list(results.keys())

    for i, (label, H) in enumerate(results.items()):
        c = colors_B[i % len(colors_B)]
        epochs = H['epoch']
        # test accuracy
        axes[0].plot(epochs, H['te_acc'], color=c, lw=2, label=label)
        # train accuracy
        axes[1].plot(epochs, H['tr_acc'], color=c, lw=2, label=label)
        # fourier vs test acc trajectory
        axes[2].scatter(H['fourier'], H['te_acc'],
                        c=epochs, cmap='viridis', s=8, alpha=0.6)
        axes[2].annotate('',
            xy=(H['fourier'][-1], H['te_acc'][-1]),
            xytext=(H['fourier'][0], H['te_acc'][0]),
            arrowprops=dict(arrowstyle='->', color=c, lw=2))

    axes[0].set(title='Test accuracy (generalization)',
                xlabel='epoch', ylabel='accuracy')
    axes[0].legend(fontsize=9)
    axes[1].set(title='Train accuracy (memorization)',
                xlabel='epoch', ylabel='accuracy')
    axes[1].legend(fontsize=9)
    axes[2].set(title='Fourier concentration vs test acc\n(arrow = time, each point = checkpoint)',
                xlabel='Fourier concentration (symmetry)',
                ylabel='test accuracy')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'),
                 ax=axes[2], label='epoch')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# ==============================================================================
# 実験 C: 対称性の食い違いと汎化
# ==============================================================================
# 仮説 H4: データ対称性とネットワークの適合が過学習を防ぐ
#
# 先行研究:
#   Intrinsic Task Symmetry (2026) — 対称性の因果的役割
#   Chen et al. (2023) Stochastic Collapse — 不変集合への引力
#   Ziyin (2023) — 鏡映対称性 → 構造的拘束
#
# 2×2 実験デザイン:
#   データ対称性 {高, 低} × ネットワーク幅 {大, 小}
#
# 測定:
#   - 訓練精度 vs テスト精度（汎化ギャップ）
#   - 学習率崩壊後の過学習度合い
#   - 重みのスペクトル集中度
# ==============================================================================

def make_sym_data(n=800, d=8, n_class=4, symmetry='high',
                  noise=0.4, seed=42):
    """
    symmetry='high': クラス重心が等間隔配置（回転対称）
    symmetry='low' : クラス重心がランダム配置
    """
    np.random.seed(seed)
    if symmetry == 'high':
        angles = [2 * np.pi * i / n_class for i in range(n_class)]
        centers = np.array([[3*np.cos(a), 3*np.sin(a)] + [0]*(d-2)
                             for a in angles])
    else:
        centers = np.random.randn(n_class, d) * 3

    X, y = [], []
    for c_idx, center in enumerate(centers):
        Xi = center + noise * np.random.randn(n // n_class, d)
        X.append(Xi)
        y.extend([c_idx] * (n // n_class))

    X = torch.tensor(np.vstack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    perm = torch.randperm(len(y))
    X, y = X[perm], y[perm]
    n_tr = int(len(y) * 0.6)
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


class WideMLP(nn.Module):
    def __init__(self, d_in, n_class, width, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, n_class))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def weight_spectral_ratio(self):
        """全重み行列の最大/全特異値比（低いほど均一 = より対称的）"""
        ratios = []
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.shape[0] > 1:
                sv = torch.linalg.svdvals(m.weight).detach().numpy()
                if sv.sum() > 0:
                    ratios.append(sv.max() / sv.mean())
        return float(np.mean(ratios)) if ratios else 0.0


def run_exp_C(configs, n_epochs=2000, lr=1e-3, wd=1e-3, log_every=100):
    """
    実験C: データ対称性とNW幅を変えた汎化比較
    """
    results = {}
    for cfg in configs:
        X_tr, y_tr, X_te, y_te = make_sym_data(
            symmetry=cfg['sym'], n=800, d=8, n_class=4)
        model = WideMLP(d_in=8, n_class=4, width=cfg['width'])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        H = dict(epoch=[], tr_acc=[], te_acc=[],
                 train_loss=[], test_loss=[], spectral_ratio=[])

        for ep in range(n_epochs):
            model.train()
            logits = model(X_tr)
            loss = F.cross_entropy(logits, y_tr)
            opt.zero_grad(); loss.backward(); opt.step()

            if ep % log_every == 0:
                model.eval()
                with torch.no_grad():
                    tr_a = (model(X_tr).argmax(-1)==y_tr).float().mean().item()
                    te_a = (model(X_te).argmax(-1)==y_te).float().mean().item()
                    tr_l = F.cross_entropy(model(X_tr), y_tr).item()
                    te_l = F.cross_entropy(model(X_te), y_te).item()
                sr = model.weight_spectral_ratio()
                H['epoch'].append(ep); H['tr_acc'].append(tr_a)
                H['te_acc'].append(te_a); H['train_loss'].append(tr_l)
                H['test_loss'].append(te_l); H['spectral_ratio'].append(sr)

        results[cfg['label']] = H
        print(f"  {cfg['label']:35s}: "
              f"tr={H['tr_acc'][-1]:.3f}  te={H['te_acc'][-1]:.3f}  "
              f"gap={H['tr_acc'][-1]-H['te_acc'][-1]:.3f}")

    return results


def plot_exp_C(results, configs, outfile):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Exp C: Data-Network Symmetry Mismatch and Generalization\n'
        '(does symmetry alignment prevent overfitting?)',
        fontsize=11)

    colors_C = [c['color'] for c in configs]
    labels_C = [c['label'] for c in configs]

    for i, (label, H) in enumerate(results.items()):
        c = colors_C[i]; ls = configs[i].get('ls', '-')
        axes[0,0].plot(H['epoch'], H['te_acc'],  color=c, lw=2, ls=ls, label=label)
        axes[0,1].plot(H['epoch'], H['tr_acc'],  color=c, lw=2, ls=ls)
        axes[0,2].plot(H['epoch'],
                       np.array(H['tr_acc']) - np.array(H['te_acc']),
                       color=c, lw=2, ls=ls, label=label)
        axes[1,0].semilogy(H['epoch'], H['train_loss'], color=c, lw=2, ls='-')
        axes[1,0].semilogy(H['epoch'], H['test_loss'],  color=c, lw=1, ls='--')
        axes[1,1].plot(H['epoch'], H['spectral_ratio'], color=c, lw=2, ls=ls)

    # Summary bar chart
    final_tr = [results[c['label']]['tr_acc'][-1] for c in configs]
    final_te = [results[c['label']]['te_acc'][-1] for c in configs]
    x = np.arange(len(configs))
    axes[1,2].bar(x - 0.2, final_tr, 0.35, color=colors_C, alpha=0.5, label='train')
    axes[1,2].bar(x + 0.2, final_te, 0.35, color=colors_C, alpha=1.0, label='test')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels([c['label'].replace(' + ', '\n') for c in configs],
                               fontsize=7)
    axes[1,2].legend(fontsize=8)
    axes[1,2].set_ylim(0, 1.1)

    titles = [
        '(a) Test accuracy',
        '(b) Train accuracy',
        '(c) Generalization gap\n(train - test)',
        '(d) Loss curves\n(solid=train, dash=test)',
        '(e) Weight spectral ratio\n(lower = more uniform/symmetric)',
        '(f) Final accuracy summary\n(light=train, dark=test)',
    ]
    xlabels = ['epoch', 'epoch', 'epoch', 'epoch', 'epoch', 'condition']
    for ax, title, xl in zip(axes.flat, titles, xlabels):
        ax.set(title=title, xlabel=xl)
    axes[0,0].legend(fontsize=7, loc='lower right')
    axes[0,2].axhline(0, color='k', ls='--', lw=0.5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# ==============================================================================
# 実験 D: Stochastic Collapse — SGDノイズと不変集合の引力
# ==============================================================================
# 仮説: SGDのノイズ（ミニバッチランダム性）が対称性に対応した
#       不変集合（スパース解・同一ニューロン）に引き寄せる
#
# 先行研究:
#   Chen et al. (2023) "Stochastic Collapse"
#
# 実験設計:
#   - 教師NNより大きな生徒NNで回帰
#   - バッチサイズ大（≈全バッチ）vs 小（=強いノイズ）
#   - 生徒NNのニューロンの「同一化」（collapse）を観測
#   - ノイズ大 → より強い引力 → より単純な解 → 汎化改善
# ==============================================================================

class TeacherStudentSetup:
    def __init__(self, d=8, n_teacher=4, n_student=8, n=500, noise=0.05):
        """
        Teacher: n_teacher ニューロン（真の複雑度）
        Student: n_student ニューロン（過剰）→ 一部がcollapseして単純化
        """
        torch.manual_seed(42)
        # 教師NN（固定）
        self.teacher = nn.Sequential(
            nn.Linear(d, n_teacher, bias=False), nn.ReLU(),
            nn.Linear(n_teacher, 1, bias=False)
        )
        for p in self.teacher.parameters():
            nn.init.normal_(p, std=1.0)

        X = torch.randn(n, d)
        with torch.no_grad():
            Y = self.teacher(X) + noise * torch.randn(n, 1)

        n_tr = int(n * 0.7)
        self.X_tr, self.Y_tr = X[:n_tr], Y[:n_tr]
        self.X_te, self.Y_te = X[n_tr:], Y[n_tr:]
        self.d = d
        self.n_student = n_student

    def make_student(self):
        return nn.Sequential(
            nn.Linear(self.d, self.n_student, bias=False), nn.ReLU(),
            nn.Linear(self.n_student, 1, bias=False)
        )

    def neuron_diversity(self, student):
        """ニューロンの多様性: 高いほど同一化していない"""
        W = student[0].weight.detach()     # (n_student, d)
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (W_norm @ W_norm.T).abs()  # (n_student, n_student)
        mask = 1 - torch.eye(self.n_student)
        avg_sim = (cosine_sim * mask).sum() / mask.sum()
        return 1.0 - avg_sim.item()   # 1=全部異なる, 0=全部同一


def run_exp_D(batch_sizes=(8, 64, 512), n_epochs=3000,
              lr=1e-3, log_every=100):
    """
    実験D: バッチサイズ（=SGDノイズ）とニューロン多様性・汎化
    """
    setup = TeacherStudentSetup()
    results = {}

    for bs in batch_sizes:
        label = f'batch={bs}'
        student = setup.make_student()
        opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.MSELoss()

        H = dict(epoch=[], tr_loss=[], te_loss=[], diversity=[])

        for ep in range(n_epochs):
            student.train()
            # ミニバッチサンプリング
            idx = torch.randperm(len(setup.X_tr))[:bs]
            Xb, Yb = setup.X_tr[idx], setup.Y_tr[idx]
            loss = loss_fn(student(Xb), Yb)
            opt.zero_grad(); loss.backward(); opt.step()

            if ep % log_every == 0:
                student.eval()
                with torch.no_grad():
                    trl = loss_fn(student(setup.X_tr), setup.Y_tr).item()
                    tel = loss_fn(student(setup.X_te), setup.Y_te).item()
                div = setup.neuron_diversity(student)
                H['epoch'].append(ep); H['tr_loss'].append(trl)
                H['te_loss'].append(tel); H['diversity'].append(div)

        results[label] = H
        print(f"  {label}: tr_loss={H['tr_loss'][-1]:.4f}  "
              f"te_loss={H['te_loss'][-1]:.4f}  "
              f"diversity={H['diversity'][-1]:.3f}")

    return results


def plot_exp_D(results, outfile):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        'Exp D: Stochastic Collapse — SGD Noise Strength and Neuron Diversity\n'
        '(small batch = large noise = stronger attraction to invariant sets)',
        fontsize=10)

    colors_D = {'batch=8': '#E8593C', 'batch=64': '#4A90D9',
                'batch=512': '#1D9E75'}

    for label, H in results.items():
        c = colors_D.get(label, '#888')
        axes[0].semilogy(H['epoch'], H['tr_loss'], color=c, lw=2, label=f'{label} (train)')
        axes[0].semilogy(H['epoch'], H['te_loss'], color=c, lw=1.5, ls='--')
        axes[1].plot(H['epoch'], H['diversity'], color=c, lw=2, label=label)
        axes[2].scatter(H['diversity'], H['te_loss'],
                        c=H['epoch'], cmap='viridis', s=10, alpha=0.6)

    axes[0].set(title='(a) Loss curves\n(solid=train, dash=test)',
                xlabel='epoch', ylabel='MSE loss (log)')
    axes[0].legend(fontsize=8)
    axes[1].set(title='(b) Neuron diversity\n(low = neurons collapsed to identical)',
                xlabel='epoch', ylabel='diversity (0=all same, 1=all different)')
    axes[1].legend(fontsize=8)
    axes[2].set(title='(c) Diversity vs test loss\n(arrow = time direction)',
                xlabel='neuron diversity', ylabel='test loss')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# ==============================================================================
# メイン
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='A', choices=['A', 'B', 'C', 'D', 'all'])
    parser.add_argument('--outdir', default='/mnt/user-data/outputs')
    parser.add_argument('--in_dim', type=int,default=10)
    parser.add_argument('--out_dim', type=int,default=10)
    args = parser.parse_args()
    
    run_A = args.exp in ('A', 'all')
    run_B = args.exp in ('B', 'all')
    run_C = args.exp in ('C', 'all')
    run_D = args.exp in ('D', 'all')

    if run_A:
        cfg=LLCConfigs()
        depths=[4,5,6]
        banner("Experiment A: Deep Linear Network — Staged SV Acquisition")
        dims=10    
        in_dim=out_dim=dims
        results_A = run_exp_A(args,cfg,d=10, depths=depths, rank=2, n_steps=12000)
        plot_exp_A(results_A, depths, rank=2,
                   outfile=f'{args.outdir}/exp_A_deep_linear.png')

    if run_B:
        banner("Experiment B: Grokking — Symmetry Acquisition")
        print("NOTE: Requires ~100k steps to observe grokking. Set n_epochs accordingly.")
        results_B = run_exp_B(
            p=23, weight_decays=[0.0, 1.0, 5.0],
            n_epochs=5000,   # 本番は100000推奨。短いと汎化未観測
        )
        plot_exp_B(results_B, outfile=f'{args.outdir}/exp_B_grokking.png')

    if run_C:
        print("\n" + "="*60)
        print("Experiment C: Data-Network Symmetry Mismatch")
        print("="*60)
        configs_C = [
            dict(sym='high', width=256, label='High sym + wide NW',  color='#1D9E75', ls='-'),
            dict(sym='low',  width=256, label='Low sym  + wide NW',  color='#E8593C', ls='-'),
            dict(sym='high', width=16,  label='High sym + narrow NW',color='#4A90D9', ls='--'),
            dict(sym='low',  width=16,  label='Low sym  + narrow NW',color='#EF9F27', ls='--'),
        ]
        results_C = run_exp_C(configs_C, n_epochs=3000)
        plot_exp_C(results_C, configs_C,
                   outfile=f'{args.outdir}/exp_C_symmetry.png')

    if run_D:
        print("\n" + "="*60)
        print("Experiment D: Stochastic Collapse")
        print("="*60)
        results_D = run_exp_D(batch_sizes=[8, 64, 512], n_epochs=3000)
        plot_exp_D(results_D, outfile=f'{args.outdir}/exp_D_collapse.png')

    print("\nAll done.")


if __name__ == '__main__':
    main()
