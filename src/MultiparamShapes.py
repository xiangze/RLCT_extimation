from dataclasses import dataclass
from typing import List, Tuple
import torch, torch.nn.functional as F

@dataclass
class MultiParamShapes:
    # 例: [in_dim, 64, 64, out_dim] なら 隠れ2層
    layer_sizes: List[int]  # [d0, d1, ..., dL], d0=in, dL=out

    @property
    def total(self) -> int:
        tot = 0
        for i in range(len(self.layer_sizes)-1):
            din, dout = self.layer_sizes[i], self.layer_sizes[i+1]
            tot += dout*din + dout  # W_i と b_i
        return tot

    def unpack(self, w: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        w: (D,) または (S, D) のフラットパラメタ
        戻り値: [W0,...,W_{L-1}], [b0,...,b_{L-1}]
          ただし W_i: (S, d_{i+1}, d_i), b_i: (S, d_{i+1})
          S はサンプル数（wが1次元なら S=1 として返す）
        """
        if w.ndim == 1:
            w = w.unsqueeze(0)  # (1, D)
        S, D = w.shape
        offs = 0
        W_list, b_list = [], []
        for i in range(len(self.layer_sizes)-1):
            din, dout = self.layer_sizes[i], self.layer_sizes[i+1]
            nW = dout*din
            W = w[:, offs:offs+nW].reshape(S, dout, din)
            offs += nW
            b = w[:, offs:offs+dout]
            offs += dout
            W_list.append(W); b_list.append(b)
        assert offs == D
        return W_list, b_list

def logits_from_w_multi(
    X: torch.Tensor,                # (N, d0)
    w: torch.Tensor,                # (D,) or (S, D)
    shapes: MultiParamShapes,
    alpha: float,
    activation: str = "relu"
) -> torch.Tensor:
    
    """
    戻り値: (S, N, dL)  （wが( D,)だったら S=1 で返る）
    最後の層は線形のまま（logits）。中間層にのみ非線形を適用。
    """
    W_list, b_list = shapes.unpack(w)     # W_i: (S, d_{i+1}, d_i), b_i: (S, d_{i+1})
    S = W_list[0].shape[0]
    N = X.shape[0]
    H = X.expand(S, N, X.shape[1])        # (S, N, d0)
    for ell, (W, b) in enumerate(zip(W_list, b_list)):
        H = torch.matmul(H, W.transpose(-1, -2)) + b.unsqueeze(1)  # (S, N, d_{ell+1})
        if ell < len(W_list) - 1:
            if activation == "relu":
                H = F.relu(H)
            elif activation == "identity":
                pass
            else:
                raise ValueError("activation must be relu or identity")
    return alpha * H  # logits



