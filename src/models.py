"""
Flexible CNN with configurable:
  - num_layers    : エンコーダの層数
  - use_resnet    : ResNet (残差接続) の有無
  - dropout_rate  : Dropout 率 (0.0 で無効)
  - use_unet      : U-Net スキップ接続の有無
  - use_layernorm : LayerNorm の有無 (False なら BatchNorm)
  - activation    : 活性化関数 (デフォルト ReLU)

使用例:
    # シンプルな CNN
    model = FlexibleCNN(in_channels=3, num_classes=10)

    # ResNet + Dropout + LayerNorm
    model = FlexibleCNN(in_channels=3, num_classes=10,
                        use_resnet=True, dropout_rate=0.3, use_layernorm=True)

    # U-Net (セグメンテーション用)
    model = FlexibleCNN(in_channels=3, num_classes=10,
                        use_unet=True, use_resnet=True, dropout_rate=0.2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# ユーティリティ: Norm ブロック
# ---------------------------------------------------------------------------
def make_norm(use_layernorm: bool, num_channels: int, spatial_size: Optional[int] = None):
    """LayerNorm or BatchNorm2d を返す"""
    if use_layernorm:
        # LayerNorm を Conv 特徴マップへ適用するには [C, H, W] を正規化
        # spatial_size が既知なら指定、なければ channels のみ
        if spatial_size is not None:
            return nn.LayerNorm([num_channels, spatial_size, spatial_size])
        # 動的サイズ対応: GroupNorm(1, C) ≈ LayerNorm for feature maps
        return nn.GroupNorm(1, num_channels)
    return nn.BatchNorm2d(num_channels)

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
# ---------------------------------------------------------------------------
# Simple models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,num_layers:int):
        super().__init__()
        self.num_layers=num_layers
        self.layers=[]
        self.layers.append(nn.Linear(in_dim, hidden))
        for l in range(1,num_layers-1):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(in_dim, hidden))
        # He init
        for l in range(num_layers):        
            nn.init.kaiming_normal_(self.layers[l].weight)
            nn.init.zeros_(self.layers[l].bias)
        
    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        h = F.relu(self.layers[0](x))
        for l in range(1,self.num_layers-1):
            h = F.relu(self.layers[l](h))
            #nn.softmax
        logits = self.layers[-1](h)
        return alpha * logits  # softmax coefficient α multiplies logits

class CNNs(nn.Module):
    def __init__(self, in_width: int,in_height:int,kernelsize:int, hidden: int, out_dim: int,num_layers:int):
        super().__init__()
        self.num_layers=num_layers
        self.layers=[]
        self.kernelsize=kernelsize
        self.layers.append(nn.conv2d(in_width,in_height,kernelsize))
        for l in range(1,num_layers-1):
            self.layers.append(nn.conv2d(hidden, hidden))
        self.layers.append(nn.conv2d(in_width,in_height, kernelsize))
        
    def forward(self, x: torch.Tensor):
        h = F.relu(self.layers[0](x))
        for l in range(1,self.num_layers-1):
            h = F.relu(self.layers[l](h))
        logits = self.layers[-1](h)
        return logits  # softmax coefficient α multiplies logits

# https://arxiv.org/pdf/2512.20607 Appendix I
class BottleNeckNet(nn.Module):
    def __init__(self, in_width: int,in_height:int,kernelsize:int, hidden: int, out_dim: int,num_layers:int):    
        pass

        torch.nn.Conv1d(in_channels=1,
                        out_channels=50,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        dilation=1,
                        groups=1,
                        bias=False)

# ---------------------------------------------------------------------------
# Conv ブロック
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv → Norm → ReLU [→ Dropout] の基本ブロック"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm = make_norm(use_layernorm, out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


# ---------------------------------------------------------------------------
# ResNet ブロック
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """2 層の ConvBlock + shortcut 接続"""

    def __init__(
        self,
        channels: int,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, use_layernorm=use_layernorm, dropout_rate=dropout_rate),
            ConvBlock(channels, channels, use_layernorm=use_layernorm, dropout_rate=0.0),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


# ---------------------------------------------------------------------------
# エンコーダ層 (downsampling)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_resnet: bool = False,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch,
                              use_layernorm=use_layernorm,
                              dropout_rate=dropout_rate)
        self.res  = ResBlock(out_ch, use_layernorm=use_layernorm,
                             dropout_rate=dropout_rate) if use_resnet else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.res(x)
        return self.pool(x), x          # pooled, skip


# ---------------------------------------------------------------------------
# デコーダ層 (upsampling / U-Net)
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_resnet: bool = False,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch,  # skip concat → *2
                              use_layernorm=use_layernorm,
                              dropout_rate=dropout_rate)
        self.res  = ResBlock(out_ch, use_layernorm=use_layernorm,
                             dropout_rate=dropout_rate) if use_resnet else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # サイズ不一致を吸収
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.conv(x))


# ---------------------------------------------------------------------------
# メインモデル: FlexibleCNN
# ---------------------------------------------------------------------------
class FlexibleCNN(nn.Module):
    """
    Args:
        in_channels   : 入力チャンネル数
        num_classes   : 出力クラス数
        base_channels : 最初の Conv のチャンネル数 (倍々で増加)
        num_layers    : エンコーダの深さ (1〜N)
        use_resnet    : 各層に ResBlock を追加するか
        dropout_rate  : Dropout 率 (0.0 で無効)
        use_unet      : U-Net スキップ接続でデコーダを追加するか
        use_layernorm : True → LayerNorm / False → BatchNorm
        task          : "classification" or "segmentation"
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
        num_layers: int = 4,
        use_resnet: bool = False,
        dropout_rate: float = 0.0,
        use_unet: bool = False,
        use_layernorm: bool = False,
        task: str = "classification",
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers は 1 以上を指定してください"
        self.use_unet = use_unet
        self.task     = task

        # ---- エンコーダ ----
        self.encoders = nn.ModuleList()
        ch = in_channels
        self.enc_channels = []
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(
                EncoderLayer(ch, out_ch,
                             use_resnet=use_resnet,
                             use_layernorm=use_layernorm,
                             dropout_rate=dropout_rate)
            )
            self.enc_channels.append(out_ch)
            ch = out_ch

        # ---- ボトルネック ----
        bottleneck_ch = ch * 2
        self.bottleneck = nn.Sequential(
            ConvBlock(ch, bottleneck_ch,
                      use_layernorm=use_layernorm,
                      dropout_rate=dropout_rate),
            ResBlock(bottleneck_ch,
                     use_layernorm=use_layernorm,
                     dropout_rate=dropout_rate) if use_resnet else nn.Identity(),
        )
        ch = bottleneck_ch

        # ---- デコーダ (U-Net) ----
        if use_unet:
            self.decoders = nn.ModuleList()
            for skip_ch in reversed(self.enc_channels):
                out_ch = skip_ch
                self.decoders.append(
                    DecoderLayer(ch, out_ch,
                                 use_resnet=use_resnet,
                                 use_layernorm=use_layernorm,
                                 dropout_rate=dropout_rate)
                )
                ch = out_ch

            self.seg_head = nn.Conv2d(ch, num_classes, kernel_size=1)

        # ---- 分類ヘッド ----
        if not use_unet or task == "classification":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier  = nn.Sequential(
                nn.Flatten(),
                nn.Linear(ch, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        skips = []

        # エンコーダ
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # ボトルネック
        x = self.bottleneck(x)

        # デコーダ (U-Net)
        if self.use_unet:
            for dec, skip in zip(self.decoders, reversed(skips)):
                x = dec(x, skip)

            if self.task == "segmentation":
                return self.seg_head(x)

        # 分類
        return alpha*self.classifier(self.global_pool(x))


# ---------------------------------------------------------------------------
# 動作確認 & パラメータ数表示
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo():
    configs = [
        dict(name="シンプル CNN",
             kwargs=dict(in_channels=3, num_classes=10, num_layers=3)),

        dict(name="ResNet + Dropout + BatchNorm",
             kwargs=dict(in_channels=3, num_classes=10, num_layers=4,
                         use_resnet=True, dropout_rate=0.3)),

        dict(name="LayerNorm のみ",
             kwargs=dict(in_channels=3, num_classes=10, num_layers=3,
                         use_layernorm=True)),

        dict(name="U-Net (分類)",
             kwargs=dict(in_channels=3, num_classes=10, num_layers=3,
                         use_unet=True, use_resnet=True, dropout_rate=0.2)),

        dict(name="U-Net (セグメンテーション)",
             kwargs=dict(in_channels=3, num_classes=10, num_layers=4,
                         use_unet=True, use_resnet=True, dropout_rate=0.2,
                         use_layernorm=True, task="segmentation")),

        dict(name="全部入り",
             kwargs=dict(in_channels=3, num_classes=10, num_layers=5,
                         use_resnet=True, dropout_rate=0.3,
                         use_unet=True, use_layernorm=True)),
    ]

    x_cls = torch.randn(2, 3, 64, 64)
    x_seg = torch.randn(2, 3, 128, 128)

    print("=" * 60)
    for cfg in configs:
        model = FlexibleCNN(**cfg["kwargs"])
        model.eval()
        x = x_seg if cfg["kwargs"].get("task") == "segmentation" else x_cls
        with torch.no_grad():
            out = model(x)
        params = count_params(model)
        print(f"[{cfg['name']}]")
        print(f"  入力: {tuple(x.shape)}  出力: {tuple(out.shape)}")
        print(f"  パラメータ数: {params:,}")
        print()


if __name__ == "__main__":
    demo()
