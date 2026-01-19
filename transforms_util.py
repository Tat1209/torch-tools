import torch
import torch.nn as nn
from typing import Any

class GrayToRGB(nn.Module):
    """1チャネル(グレースケール)画像を3チャネルに拡張する変換クラス。
    
    3チャネル(RGB)画像の場合は何もしない(Identity)ため、
    色情報を維持したまま、データセット内のチャネル数を統一できる。
    multiprocessing (Pickle) に対応。
    """

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """変換を実行する。

        Args:
            img (torch.Tensor): 入力画像テンソル (C, H, W) または (B, C, H, W)。

        Returns:
            torch.Tensor: 3チャネル化された画像テンソル。
        """
        # 画像のチャネル次元を確認 ( (C, H, W) なら index 0, (..., C, H, W) なら index -3 )
        # v2.ToImage() 直後なら通常 (3, H, W) などの形式
        if img.shape[-3] == 1:
            return img.repeat(3, 1, 1)
        return img

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"
