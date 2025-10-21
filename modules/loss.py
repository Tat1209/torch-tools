from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyLossT(nn.CrossEntropyLoss):
    def __init__(self, T: float = 1.0, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction, label_smoothing=label_smoothing)
        self.T = T

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input / self.T
        return super().forward(input, target)

class FixedLoss(nn.Module):
    def __init__(self, fixed_value=1.0):
        super(FixedLoss, self).__init__()
        self.fixed_value = fixed_value

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.tensor(self.fixed_value, requires_grad=True)
    