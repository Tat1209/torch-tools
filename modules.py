import torch
from typing import Callable, Optional
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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
    
class SchedulerCountingWrapper(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, base_scheduler: torch.optim.lr_scheduler._LRScheduler):
        """
        base_scheduler: ラップしたい既存の Scheduler インスタンス。
        """
        self._base_scheduler = base_scheduler
        self.call_count = 0

    def step(self, *args, **kwargs):
        """
        ステップを進めるたびに呼ばれる。呼び出し回数をインクリメントして出力し、
        実際のスケジューラの step() を呼ぶ。
        """
        self.call_count += 1
        print(f"Called {self.call_count} times.")
        return self._base_scheduler.step(*args, **kwargs)

    def __getattr__(self, name):
        """
        その他の属性やメソッドは、base_scheduler に委譲する。
        """
        return getattr(self._base_scheduler, name)