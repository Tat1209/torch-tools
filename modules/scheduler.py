from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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