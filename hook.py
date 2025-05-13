import torch
import torch.nn as nn
from typing import Callable, Dict, List, Union
from collections import defaultdict

class HookManager:
    def __init__(self, model: nn.Module):
        """
        Args:
            model: フックを登録する対象の nn.Module（モデル全体 or 部分モジュール）
        """
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.storage = defaultdict(list)

    def _wrap_hook(self, name: str, fn: Callable):
        """内部用：フック関数をラップして storage に格納"""
        def hook(module, input, output):
            # detach してから CPU 保存
            obj = module, input, output
            if fn:
                obj = fn(module, input, output)
            self.storage[name].append(obj)
        return hook

    def register_forward(self, module: nn.Module, name: str, fn: Callable = None):
        """
        フォワードフックを登録
        
        Args:
            module: 対象モジュール
            name: 保存先キー
            fn: （任意）追加用コールバック
        """
        h = module.register_forward_hook(self._wrap_hook(name, fn))
        self.handles.append(h)

    def register_backward(self, module: nn.Module, name: str, fn: Callable = None):
        """
        バックワードフックを登録（勾配キャプチャ用）
        
        Args:
            module: 対象モジュール
            name: 保存先キー
            fn: （任意）追加用コールバック
        """
        h = module.register_full_backward_hook(self._wrap_hook(name, fn))
        self.handles.append(h)

    def get(self, name: str) -> Union[torch.Tensor, None]:
        """保存されたテンソルを取得"""
        return self.storage.get(name, None)

    def clear(self):
        """保存ストレージをクリア"""
        self.storage.clear()

    def remove(self):
        """全フックを解除"""
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
        self.clear()
