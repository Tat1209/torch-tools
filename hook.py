import torch
import torch.nn as nn
from collections import defaultdict
import re
from typing import List, Dict, Any, Callable

def default_processor(data: Any) -> Any:
    """
    デフォルトのプロセッサ。
    Tensorであれば計算グラフを切って(detach)CPUに移動させる。
    それ以外はそのまま返す。
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    return data

class HookManager:
    """
    PyTorchモデルの特定の層から中間出力を取得・加工するためのマネージャクラス。
    Context Managerとして使用可能。
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.storage: Dict[str, List[Any]] = defaultdict(list)

    def register(self, 
                 target_pattern: str, 
                 processor: Callable[[Any], Any] = default_processor, # デフォルト引数に指定
                 hook_type: str = 'forward',
                 pass_full_args: bool = False):
        """
        指定した正規表現パターンに一致する層にフックを登録する。

        Args:
            target_pattern: 層の名前を特定する正規表現
            processor: 出力を加工する関数。
                       デフォルトは `default_processor` (detach & cpu)。
                       Noneを返すとストレージに保存されない(コールバック用途)。
            hook_type: 'forward' または 'backward'
            pass_full_args: Trueの場合、processorに (module, input, output) を渡す。
                            Falseの場合、output (forward時) のみを渡す。
        """
        # 明示的に None が渡された場合のガード（デフォルト引数は引数省略時のみ効くため）
        if processor is None:
            processor = default_processor

        compiled_pattern = re.compile(target_pattern)
        matched = False

        for name, module in self.model.named_modules():
            if name == "":
                continue
            
            if compiled_pattern.fullmatch(name):
                matched = True
                hook_fn = self._create_hook(name, processor, hook_type, pass_full_args)
                
                if hook_type == 'forward':
                    handle = module.register_forward_hook(hook_fn)
                elif hook_type == 'backward':
                    handle = module.register_full_backward_hook(hook_fn)
                else:
                    raise ValueError(f"Unknown hook_type: {hook_type}")
                
                self.hooks.append(handle)

        if not matched:
            print(f"Warning: No modules matched pattern '{target_pattern}'")

    def _create_hook(self, name: str, processor: Callable, hook_type: str, pass_full_args: bool):
        def hook(module, inp, out):
            # データの正規化
            if hook_type == 'forward':
                data = out
            else:
                data = out[0] if isinstance(out, tuple) and len(out) == 1 else out
            
            # プロセッサの適用
            if pass_full_args:
                processed = processor(module, inp, out)
            else:
                processed = processor(data)

            # 結果を格納 (Noneの場合は保存しない)
            if processed is not None:
                self.storage[name].append(processed)
        
        return hook

    def reset(self):
        self.storage.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False