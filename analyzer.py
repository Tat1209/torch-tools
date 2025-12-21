import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Callable, Tuple, Union, Literal
import numpy as np

from hook import HookManager


class Analyzer:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.hook_manager = HookManager(self.model)

    def calc_fire_rate(self, 
                       dl: DataLoader, 
                       layer_pattern: str = r".*relu.*", 
                       with_avg: bool = True,
                       with_layer_rates: bool = False,
                       with_neuron_counts: bool = False
                       ) -> Union[float, Tuple, Dict[str, float], Dict[str, int], None]:
        """
        指定された層の発火率(Activation Rate)を算出する。
        引数のフラグに応じて、必要な情報をタプル（要素が1つの場合はその値）で返す。

        Args:
            dl: 評価用データローダ
            layer_pattern: 対象層の正規表現 (例: r".*relu.*")
            with_avg: 全体の平均発火率を含めるか (Default: True)
            with_layer_rates: 層ごとの発火率辞書を含めるか (Default: False)
            with_neuron_counts: 層ごとのニューロン数辞書を含めるか (Default: False)

        Returns:
            選択された要素を含むタプル、または単一の値。None(データなし)。
            返却順序は引数の並びと同様 ( [avg], [layer_rates], [layer_neurons] ) となる。
        """
        
        # 内部関数: データを圧縮して保存するプロセッサ
        def _fire_rate_processor(out: torch.Tensor) -> Tuple[float, int, int]:
            # 出力がTensorでない場合のガード
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach()
            # (バッチ内の平均発火率, バッチサイズ, 1サンプルあたりのニューロン数)
            return ((t > 0).float().mean().item(), t.shape[0], t[0].numel())

        # 1. フック登録
        self.hook_manager.register(layer_pattern, processor=_fire_rate_processor)

        # 2. 推論実行 (データの蓄積)
        # HookManagerの仕様変更により、withブロックを抜けてもデータは保持される
        with torch.no_grad(), self.hook_manager:
            for inputs, _ in dl:
                _ = self.model(inputs.to(self.device))

        # 3. 集計処理
        layer_rates: Dict[str, float] = {}
        layer_neurons: Dict[str, int] = {}
        
        # データが取れていない場合のガード
        if not self.hook_manager.storage:
            self.hook_manager.reset()
            return None

        for name, data_list in self.hook_manager.storage.items():
            total_weighted_rate = 0.0
            total_samples = 0
            
            for rate, size, _ in data_list:
                total_weighted_rate += rate * size
                total_samples += size
            
            layer_rates[name] = total_weighted_rate / total_samples if total_samples > 0 else 0.0
            # ニューロン数は最初のバッチの情報を参照
            layer_neurons[name] = data_list[0][2] if data_list else 0

        # 4. クリーンアップ (次の実験のためにリセット)
        self.hook_manager.reset()
        self.hook_manager.remove_hooks() # 念のため

        # 5. 返り値の構築 (引数の順序通り)
        ret_list = []

        # [1] 平均発火率
        if with_avg:
            if layer_rates:
                total_weighted_sum = sum(layer_rates[n] * layer_neurons[n] for n in layer_rates)
                total_neuron_sum = sum(layer_neurons.values())
                avg_val = total_weighted_sum / total_neuron_sum if total_neuron_sum > 0 else 0.0
            else:
                avg_val = 0.0
            ret_list.append(float(avg_val))

        # [2] 層別発火率
        if with_layer_rates:
            ret_list.append(layer_rates)

        # [3] 層別ニューロン数
        if with_neuron_counts:
            ret_list.append(layer_neurons)

        # 返り値の型調整
        if len(ret_list) == 0:
            return None 
        elif len(ret_list) == 1:
            return ret_list[0]
        else:
            return tuple(ret_list)