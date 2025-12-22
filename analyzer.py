import re
import warnings
from contextlib import ExitStack
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hook import HookManager
from utils import LazyPipeline


class Analyzer(LazyPipeline):
    def __init__(self, model: nn.Module, device: str = "cuda", _steps: List[dict] = None):
        super().__init__(seed_obj=model, _steps=_steps)
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    def pipe(self, func: Callable, *args, **kwargs) -> 'Analyzer':
        step_info = {'func': func, 'args': args, 'kwargs': kwargs}
        return Analyzer(self.model, self.device, self._steps + [step_info])
    
    def calc_fire_rate(self, 
                       key: str = "fire_rate",
                       layer_pattern: str = r".*relu.*", 
                       with_avg: bool = True,
                       with_layer_rate: bool = False,
                       with_neuron_counts: bool = False) -> 'Analyzer':
        """
        指定された層の発火率（Fire Ratio）を算出する。
        
        Args:
            key: 結果を格納する辞書のキー名 (Default: "fire_rate")
            layer_pattern: 対象層の正規表現 (例: r".*relu.*")
            with_avg: 全体の平均発火率を含めるか (Default: True)
            with_layer_rate: 層ごとの発火率辞書を含めるか (Default: False)
            with_neuron_counts: 層ごとのニューロン数辞書を含めるか (Default: False)
        """
        
        def _calc_fire_rate(mgr: HookManager) -> Generator:
            # --- Setup Phase ---
            def _processor(out):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach()
                # (Rate, BatchSize, Neurons)
                return ((t > 0).float().mean().item(), t.shape[0], t[0].numel())
            
            mgr.register(layer_pattern, processor=_processor)

            yield # Wait for Inference

            # --- Aggregate Phase ---
            if not mgr.storage:
                return None

            layer_rates = {}
            layer_neurons = {}
            
            for name, data_list in mgr.storage.items():
                total_w = sum(r * s for r, s, _ in data_list)
                total_s = sum(s for _, s, _ in data_list)
                layer_rates[name] = total_w / total_s if total_s > 0 else 0.0
                layer_neurons[name] = data_list[0][2]

            ret = []
            
            # 1. 全体平均
            if with_avg:
                if layer_rates:
                    w_sum = sum(layer_rates[n] * layer_neurons[n] for n in layer_rates)
                    n_sum = sum(layer_neurons.values())
                    ret.append(w_sum / n_sum if n_sum > 0 else 0.0)
                else:
                    ret.append(0.0)
            
            # 2. 層別発火率
            if with_layer_rate:
                ret.append(layer_rates)

            # 3. ニューロン数
            if with_neuron_counts:
                ret.append(layer_neurons)

            # 結果の返却形式を調整
            result_val = None
            if len(ret) == 1: 
                result_val = ret[0]
            elif len(ret) > 1:
                result_val = tuple(ret)
            
            # (Key, Value) のペアを返して、flow側で辞書に格納できるようにする
            return key, result_val

        return self.pipe(_calc_fire_rate)

    def calc_dead_neurons(self, 
                          key: str = "dead_neurons",
                          layer_pattern: str = r".*relu.*", 
                          threshold: float = 0.0,
                          with_avg: bool = True,
                          with_layer_rate: bool = False,
                          with_neuron_counts: bool = False) -> 'Analyzer':
        """
        死にニューロン（全データセットを通して一度も発火しなかったニューロン）の割合を算出する。
        
        Args:
            key: 結果を格納する辞書のキー名 (Default: "dead_neurons")
            layer_pattern: 対象層の正規表現
            threshold: この値以下を「発火していない」とみなす（通常 0.0）
            with_avg: 全層の加重平均割合を返すか
            with_layer_rate: 層ごとの割合辞書を返すか
            with_neuron_counts: 層ごとのニューロン数辞書を返すか
        """
        
        def _calc_dead_neurons(mgr: HookManager) -> Generator:
            # --- Setup Phase ---
            
            def _processor(out):
                # 出力がタプルの場合のガード
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach()
                
                # メモリ節約のため、このバッチ内での「各ニューロンの最大値」のみを保存する
                # t: (Batch, C, H, W) -> バッチ次元(0)を潰して (C, H, W) の最大値を返す
                return t.amax(dim=0)
            
            mgr.register(layer_pattern, processor=_processor)

            yield # Wait for Inference (flowメソッド側で全データが流れるのを待つ)

            # --- Aggregate Phase ---
            if not mgr.storage:
                return None

            layer_dead_ratios = {}
            layer_neurons = {}

            for name, batch_max_list in mgr.storage.items():
                if not batch_max_list:
                    continue
                
                # 全バッチの最大値をまとめる -> (Num_Batches, C, H, W)
                # その上でさらに最大値を取る -> 全データセットを通した (C, H, W) の最大値
                # ※ batch_max_listはCPUにある前提(default_processor)だが、計算のためTensor操作を行う
                all_max_tensor = torch.stack(batch_max_list).amax(dim=0)
                
                # 閾値以下のニューロン数をカウント
                dead_count = (all_max_tensor <= threshold).sum().item()
                total_count = all_max_tensor.numel()
                
                layer_dead_ratios[name] = dead_count / total_count if total_count > 0 else 0.0
                layer_neurons[name] = total_count

            ret = []

            # 1. 全体平均 (死にニューロン総数 / 全ニューロン総数)
            if with_avg:
                if layer_neurons:
                    total_dead = sum(layer_dead_ratios[n] * layer_neurons[n] for n in layer_neurons)
                    total_all = sum(layer_neurons.values())
                    ret.append(total_dead / total_all if total_all > 0 else 0.0)
                else:
                    ret.append(0.0)

            # 2. 層別割合
            if with_layer_rate:
                ret.append(layer_dead_ratios)

            # 3. ニューロン数
            if with_neuron_counts:
                ret.append(layer_neurons)

            # 結果の返却形式を調整
            result_val = None
            if len(ret) == 1: 
                result_val = ret[0]
            elif len(ret) > 1:
                result_val = tuple(ret)
            
            return key, result_val

        return self.pipe(_calc_dead_neurons)

    def calc_linear_cka(self, 
                            key: str = "linear_cka",
                            layer_pattern: str = r".*conv.*", 
                            channels: int = None,
                            groups: int = None,
                            reduction: str = "mean",
                            with_layer_avg: bool = True,
                            with_matrix: bool = False,
                            with_group_info: bool = False) -> 'Analyzer':
            """
            指定した層ごとに Inter-Group Linear CKA を計算する。
            省メモリ化のため、O(N^2)のグラム行列ではなくO(C^2)の共分散行列を用いて計算を行う。
            
            Args:
                key (str): 結果を格納する辞書のキー。
                layer_pattern (str): 対象層の正規表現。 Default: r".*conv.*"
                channels (int, optional): 1グループあたりのチャネル幅。groupsと排他。
                groups (int, optional): 分割するグループ数。channelsと排他。
                reduction (str): 次元削減手法。'mean' (推奨) または 'flatten'。
                with_layer_avg (bool): Trueの場合、非対角成分(グループ間)の平均値辞書を作成する。 Default: True
                with_matrix (bool): Trueの場合、CKA行列全体の辞書を作成する。 Default: False
                with_group_info (bool): Trueの場合、(グループ数, 1グループのチャネル幅) のタプル辞書を作成する。 Default: False

            Returns:
                tuple: (key, result_tuple)
                    result_tuple (tuple): 設定されたフラグに対応する辞書のタプル。
                        順序は (dict_avg, dict_matrix, dict_info) の順で、Trueのもののみが含まれる。
                        
                        - dict_avg (dict): {層名: 非対角成分の平均値(float)}
                        - dict_matrix (dict): {層名: CKA行列(np.ndarray)}
                        - dict_info (dict): {層名: (グループ数(int), 1グループのチャネル幅(int))}

            Raises:
                ValueError: channels/groups の指定が不正な場合。
            """
            
            # --- Validation ---
            if (channels is not None) and (groups is not None):
                raise ValueError("Arguments 'channels' and 'groups' are mutually exclusive.")
            if (channels is None) and (groups is None):
                raise ValueError("Either 'channels' or 'groups' must be specified.")

            def _cka_processor(out):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach()
                # 次元汎用化: (B, C, ...) -> (B, C) or (B, C*Spatial)
                if t.dim() > 2:
                    if reduction == "mean":
                        t = t.flatten(2).mean(2)
                    elif reduction == "flatten":
                        t = t.flatten(2)
                return t

            def _compute_cka_via_covariance(features, num_groups):
                """共分散行列を用いた省メモリ Linear CKA 計算"""
                N, C_total = features.shape
                C_sub = C_total // num_groups
                
                # Centering & Covariance
                features_c = features - features.mean(dim=0, keepdim=True)
                cov_matrix = torch.mm(features_c.t(), features_c)

                # Block Reshape & Linear CKA Calculation
                cov_tensor = cov_matrix.view(num_groups, C_sub, num_groups, C_sub).permute(0, 2, 1, 3)
                numerator = torch.sum(cov_tensor ** 2, dim=(2, 3))
                diag_norms = torch.diagonal(numerator, offset=0, dim1=0, dim2=1)
                denominator = torch.sqrt(diag_norms.unsqueeze(1) * diag_norms.unsqueeze(0))
                
                return numerator / (denominator + 1e-8)

            def _calc_linear_cka(mgr: HookManager):
                mgr.register(layer_pattern, processor=_cka_processor)
                yield
                
                # 結果格納用辞書
                dict_avg = {}
                dict_matrix = {}
                dict_info = {}
                
                target_layers = [name for name in mgr.storage.keys() if re.fullmatch(layer_pattern, name)]
                
                for layer_name in target_layers:
                    data_list = mgr.storage[layer_name]
                    if not data_list:
                        continue
                    
                    # Tensor結合 & Flatten (if needed)
                    feats = torch.cat(data_list, dim=0)
                    if feats.dim() > 2:
                        feats = feats.flatten(1)

                    C_total = feats.shape[1]
                    
                    # --- Grouping Logic ---
                    if groups is not None:
                        num_groups = groups
                        if C_total % num_groups != 0:
                            warnings.warn(f"Skipping '{layer_name}': Channels ({C_total}) not divisible by groups ({num_groups}).")
                            continue
                        width = C_total // num_groups
                    else:
                        if C_total % channels != 0:
                            warnings.warn(f"Skipping '{layer_name}': Channels ({C_total}) not divisible by width ({channels}).")
                            continue
                        num_groups = C_total // channels
                        width = channels

                    # Compute CKA
                    cka_mat = _compute_cka_via_covariance(feats, num_groups)
                    
                    # --- Store Results ---
                    if with_layer_avg:
                        if num_groups > 1:
                            sum_all = torch.sum(cka_mat)
                            sum_diag = torch.trace(cka_mat)
                            avg_val = (sum_all - sum_diag) / (num_groups * num_groups - num_groups)
                            dict_avg[layer_name] = avg_val.item()
                        else:
                            dict_avg[layer_name] = 0.0
                    
                    if with_matrix:
                        dict_matrix[layer_name] = cka_mat.cpu().numpy()
                        
                    if with_group_info:
                        dict_info[layer_name] = (num_groups, width)
                
                # 要求されたデータのみをタプルとして返す
                result_tuple = []
                if with_layer_avg:
                    result_tuple.append(dict_avg)
                if with_matrix:
                    result_tuple.append(dict_matrix)
                if with_group_info:
                    result_tuple.append(dict_info)
                    
                return key, tuple(result_tuple)

            return self.pipe(_calc_linear_cka)

    def flow(self, dl: DataLoader) -> Dict[str, Any]:
        """
        パイプラインを実行し、結果を辞書形式で返す。
        
        Returns:
            Dict[str, Any]: {key: result, ...} の形式の辞書
        """
        results = {}
        generators = []
        managers = []

        with torch.no_grad(), ExitStack() as stack:
            # 1. Setup (Generator initialization)
            for step in self._steps:
                mgr = HookManager(self.model)
                stack.enter_context(mgr)
                managers.append(mgr)

                task_func = step['func']
                gen = task_func(mgr, *step['args'], **step['kwargs'])
                
                next(gen) # Run until yield
                generators.append(gen)
            
            # 2. Inference
            for inputs, _ in dl:
                _ = self.model(inputs.to(self.device))

        # 3. Aggregate (Resume Generator)
        for gen, mgr in zip(generators, managers):
            try:
                next(gen)
            except StopIteration as e:
                # e.value は (key, result_val) のタプルになっている
                if e.value is not None:
                    key, val = e.value
                    results[key] = val
            
            mgr.reset()
            mgr.remove_hooks()

        return results
