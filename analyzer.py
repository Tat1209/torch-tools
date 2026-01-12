import re
import warnings
from contextlib import ExitStack
from functools import partial
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

    def _infer_layer_info(self, layer_pattern: str) -> Tuple[int, Optional[int]]:
        """指定パターンに一致する層からグループ数と出力特徴量数を推定する。

        Args:
            layer_pattern (str): 対象層の正規表現。

        Returns:
            Tuple[int, Optional[int]]: (groups, out_features)。推定不可時は (1, None)。
        """
        for name, mod in self.model.named_modules():
            if re.fullmatch(layer_pattern, name):
                groups = getattr(mod, 'groups', 1)
                out_features = getattr(mod, 'out_features', getattr(mod, 'out_channels', None))
                return groups, out_features
        return 1, None

    def calc_activation_rate(self,
                             key: str = "activation_rate",
                             layer_pattern: str = r".*relu.*",
                             groups: Optional[int] = None,
                             ref_pattern: str = r".*fc.*",
                             with_avg: bool = True,
                             with_layer: bool = False,
                             with_counts: bool = False,
                             with_path: bool = False,
                             with_layer_path: bool = False) -> 'Analyzer':
        """活性化密度（Activation Density / Sparsity）を算出する。

        GPU上で統計量を累積更新し、メモリ転送とPythonオーバーヘッドを最小化する。

        Args:
            key (str): 結果格納キー。
            layer_pattern (str): 対象層の正規表現。
            groups (Optional[int]): パス数。
            ref_pattern (str): groups推定用の参照層パターン。
            with_avg (bool): 'avg' を含めるか。
            with_layer (bool): 'layer' を含めるか。
            with_counts (bool): 'counts' を含めるか。
            with_path (bool): 'path' を含めるか。
            with_layer_path (bool): 'layer_path' を含めるか。

        Returns:
            Analyzer: パイプラインオブジェクト。
        """
        num_groups = groups
        if num_groups is None:
            num_groups, _ = self._infer_layer_info(ref_pattern)

        def _calc_activation(mgr: HookManager) -> Generator:
            # {layer_name: [accumulated_active_tensor, accumulated_total_scalar]}
            stats = {}

            def _processor(out, name: str):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach()

                batch_size = t.shape[0]
                channels = t.shape[1]

                if channels % num_groups != 0:
                    current_active = (t > 0).float().sum().view(1)
                    current_total = torch.tensor(t.numel(), device=t.device)
                else:
                    t_grouped = t.view(batch_size, num_groups, -1)
                    # (Groups,)
                    current_active = (t_grouped > 0).float().sum(dim=(0, 2))
                    current_total = torch.tensor(t_grouped.shape[0] * t_grouped.shape[2], device=t.device)

                if name not in stats:
                    stats[name] = [current_active, current_total]
                else:
                    stats[name][0] += current_active
                    stats[name][1] += current_total

                return None

            target_found = False
            for name, _ in self.model.named_modules():
                if re.fullmatch(layer_pattern, name):
                    target_found = True
                    p = partial(_processor, name=name)
                    mgr.register(re.escape(name), processor=p)

            if not target_found:
                warnings.warn(f"[{key}] No layers matched pattern '{layer_pattern}'")

            yield

            if not stats:
                return None

            result_dict = {}
            layer_rates = {}
            layer_path_rates = {}
            layer_neurons = {}

            total_active_per_group = torch.zeros(num_groups)
            total_elements_per_group = 0

            for name, (sum_active_gpu, sum_total_gpu) in stats.items():
                sum_active = sum_active_gpu.cpu()
                sum_total_scalar = sum_total_gpu.item()

                if sum_active.numel() != num_groups:
                    l_rate = (sum_active.sum() / sum_total_scalar).item() if sum_total_scalar > 0 else 0.0
                    layer_rates[name] = l_rate
                    layer_neurons[name] = sum_total_scalar
                    layer_path_rates[name] = [l_rate] * num_groups
                else:
                    path_rates_tensor = sum_active / sum_total_scalar
                    layer_rates[name] = path_rates_tensor.mean().item()
                    layer_path_rates[name] = path_rates_tensor.tolist()
                    layer_neurons[name] = sum_total_scalar * num_groups

                    total_active_per_group += sum_active
                    total_elements_per_group += sum_total_scalar

            if with_avg:
                total_elems_all = total_elements_per_group * num_groups
                val = (total_active_per_group.sum() / total_elems_all).item() if total_elems_all > 0 else 0.0
                result_dict['avg'] = val

            if with_layer:
                result_dict['layer'] = layer_rates

            if with_counts:
                result_dict['counts'] = layer_neurons

            if with_path:
                val = (total_active_per_group / total_elements_per_group).tolist() if total_elements_per_group > 0 else [0.0] * num_groups
                result_dict['path'] = val

            if with_layer_path:
                result_dict['layer_path'] = layer_path_rates

            return key, result_dict

        return self.pipe(_calc_activation)

    def calc_channel_survival_rate(self,
                                   key: str = "survival_rate",
                                   layer_pattern: str = r".*relu.*",
                                   threshold: float = 0.0,
                                   groups: Optional[int] = None,
                                   ref_pattern: str = r".*fc.*",
                                   with_avg: bool = True,
                                   with_layer: bool = False,
                                   with_counts: bool = False,
                                   with_path: bool = False,
                                   with_layer_path: bool = False) -> 'Analyzer':
        """チャネル生存率（Channel Survival Rate）を算出する。

        GPU上でRunning Max計算（Fused Reduction & In-place更新）を行い、高速に判定する。

        Args:
            key (str): 結果格納キー。
            layer_pattern (str): 対象層の正規表現。
            threshold (float): 生存判定しきい値。
            groups (Optional[int]): パス数。
            ref_pattern (str): groups推定用の参照層パターン。
            with_avg (bool): 'avg' を含めるか。
            with_layer (bool): 'layer' を含めるか。
            with_counts (bool): 'counts' を含めるか。
            with_path (bool): 'path' を含めるか。
            with_layer_path (bool): 'layer_path' を含めるか。

        Returns:
            Analyzer: パイプラインオブジェクト。
        """
        num_groups = groups
        if num_groups is None:
            num_groups, _ = self._infer_layer_info(ref_pattern)

        def _calc_survival(mgr: HookManager) -> Generator:
            # {layer_name: max_tensor_gpu}
            stats = {}

            def _processor(out, name: str):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach()

                # Fused Reduction: (Batch, C, H, W) -> (C,)
                if t.ndim == 4:
                    batch_max = t.amax(dim=(0, 2, 3))
                elif t.ndim == 2:
                    batch_max = t.amax(dim=0)
                else:
                    dims = [0] + list(range(2, t.ndim))
                    batch_max = t.amax(dim=dims)

                # In-place Update
                if name not in stats:
                    stats[name] = batch_max
                else:
                    torch.max(stats[name], batch_max, out=stats[name])

                return None

            target_found = False
            for name, _ in self.model.named_modules():
                if re.fullmatch(layer_pattern, name):
                    target_found = True
                    p = partial(_processor, name=name)
                    mgr.register(re.escape(name), processor=p)

            if not target_found:
                warnings.warn(f"[{key}] No layers matched pattern '{layer_pattern}'")

            yield

            if not stats:
                return None

            layer_survivals = {}
            layer_path_survivals = {}
            layer_counts = {}

            total_survived_per_group = torch.zeros(num_groups)
            total_channels_per_group = 0

            for name, max_tensor_gpu in stats.items():
                all_max_tensor = max_tensor_gpu.cpu()
                total_c = all_max_tensor.numel()
                layer_counts[name] = total_c

                if total_c % num_groups != 0:
                    survived_count = (all_max_tensor > threshold).sum().item()
                    layer_survivals[name] = survived_count / total_c
                    continue

                grouped_max = all_max_tensor.view(num_groups, -1)
                survived_per_group = (grouped_max > threshold).sum(dim=1).float()
                c_sub = grouped_max.size(1)

                path_ratios = (survived_per_group / c_sub).tolist()
                layer_path_survivals[name] = path_ratios
                layer_survivals[name] = sum(path_ratios) / num_groups

                total_survived_per_group += survived_per_group
                total_channels_per_group += c_sub

            result_dict = {}

            if with_avg:
                div = total_channels_per_group * num_groups
                val = (total_survived_per_group.sum() / div).item() if div > 0 else 0.0
                result_dict['avg'] = val

            if with_layer:
                result_dict['layer'] = layer_survivals

            if with_counts:
                result_dict['counts'] = layer_counts

            if with_path:
                val = (total_survived_per_group / total_channels_per_group).tolist() if total_channels_per_group > 0 else [0.0] * num_groups
                result_dict['path'] = val

            if with_layer_path:
                result_dict['layer_path'] = layer_path_survivals

            return key, result_dict

        return self.pipe(_calc_survival)

    def calc_magnitude(self,
                       key: str = "magnitude",
                       layer_pattern: str = r".*relu.*",
                       threshold: float = 1e-6,
                       groups: Optional[int] = None,
                       ref_pattern: str = r".*fc.*",
                       spatial_mode: str = "channel",
                       with_avg: bool = True,
                       with_layer: bool = False,
                       with_counts: bool = False,
                       with_path: bool = False,
                       with_layer_path: bool = False) -> 'Analyzer':
        """層のMagnitude（活性化強度）を算出する。

        Global Mean（全体平均）と Active Mean（発火時平均）を計算する。

        Args:
            key (str): 結果格納キー。
            layer_pattern (str): 対象層の正規表現。
            threshold (float): Active判定しきい値。
            groups (Optional[int]): パス数。
            ref_pattern (str): groups推定用の参照層パターン。
            spatial_mode (str): 'channel' or 'pixel'。
            with_avg (bool): 'avg', 'active_avg' を含めるか。
            with_layer (bool): 'layer', 'layer_active' を含めるか。
            with_counts (bool): 'counts' を含めるか。
            with_path (bool): 'path', 'path_active' を含めるか。
            with_layer_path (bool): 'layer_path', 'layer_active_path' を含めるか。

        Returns:
            Analyzer: パイプラインオブジェクト。
        """
        num_groups = groups
        if num_groups is None:
            num_groups, _ = self._infer_layer_info(ref_pattern)

        def _calc_magnitude(mgr: HookManager) -> Generator:
            def _processor(out):
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach()

                if t.ndim > 2:
                    if spatial_mode == "channel":
                        t = t.flatten(2).abs().mean(dim=2)
                    elif spatial_mode == "pixel":
                        b, c = t.shape[:2]
                        t = t.permute(0, 2, 3, 1).reshape(-1, c).abs()
                    else:
                        raise ValueError(f"Invalid spatial_mode: {spatial_mode}")
                else:
                    t = t.abs()

                is_active = t > threshold
                sum_all = t.sum(dim=0)
                sum_active = (t * is_active.float()).sum(dim=0)
                count_active = is_active.float().sum(dim=0)
                total_samples = t.shape[0]

                return (sum_all, sum_active, count_active, total_samples)

            mgr.register(layer_pattern, processor=_processor)

            yield

            if not mgr.storage:
                return None

            res_global = {'layer': {}, 'path': {}, 'layer_path': {}}
            res_active = {'layer': {}, 'path': {}, 'layer_path': {}}
            res_counts = {}

            acc_global = {'sum': torch.zeros(num_groups), 'count': 0}
            acc_active = {'sum': torch.zeros(num_groups), 'count': torch.zeros(num_groups)}

            for name, data_list in mgr.storage.items():
                if not data_list:
                    continue

                layer_sum_all = 0
                layer_sum_active = 0
                layer_count_active = 0
                layer_total_samples = 0

                for sa, sac, cac, n in data_list:
                    layer_sum_all += sa
                    layer_sum_active += sac
                    layer_count_active += cac
                    layer_total_samples += n

                layer_sum_all = layer_sum_all.cpu()
                layer_sum_active = layer_sum_active.cpu()
                layer_count_active = layer_count_active.cpu()

                channels = layer_sum_all.numel()
                res_counts[name] = channels

                if channels % num_groups != 0:
                    g_mean = (layer_sum_all / layer_total_samples).mean().item()
                    total_active_c = layer_count_active.sum().item()
                    a_mean = (layer_sum_active.sum().item() / total_active_c) if total_active_c > 0 else 0.0

                    res_global['layer'][name] = g_mean
                    res_active['layer'][name] = a_mean
                    continue

                grouped_sum_all = layer_sum_all.view(num_groups, -1)
                grouped_sum_active = layer_sum_active.view(num_groups, -1)
                grouped_count_active = layer_count_active.view(num_groups, -1)
                c_sub = grouped_sum_all.size(1)

                # Global Metrics
                g_mean_per_group = grouped_sum_all.sum(dim=1) / (layer_total_samples * c_sub)
                res_global['layer'][name] = g_mean_per_group.mean().item()
                res_global['layer_path'][name] = g_mean_per_group.tolist()

                acc_global['sum'] += g_mean_per_group * c_sub
                acc_global['count'] += c_sub

                # Active Metrics
                grp_active_sum = grouped_sum_active.sum(dim=1)
                grp_active_cnt = grouped_count_active.sum(dim=1)

                a_mean_per_group = torch.zeros_like(grp_active_sum)
                mask_nz = grp_active_cnt > 0
                a_mean_per_group[mask_nz] = grp_active_sum[mask_nz] / grp_active_cnt[mask_nz]

                res_active['layer'][name] = a_mean_per_group.mean().item()
                res_active['layer_path'][name] = a_mean_per_group.tolist()

                acc_active['sum'] += grp_active_sum
                acc_active['count'] += grp_active_cnt

            result_dict = {}

            if with_avg:
                total_neurons = acc_global['count']
                div_g = total_neurons * num_groups
                result_dict['avg'] = (acc_global['sum'].sum() / div_g).item() if total_neurons > 0 else 0.0

                total_active_sum = acc_active['sum'].sum()
                total_active_cnt = acc_active['count'].sum()
                result_dict['active_avg'] = (total_active_sum / total_active_cnt).item() if total_active_cnt > 0 else 0.0

            if with_layer:
                result_dict['layer'] = res_global['layer']
                result_dict['layer_active'] = res_active['layer']

            if with_counts:
                result_dict['counts'] = res_counts

            if with_path:
                div_p = acc_global['count']
                result_dict['path'] = (acc_global['sum'] / div_p).tolist() if div_p > 0 else [0.0] * num_groups

                p_act_sum = acc_active['sum']
                p_act_cnt = acc_active['count']
                path_active_list = []
                for s, c in zip(p_act_sum, p_act_cnt):
                    path_active_list.append((s / c).item() if c > 0 else 0.0)
                result_dict['path_active'] = path_active_list

            if with_layer_path:
                result_dict['layer_path'] = res_global['layer_path']
                result_dict['layer_active_path'] = res_active['layer_path']

            return key, result_dict

        return self.pipe(_calc_magnitude)

    def calc_linear_cka(self,
                        key: str = "linear_cka",
                        layer_pattern: str = r".*conv.*",
                        channels: int = None,
                        groups: int = None,
                        reduction: str = "mean",
                        with_avg: bool = True,
                        with_layer: bool = True,
                        with_counts: bool = False,
                        with_matrix: bool = False,
                        with_group_info: bool = False) -> 'Analyzer':
        """Inter-Group Linear CKA を計算する。

        Args:
            key (str): 結果格納キー。
            layer_pattern (str): 対象層の正規表現。
            channels (int): 1グループあたりのチャネル幅。
            groups (int): グループ数。channelsと排他。
            reduction (str): 'mean' or 'flatten'。
            with_avg (bool): 'avg' を含めるか。
            with_layer (bool): 'layer' を含めるか。
            with_counts (bool): 'counts' を含めるか。
            with_matrix (bool): 'matrix' を含めるか。
            with_group_info (bool): 'group_info' を含めるか。

        Returns:
            Analyzer: パイプラインオブジェクト。
        """
        if (channels is not None) and (groups is not None):
            raise ValueError("Arguments 'channels' and 'groups' are mutually exclusive.")
        if (channels is None) and (groups is None):
            raise ValueError("Either 'channels' or 'groups' must be specified.")

        def _cka_processor(out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach()
            if t.dim() > 2:
                if reduction == "mean":
                    t = t.flatten(2).mean(2)
                elif reduction == "flatten":
                    t = t.flatten(2)
            return t

        def _compute_cka_via_covariance(features, num_groups):
            N, C_total = features.shape
            C_sub = C_total // num_groups

            features_c = features - features.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(features_c.t(), features_c)

            cov_tensor = cov_matrix.view(num_groups, C_sub, num_groups, C_sub).permute(0, 2, 1, 3)
            numerator = torch.sum(cov_tensor ** 2, dim=(2, 3))
            diag_norms = torch.diagonal(numerator, offset=0, dim1=0, dim2=1)
            denominator = torch.sqrt(diag_norms.unsqueeze(1) * diag_norms.unsqueeze(0))

            return numerator / (denominator + 1e-8)

        def _calc_linear_cka(mgr: HookManager):
            mgr.register(layer_pattern, processor=_cka_processor)
            yield

            dict_layer_scores = {}
            dict_counts = {}
            dict_matrix = {}
            dict_info = {}

            global_sum_off_diag = 0.0
            global_count_off_diag = 0

            target_layers = [name for name in mgr.storage.keys() if re.fullmatch(layer_pattern, name)]

            for layer_name in target_layers:
                data_list = mgr.storage[layer_name]
                if not data_list:
                    continue

                feats = torch.cat(data_list, dim=0)
                if feats.dim() > 2:
                    feats = feats.flatten(1)

                C_total = feats.shape[1]

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

                cka_mat = _compute_cka_via_covariance(feats, num_groups)

                if with_layer or with_avg:
                    if num_groups > 1:
                        sum_all = torch.sum(cka_mat)
                        sum_diag = torch.trace(cka_mat)
                        current_sum_off = sum_all - sum_diag
                        current_count_off = num_groups * num_groups - num_groups

                        avg_val = current_sum_off / current_count_off
                        if with_layer:
                            dict_layer_scores[layer_name] = avg_val.item()

                        global_sum_off_diag += current_sum_off.item()
                        global_count_off_diag += current_count_off
                    else:
                        if with_layer:
                            dict_layer_scores[layer_name] = 0.0

                if with_counts:
                    dict_counts[layer_name] = C_total

                if with_matrix:
                    dict_matrix[layer_name] = cka_mat.cpu().numpy()

                if with_group_info:
                    dict_info[layer_name] = (num_groups, width)

            result_dict = {}

            if with_avg:
                div = global_count_off_diag
                result_dict['avg'] = global_sum_off_diag / div if div > 0 else 0.0

            if with_layer:
                result_dict['layer'] = dict_layer_scores

            if with_counts:
                result_dict['counts'] = dict_counts

            if with_matrix:
                result_dict['matrix'] = dict_matrix

            if with_group_info:
                result_dict['group_info'] = dict_info

            return key, result_dict

        return self.pipe(_calc_linear_cka)

    def calc_eval(self,
                  key: str = "eval",
                  layer_pattern: str = r".*fc.*",
                  num_classes: Optional[int] = None,
                  num_groups: Optional[int] = None,
                  with_ens: bool = True,
                  with_path: bool = True,
                  with_oracle: bool = True,
                  cumulative_modes: Optional[List[str]] = None,
                  calc_device: str = "cpu") -> 'Analyzer':
        """分類精度・損失等の評価指標を算出する。

        累積アンサンブル分析に対応。LossおよびAccuracyに基づくソート順を指定可能。

        Args:
            key (str): 結果格納キー。
            layer_pattern (str): 対象層の正規表現。
            num_classes (Optional[int]): 1パスあたりのクラス数。
            num_groups (Optional[int]): パス数。
            with_ens (bool): 全パスのアンサンブル指標を含めるか。
            with_path (bool): パスごとの指標を含めるか。
            with_oracle (bool): Oracle指標を含めるか。
            cumulative_modes (Optional[List[str]]): 累積分析モードのリスト。
                以下のキーを指定可能:
                - 'original': 元の順序。
                - 'loss_asc' (or 'asc'): Loss昇順 (Best -> Worst)。
                - 'loss_desc' (or 'desc'): Loss降順 (Worst -> Best)。
                - 'acc_desc': Accuracy降順 (Best -> Worst)。
                - 'acc_asc': Accuracy昇順 (Worst -> Best)。
            calc_device (str): 集計計算を行うデバイス ('cpu' or 'cuda')。

        Returns:
            Analyzer: パイプラインオブジェクト。
        """
        # --- Auto-Detect Parameters ---
        if num_classes is None or num_groups is None:
            g, f = self._infer_layer_info(layer_pattern)
            if f is None:
                raise ValueError(f"Parameters could not be auto-detected for {layer_pattern}.")
            if num_groups is None:
                num_groups = g
            if num_classes is None:
                num_classes = f // num_groups

        def _calc_eval(mgr: HookManager) -> Generator:
            criterion = nn.CrossEntropyLoss(reduction='none').to(calc_device)

            def _processor(out):
                t = out[0] if isinstance(out, tuple) else out
                return t.detach()

            mgr.register(layer_pattern, processor=_processor)

            targets = yield

            if targets is None:
                warnings.warn(f"[{key}] Targets not provided. Skipping calculation.")
                return None

            if not mgr.storage:
                return None

            # --- Data Preparation on Calculation Device ---
            logits_all = torch.cat(list(mgr.storage.values())[0], dim=0).to(calc_device)
            targets = targets.to(calc_device)
            
            N = logits_all.shape[0]
            logits_reshaped = logits_all.view(N, num_groups, num_classes)

            res = {}

            # --- Basic Metrics Calculation ---
            p_preds = logits_reshaped.argmax(dim=2)
            p_correct = (p_preds == targets.unsqueeze(1))
            
            l_flat = logits_reshaped.view(-1, num_classes)
            t_flat = targets.unsqueeze(1).expand(-1, num_groups).reshape(-1)
            loss_flat = criterion(l_flat, t_flat)
            p_losses = loss_flat.view(N, num_groups)

            # Metrics per path (used for sorting)
            path_acc_mean = p_correct.float().mean(dim=0)
            path_loss_mean = p_losses.mean(dim=0)

            if with_ens:
                ens_logits = logits_reshaped.mean(dim=1)
                ens_acc = (ens_logits.argmax(dim=1) == targets).float().mean().item()
                ens_loss = criterion(ens_logits, targets).mean().item()
                res['ens'] = {'acc': ens_acc, 'loss': ens_loss}

            if with_path:
                res['path'] = {
                    'acc': path_acc_mean.tolist(),
                    'loss': path_loss_mean.tolist()
                }

            if with_oracle:
                oracle_acc = p_correct.any(dim=1).float().mean().item()
                oracle_loss = p_losses.min(dim=1).values.mean().item()
                res['oracle'] = {'acc': oracle_acc, 'loss': oracle_loss}

            # --- Cumulative Ensemble Analysis ---
            if cumulative_modes:
                res['cumulative'] = {}
                
                indices_map = {}
                divisors = torch.arange(1, num_groups + 1, device=calc_device).view(1, -1, 1)

                for mode in cumulative_modes:
                    # 1. Determine Sort Order
                    if mode == 'original':
                        idxs = torch.arange(num_groups, device=calc_device)
                    
                    # Loss based (asc=Small is Best)
                    elif mode in ('loss_asc', 'asc'):
                        idxs = torch.argsort(path_loss_mean, descending=False)
                    elif mode in ('loss_desc', 'desc'):
                        idxs = torch.argsort(path_loss_mean, descending=True)
                    
                    # Accuracy based (desc=Large is Best)
                    elif mode == 'acc_desc':
                        idxs = torch.argsort(path_acc_mean, descending=True)
                    elif mode == 'acc_asc':
                        idxs = torch.argsort(path_acc_mean, descending=False)
                    else:
                        warnings.warn(f"Unknown cumulative mode: {mode}")
                        continue

                    # 2. Reorder Logits
                    sorted_logits = logits_reshaped.index_select(1, idxs)
                    
                    # 3. Cumulative Sum & Average
                    cumulative_logits = sorted_logits.cumsum(dim=1)
                    cumulative_means = cumulative_logits / divisors

                    # 4. Batch Evaluation (Vectorized)
                    c_flat_logits = cumulative_means.view(-1, num_classes)
                    c_loss_flat = criterion(c_flat_logits, t_flat)
                    
                    c_loss_steps = c_loss_flat.view(N, num_groups).mean(dim=0)
                    
                    c_preds = c_flat_logits.argmax(dim=1).view(N, num_groups)
                    c_acc_steps = (c_preds == targets.unsqueeze(1)).float().mean(dim=0)

                    res['cumulative'][mode] = {
                        'acc': c_acc_steps.tolist(),
                        'loss': c_loss_steps.tolist(),
                        'order': idxs.tolist()
                    }

            return key, res

        return self.pipe(_calc_eval)

    def flow(self, dl: DataLoader) -> Dict[str, Dict[str, Any]]:
        """パイプラインを実行する。

        Args:
            dl (DataLoader): (inputs, targets) を返すデータローダー。

        Returns:
            Dict[str, Dict[str, Any]]: 分析結果。
        """
        results = {}
        generators = []
        managers = []
        ground_truths = []

        with torch.no_grad(), ExitStack() as stack:
            # 1. Setup Phase
            for step in self._steps:
                mgr = HookManager(self.model)
                stack.enter_context(mgr)
                managers.append(mgr)

                task_func = step['func']
                gen = task_func(mgr, *step['args'], **step['kwargs'])
                next(gen)
                generators.append(gen)

            # 2. Inference Phase
            for batch in dl:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                _ = self.model(inputs.to(self.device))

                if targets is not None:
                    ground_truths.append(targets.to("cpu"))

        all_targets = torch.cat(ground_truths) if ground_truths else None

        # 3. Aggregate Phase
        for gen, mgr in zip(generators, managers):
            try:
                gen.send(all_targets)
            except StopIteration as e:
                if e.value is not None:
                    key, val = e.value
                    results[key] = val

            mgr.reset()
            mgr.remove_hooks()

        return results

