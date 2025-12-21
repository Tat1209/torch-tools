import copy
import fnmatch
import io
import itertools
import math
import sys
import warnings
from functools import partial
from typing import List, Literal, Callable

import torch
from torch import nn, Tensor
from torchinfo import summary

from utils import LazyPipeline


class Network(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    @property
    def device(self):
        return next(self.base_model.parameters()).device
        
    def forward(self, x):
        return self.base_model(x)

    def count_params(self, trainable=False, with_grad=False):
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif with_grad:
            return sum(p.numel() for p in self.parameters() if p.grad is not None)
        return sum(p.numel() for p in self.parameters())

    def grad_mean(self, abs_val=True):
        grads = [p.grad.view(-1) for p in self.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        all_grads = torch.cat(grads)
        return all_grads.abs().mean().item() if abs_val else all_grads.mean().item()

    def torchinfo(self,
                  dl=None,
                  input_size=(1, 3, 224, 224),
                  batch_dim=None,
                  verbose=1,
                  depth=8,
                  col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds", ],
                  row_settings=["var_names"],
                  output_path=None,
                  ):
        if dl is not None:
            inputs, _ = next(iter(dl))
            input_size = inputs.shape

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        summary(model=self.base_model, input_size=input_size, batch_dim=batch_dim, depth=depth, verbose=verbose, col_names=col_names, row_settings=row_settings)
        sys.stdout = old_stdout
        summary_str = buf.getvalue()
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(summary_str)

        return summary_str

    def tl_setup(self, num_classes, linear_layer="fc"):
        for param in self.base_model.parameters():
            param.requires_grad = False

        old_linear = self.base_model.get_submodule(linear_layer)
        in_feats = old_linear.in_features

        new_linear = nn.Linear(in_feats, num_classes)
        self.base_model.set_submodule(linear_layer, new_linear)
        return self
        
    def unfreeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
        return self

    def param_stat(self, stat_f=lambda p: p.numel(), incl_if=lambda p: True):
        """
        パラメータに対して統計を計算し、 統計値を返す。

        Args:
            stat_f (Callable): パラメータに適用する関数。例: lambda p: p.norm().item()
            incl_if (Callable): 統計の対象とする条件関数。例: lambda p: p.requires_grad

        Returns:
            float: 統計値
        """
        with torch.no_grad():
            def flatten_params():
                for p in self.base_model.parameters():
                    if incl_if(p):
                        yield p.flatten()

        all_concat = torch.cat(tuple(flatten_params()))
        return stat_f(all_concat)
    
    def param_stat_layer(self, stat_f=lambda p: p.numel(), incl_if=lambda p: True):
        """
        各層のパラメータに対して統計を計算し、{層名: 統計値}の辞書を返す。

        Args:
            stat_f (Callable): パラメータに適用する関数。例: lambda p: p.norm().item()
            incl_if (Callable): 統計の対象とする条件関数。例: lambda p: p.requires_grad

        Returns:
            dict: {層名(str): 統計値(float)}
        """
        stats = {}
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if incl_if(param):
                    stats[name] = stat_f(param)
        return stats
    
    def grad_stat(self, stat_f=lambda g: g.numel(), incl_if=lambda p: p.grad is not None) -> float:
        """
        全勾配に対して統計を計算し、単一の統計値を返す。

        Args:
            stat_f (Callable): 勾配に適用する関数。例: lambda g: g.norm().item()
            incl_if (Callable): 対象パラメータを選別する関数。例: lambda p: p.requires_grad and p.grad is not None

        Returns:
            float: 統計値 (勾配が存在しない場合は0)
        """
        grads = []
        for p in self.base_model.parameters():
            if incl_if(p):
                g = p.grad
                grads.append(g.flatten())
        if not grads:
            return 0.0
        all_grads = torch.cat(grads)
        return stat_f(all_grads)

    def grad_stat_layer(self, stat_f=lambda g: g.numel(), incl_if=lambda p: p.grad is not None) -> dict:
        """
        各勾配層ごとに統計を計算し、層名->統計値の辞書を返す。

        Args:
            stat_f (Callable): 勾配に適用する関数。例: lambda g: g.norm().item()
            incl_if (Callable): 対象パラメータを選別する関数。例: lambda p: p.requires_grad and p.grad is not None

        Returns:
            Dict[str, float]: 層名->統計値
        """
        stats = {}
        for name, param in self.base_model.named_parameters():
            if incl_if(param):
                g = param.grad
                stats[name] = stat_f(g)
        return stats

    def repr_network(self):
        return repr(self.base_model)

    def load_state_dict_flexible(self, state_dict, strict=True):
        """
        プレフィックスや階層構造の違いを吸収して state_dict をロードするメソッド。
        
        以下の順序でキーの適合を試みます:
        1. 不要なプレフィックス(_orig_mod, module)の削除
        2. そのままマッチするか確認
        3. 'base_model.' を付与してマッチするか確認 (直のモデルをラップしたモデルにロードする場合)
        4. 'base_model.' を削除してマッチするか確認 (ラップしたモデルを直のモデルにロードする場合)
        """
        own_keys = set(self.state_dict().keys())
        
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # torch.compileの '_orig_mod.', DataParallelの 'module.' を除去
            k_clean = k.replace("_orig_mod.", "").replace("module.", "")
            
            if k_clean in own_keys:
                # ケースA: クリーンなキーがそのまま一致する
                new_state_dict[k_clean] = v
                
            elif f"base_model.{k_clean}" in own_keys:
                # ケースB: 手元が 'base_model.xxx' で、ロード元が 'xxx' (base_model単体を保存していた場合)
                new_state_dict[f"base_model.{k_clean}"] = v
                
            elif k_clean.startswith("base_model.") and k_clean.replace("base_model.", "") in own_keys:
                # ケースC: 手元が 'xxx' で、ロード元が 'base_model.xxx' (逆のケース)
                new_state_dict[k_clean.replace("base_model.", "")] = v
                
            else:
                new_state_dict[k_clean] = v

        return super().load_state_dict(new_state_dict, strict=strict)

class Networks(list):
    def __init__(self, networks=None, agg_f=None):
        super().__init__(networks or [])
        self.agg_f = agg_f

    def parameters(self):
        for network in self:
            yield network.parameters()

    def __getattr__(self, attr):
        def wrapper(*args, agg_f=None, **kwargs):
            results = []
            for m in self:
                fn = getattr(m, attr)
                results.append(fn(*args, **kwargs))
            
            if agg_f:
                return agg_f(results)
            elif self.agg_f:
                return self.agg_f(results)
            return results
        return wrapper

class Refiner(LazyPipeline):
    def __init__(self, model: nn.Module, _steps: List = None):
        super().__init__(model, _steps)
    
    @property
    def model(self) -> nn.Module:
        return self._seed_obj
    
    @property
    def _linear_types(self):
        return (nn.Linear,)

    def build(self, _inplace=False, dbg=False) -> nn.Module:
        # inplaceは未実装．forwardの変換が必要で，新たにnn.Moduleを返す場合，in-placeはできないっぽい．
        target_model = self.model if _inplace else self._check_and_copy()
        model = self._execute(target_model)
        
        if dbg:
            print(self.history())
        
        if _inplace:
            return
        else:
            return model

    def _check_and_copy(self) -> nn.Module:
        """GPU上のモデルをコピーする際のリスク管理"""
        has_gpu_tensor = any(p.is_cuda for p in self.model.parameters()) or \
                         any(b.is_cuda for b in self.model.buffers())

        if has_gpu_tensor:
            warnings.warn(
                "Warning: Deepcopying a model located on GPU. "
                "This doubles VRAM usage temporarily.", 
                UserWarning
            )
        
        return copy.deepcopy(self.model)
                
    def apply_policies(self, module, policies, full_path=False, _prefix=""):
        """
        モジュールを再帰的に走査し、policies (関数またはそのリスト) を適用します。
        
        policy_fn(name, child) の戻り値による挙動:
        - None (暗黙含む): ヒットせず。子モジュールへの再帰探索を継続します。
        - module (自分自身): 自身に置き換え (=置換せず)、探索を終了します。
        - new_module (別のもの): new_module に置換し、探索を終了します。
        
        Args:
            full_path (bool): 判定に用いる名前の種別を指定します。
                - False: 自身の `name` (例: "conv1") で判定します。
                - True: ルートからの `full_path` (例: "layer1.conv1") で判定します。
            _prefix (str, optional): 再帰用の内部変数です。指定不要です。

        Example:
            def policy(name, module):
                if isinstance(module, nn.Linear):
                    return nn.Identity()  # 置換して探索終了
                if name == "frozen_block":
                    return module         # 自身に置き換え (=置換せず)、探索終了
                # else return             # (or Implicit None) -> 探索を継続

            self.apply_policies(model, [policy_linear])                  # name には "conv1" 等が渡る
            self.apply_policies(model, [policy_linear], full_path=True)  # name には "backbone.layer1.conv1" 等が渡る
        """
        if not isinstance(policies, (list, tuple)):
            policies = [policies]

        for name, child in module.named_children():
            if full_path:
                # _prefix は常に文字列なので直接判定可能
                identifier = f"{_prefix}.{name}" if _prefix else name
                next_prefix = identifier
            else:
                identifier = name
                next_prefix = ""

            for policy_fn in policies:
                new_child = policy_fn(identifier, child)

                if new_child is not None:
                    if new_child is not child:
                        setattr(module, name, new_child)
                    break
            else:
                self.apply_policies(child, policies, full_path=full_path, _prefix=next_prefix)
        
    def _match_layername(self, name: str, targets: str | List[str]) -> bool:
        if isinstance(targets, str):
            targets = [targets]

        return any(fnmatch.fnmatch(name, t) or fnmatch.fnmatch(name, f"*model.{t}") for t in targets) # model.付きも念のため判定

    def remove_downsample(self, target_layer: List[str] | str) -> "Refiner":
        def _remove_downsample(model, target_layer=target_layer):
            def policy_fn(name, module: nn.Module) -> nn.Module | None:
                if self._match_layername(name, target_layer):
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        return module.__class__(
                            in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=1,
                            padding=module.padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias is not None,
                            padding_mode=module.padding_mode,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
                    elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                        return nn.Identity()

            self.apply_policies(model, policy_fn, full_path=True)

        return self.pipe(_remove_downsample, target_layer=target_layer)

    def kernel_size_adjust(self, target_layer: List[str] | str, to_size: int, padding: int | None = None) -> "Refiner":
        if padding is None:
            padding = (to_size - 1) // 2

        def _kernel_size_adjust(model, target_layer=target_layer, to_size=to_size, padding=padding):
            def policy_fn(name, module: nn.Module) -> nn.Module | None:
                if self._match_layername(name, target_layer):
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        return module.__class__(
                            in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=to_size,
                            stride=module.stride,
                            padding=padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias is not None,
                            padding_mode=module.padding_mode,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
            self.apply_policies(model, policy_fn, full_path=True)

        return self.pipe(_kernel_size_adjust, target_layer=target_layer)

    def cifar_style(self, arch: Literal["auto", "resnet", "regnet", "mobilenet", "efficientnet", "convnext"] = "auto") -> "Refiner":
        # 必ず最初に適用すること．full_pathがの完全一致で判定するため，ラップされてからだとmatchしない．（一応，_match_layernameにmodel.も同時に判定する．）
        # CIFAR画像を入力した際，4x4の特徴マップがGAPに入るように調整する
        if arch == "auto":
            arch = self.get_arch()
            
        if arch in ("resnet"):
            return self.remove_downsample(["conv1", "maxpool"]).kernel_size_adjust("conv1", to_size=3, padding=1)

        if arch in ("regnet"):
            return self.remove_downsample(["stem.0", "trunk_output.block1.block1-0.*"])

        if arch in ("mobilenet"):
            return self.remove_downsample(["features.0.0", "features.2.conv.0.0"])

        if arch in ("efficientnet"):
            return self.remove_downsample(["features.0.0", "features.2.0.block.1.0"])

        if arch in ("convnext"):
            return self.remove_downsample(["features.0.0"]).kernel_size_adjust("features.0.0", to_size=3, padding=1)
        return self

    def _apply_init(self, m: nn.Module, init_func: Callable[[Tensor], None], **kwargs):
        init_func(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def init_weights(self, arch: Literal["auto", "resnet", "regnet", "mobilenet", "efficientnet", "convnext"] = "auto", **kwargs) -> "Refiner":
        def _init_weights(model, arch=arch, **kwargs):
            if arch == "auto":
                arch = self.get_arch()

            if arch == "resnet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        fn = partial(nn.init.kaiming_normal_, mode="fan_out", nonlinearity="relu")
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            elif arch == "regnet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        def regnet_init(w):
                            fan_out = w.size(0) * w.size(2) * w.size(3)
                            nn.init.normal_(w, mean=0.0, std=math.sqrt(2.0 / fan_out))
                        self._apply_init(m, regnet_init, **kwargs)
                    elif isinstance(m, self._linear_types):
                        fn = partial(nn.init.normal_, mean=0.0, std=0.01)
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            elif arch == "mobilenet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        fn = partial(nn.init.kaiming_normal_, mode="fan_out")
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, self._linear_types):
                        fn = partial(nn.init.normal_, mean=0.0, std=0.01)
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            elif arch == "efficientnet":
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        fn = partial(nn.init.kaiming_normal_, mode="fan_out")
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, self._linear_types):
                        init_range = 1.0 / math.sqrt(m.out_features)
                        fn = partial(nn.init.uniform_, a=-init_range, b=init_range)
                        self._apply_init(m, fn, **kwargs)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

            elif arch == "convnext":
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, *self._linear_types)):
                        fn = partial(nn.init.trunc_normal_, std=0.02)
                        self._apply_init(m, fn, **kwargs)

            return model

        return self.pipe(_init_weights, arch=arch, **kwargs)

    def get_arch(self) -> str:
        model_name = self.model.__class__.__name__.lower()
        if "resnet" in model_name:
            return "resnet"
        elif "mobilenet" in model_name:
            return "mobilenet"
        elif "convnext" in model_name:
            return "convnext"
        elif "efficientnet" in model_name:
            return "efficientnet"
        elif "regnet" in model_name:
            return "regnet"
        else:
            warnings.warn(
                    f"Unknown model architecture detected: '{model_name}'. "
                    "Returned 'unknown'. Check if logic update is needed.",
                    category=UserWarning,
                    stacklevel=2
                )
            return "unknown"


def comp_param_stat(models: List, layer_width=30):
    """
    nn.Moduleのリストを受け取り、各モデルの層ごとのパラメータ統計（平均と標準偏差）を表示する。
    モデルが2つの場合は、平均の差(Mean1 - Mean2)と分散の商(Var1 / Var2)も表示する。

    Args:
        models (List[nn.Module]): 統計を比較したいPyTorchモデルのリスト。
        layer_width (int): 層名の表示幅。
    """
    if not models:
        print("No models provided.")
        return

    all_stats_data = []
    network_names = []

    # データを収集
    for model in models:
        # Networkクラスでラップする（ユーザー環境依存）
        net = Network(model) 
        network_names.append(model.__class__.__name__)
        
        # 層ごとの統計値を取得
        means = net.param_stat_layer(stat_f=lambda p: p.mean().item())
        vars = net.param_stat_layer(stat_f=lambda p: p.var().item())
        
        # (層名, 平均, 分散) のタプルのリストを作成
        model_stats = []
        for name in means.keys():
            model_stats.append((name, means[name], vars[name]))
        all_stats_data.append(model_stats)

    # --- テーブル形式での出力設定 ---

    # 基本カラム幅定義
    L_WIDTH = layer_width  # Layer name
    M_WIDTH = 10  # Mean
    S_WIDTH = 10  # Var
    # 個別モデルのカラム幅
    COL_WIDTH = L_WIDTH + M_WIDTH + S_WIDTH + 2 
    
    # 比較用カラム幅定義（Diff, Ratio）
    D_WIDTH = 10
    R_WIDTH = 10
    COMP_COL_WIDTH = D_WIDTH + R_WIDTH + 2 # マージン含む
    
    SEPARATOR = " " * 4 

    # ヘッダー作成
    header_names_list = []
    header_cols_list = []
    separator_line_list = []
    
    # 各モデルのヘッダー
    for name in network_names:
        header_names_list.append(f"Network: {name:<{COL_WIDTH - len('Network: ')}}")
        header_cols_list.append(f"{'Layer':<{L_WIDTH}} {'Mean':>{M_WIDTH}} {'Var':>{S_WIDTH}}")
        separator_line_list.append("-" * COL_WIDTH)

    # ★追加: モデルが2つの場合、比較用ヘッダーを追加
    is_comparison = (len(models) == 2)
    if is_comparison:
        header_names_list.append(f"Comparison:{' ':<{COMP_COL_WIDTH - len('Comparison:')}}")
        header_cols_list.append(f"{'Diff(M)':>{D_WIDTH}} {'Ratio(V)':>{R_WIDTH}}")
        separator_line_list.append("-" * COMP_COL_WIDTH)

    # ヘッダー出力
    print(SEPARATOR.join(header_names_list))
    print(SEPARATOR.join(separator_line_list))
    print(SEPARATOR.join(header_cols_list))
    print(SEPARATOR.join(separator_line_list))

    # データ行の作成
    filler = (None, None, None)
    for row_data in itertools.zip_longest(*all_stats_data, fillvalue=filler):
        row_str_list = []
        
        # 各モデルの統計値を文字列化
        for name, mean, var in row_data:
            if name is not None:
                part = f"{name:<{L_WIDTH}} {mean:>{M_WIDTH}.6f} {var:>{S_WIDTH}.6f}"
            else:
                part = " " * COL_WIDTH
            row_str_list.append(part)
        
        # ★追加: 比較計算ロジック
        if is_comparison:
            # row_data = [(name1, m1, v1), (name2, m2, v2)]
            d1, d2 = row_data
            
            # 両方のモデルに層が存在する場合のみ計算
            if d1[0] is not None and d2[0] is not None:
                mean1, var1 = d1[1], d1[2]
                mean2, var2 = d2[1], d2[2]
                
                # 平均の差: M1 - M2
                diff_mean = mean1 - mean2
                
                # 分散の商: V1 / V2 (ゼロ除算対策)
                if var2 != 0:
                    ratio_var = var1 / var2
                else:
                    ratio_var = float('inf') # または 0.0 など状況に応じて
                
                comp_part = f"{diff_mean:>{D_WIDTH}.6f} {ratio_var:>{R_WIDTH}.6f}"
            else:
                # 片方の層が欠けている場合は空白
                comp_part = " " * COMP_COL_WIDTH
            
            row_str_list.append(comp_part)

        print(SEPARATOR.join(row_str_list))
    
    print(SEPARATOR.join(separator_line_list))