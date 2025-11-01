import io
import sys

import torch
import torch.nn as nn
from torchinfo import summary

class Network(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return self.base_model(x)

    def get_sd(self, **kwargs):
        return self.state_dict()

    def load_sd(self, sd_path, **kwargs):
        sd = torch.load(sd_path)
        self.load_state_dict(sd)

    def count_params(self, trainable=False, with_grad=False, **kwargs):
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif with_grad:
            return sum(p.numel() for p in self.parameters() if p.grad is not None)
        return sum(p.numel() for p in self.parameters())

    def grad_mean(self, abs_val=True, **kwargs):
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
                  depth=4,
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

    def param_stat(self, stat_f=lambda p: p.numel(), incl_if=lambda p: True, **kwargs):
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
    
    def param_stat_layer(self, stat_f=lambda p: p.numel(), incl_if=lambda p: True, **kwargs):
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

    def repr_network(self, **kwargs):
        return repr(self.base_model)


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

