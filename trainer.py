import copy
import functools
import io
import sys
from time import time

import torch
import torch.nn as nn
from torch.func import functional_call, grad, grad_and_value, stack_module_state, vmap
from torchinfo import summary

import utils
from utils import TimeLog


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
                  **kwargs
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


class TrainerUtils:
    def __init__(self, device):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.created_time = time()

    def printmet(self, met_dict, e=None, epochs=None, itv: int | float =1, force_print=False):
        """
        メトリクスをフォーマットして標準出力に表示する関数。

        Args:
            met_dict (dict): 表示する内容を格納した辞書。
            e (int): 現在のエポック数（1から開始）。
            epochs (int): 総エポック数。
            itv (int | float): 
                このインターバルごとに1行でstdoutに出力する。  
                それ以外のエポックでは、前回の出力を上書きしてstdoutに出力する。  
                最終エポックでは必ず1行出力される。
            force_print (bool, optional): 
                `True` にすると、`met_dict` の各要素が float 以外でも表示する。

        Example:
            # e+1 エポック目の met_dict の内容を、100 エポックまで出力。
            # 5 エポックごとに出力を残す。
            trainer.printmet(met_dict, e + 1, 100, itv=5)

            # e+1 エポック目の met_dict の内容を、epochs エポックまで出力。
            # epochs のうち 4 回出力を残す。
            trainer.printmet(met_dict, e + 1, epochs, itv=epochs / 4)
        """
        if 'epoch' in (key.lower() for key in met_dict.keys()):
            e_fix = met_dict["epoch"]
        else:
            e_fix = e

        disp_str = ""
        if e_fix is not None:
            disp_str += f"Epoch:{e_fix:>4}"
        if e is not None and epochs is not None:
            disp_str += f"/{e_fix - (e-1) - 1 + epochs:>4}"
        for key, value in met_dict.items():
            if key.lower() != "epoch":
                if force_print:
                    disp_str += f"    {key}: {value}"
                else:
                    try:
                        disp_str += f"    {key}: {value:<8.6f}"
                    except (ValueError, TypeError):
                        pass

        if e is not None and epochs is not None:
            current_time = time()
            if e == 1:
                self.start_time = time()
            else:
                req_time = (current_time - self.start_time) / (e - 1) * epochs
                eta = self.start_time + req_time
                left = eta - current_time
                eta_fmt = utils.format_time(eta, style=0)
                left_fmt = utils.format_duration(left, style=0)

            if (e - 1) != 0:
                disp_str += f"    eta: {eta_fmt} (left: {left_fmt})"

            disp_str += f"    duration: {utils.format_duration(current_time - self.created_time, style=0)}"

        if utils.interval(step=e, itv=itv, last_step=epochs):
            print(disp_str)
        else:
            print(disp_str, end="\r")

    def time_info(self, style_time=1, style_dur=0, incl_fmt=True):
        current_time = time()
        duration = current_time - self.created_time

        if incl_fmt:
            current_time_fmt = utils.format_time(current_time, style=style_time)
            duration_fmt = utils.format_duration(duration, style=style_dur)
            return {"timestamp": current_time, "timestamp_fmt": current_time_fmt, "duration": duration, "duration_fmt": duration_fmt}
        else:
            return {"timestamp": current_time, "duration": duration}
    

@TimeLog()
class Trainer(TrainerUtils):
    def __init__(self, network, criterion, optimizer, scheduler=None, device=None):
        super().__init__(device)
        self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    @TimeLog("dur_train", mode="add")
    @TimeLog("dur_total", mode="add")
    def train_1epoch(self, dl):
        stats_l = []
        self.network.train()
        for inputs, labels in dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            stats = self.train_1batch(inputs, labels)
            stats_l.append(stats)

        if self.scheduler is not None:
            self.scheduler.step()

        return self.train_agg(stats_l)
    
    # @TimeLog("dur_val", mode="add")
    @TimeLog("dur_total", mode="add")
    def val_1epoch(self, dl):
        if dl is None:
            return None

        stats_l = []
        self.network.eval()
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                stats = self.val_1batch(inputs, labels)
                stats_l.append(stats)

        return self.val_agg(stats_l)
    
    def train_agg(self, stats_l, mode=None):
        total_loss = sum(stats["batch_loss"] for stats in stats_l)
        total_corr = sum(stats["batch_corr"] for stats in stats_l)
        total_ndata = sum(stats["batch_ndata"] for stats in stats_l)
        
        loss = total_loss / total_ndata
        acc = total_corr / total_ndata
        
        return loss, acc

    def val_agg(self, stats_l, mode=None):
        total_loss = sum(stats["batch_loss"] for stats in stats_l)
        total_corr = sum(stats["batch_corr"] for stats in stats_l)
        total_ndata = sum(stats["batch_ndata"] for stats in stats_l)
        
        loss = total_loss / total_ndata
        acc = total_corr / total_ndata
        
        return loss, acc
    
    @TimeLog("dur_train_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def train_1batch(self, inputs, labels):
        outputs, loss = self.forward_flow(inputs, labels)
        preds, corr = self.eval_flow(outputs, labels)
        self.update_grad(loss)

        stats = {"batch_loss": loss.item() * len(inputs), "batch_corr": corr, "batch_ndata": len(inputs)}
        return stats.copy()

    # @TimeLog("dur_val_core", mode="add")
    @TimeLog("dur_total_core", mode="add")
    def val_1batch(self, inputs, labels):
        outputs, loss = self.forward_flow(inputs, labels)
        preds, corr = self.eval_flow(outputs, labels)

        stats = {"batch_loss": loss.item() * len(inputs), "batch_corr": corr, "batch_ndata": len(inputs)}
        return stats.copy()

    @TimeLog("dur_pred", mode="add")
    def pred_1iter(self, dl, categorize=True, prob=False):
        all_outputs = []
        all_labels = []

        self.network.eval()
        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)

                outputs = self.network(inputs)
                
                if prob:
                    outputs = torch.softmax(outputs, dim=1)
                elif categorize:
                    outputs = torch.argmax(outputs, dim=1)

                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return all_outputs, all_labels

    def forward_flow(self, inputs, labels):
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def eval_flow(self, outputs, labels):
        preds = torch.argmax(outputs.detach(), dim=1)
        corr = torch.sum(preds == labels.data).item()

        return preds, corr

    def update_grad(self, loss):
        # 通常は zero_grad -> bachward -> step  効率を考慮するなら backward -> step -> zero_grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0] # recommended
        return self.optimizer.param_groups[0]["lr"]

    def fmt_criterion(self):
        return self.criterion.__class__.__name__

    def fmt_optimizer(self, verbose=False, use_break=False):
        if verbose:
            string = repr(self.optimizer)
            if not use_break:
                lines = string.split("\n")
                lines = [line.strip() for line in lines if line.strip()]
                string = " ".join(lines)
        else:
            string = self.optimizer.__class__.__name__
        return string

    def fmt_scheduler(self, verbose=False, use_break=False):
        if self.scheduler:
            if verbose:
                string = self.scheduler.__class__.__name__ + "(\n"
                for attr in dir(self.scheduler):
                    if not attr.startswith("_") and not callable(getattr(self.scheduler, attr)):  # exclude special attributes and methods
                        if attr.startswith("optimizer"):
                            value = f"{getattr(self.scheduler, attr).__class__.__name__}()"
                        else:
                            value = getattr(self.scheduler, attr)
                        string += f"{attr} = {value}\n"
                string += ")"
            else:
                string = self.scheduler.__class__.__name__

            if not use_break:
                lines = string.split("\n")
                lines = [line.strip() for line in lines if line.strip()]
                string = " ".join(lines)
            return string
        else:
            return None


@TimeLog(stats_name="time_stats_mt")
class MultiTrainer(TrainerUtils):
    def __init__(self, trainers, device=None):
        super().__init__(device)
        self.trainers = trainers

    def train_1epoch(self, dl, agg_f=lambda agg_l: map(list, zip(*agg_l))):
        [trainer.network.train() for trainer in self.trainers]
        
        stats_ll = [[] for _ in self.trainers]
        for inputs, labels in TimeLog.iter_load(self, "dur_dl_train", dl):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for i, trainer in enumerate(self.trainers):
                stats = trainer.train_1batch(inputs, labels)
                stats_ll[i].append(stats)

        [trainer.scheduler.step() for trainer in self.trainers if trainer.scheduler]

        agg_l = [trainer.train_agg(stats_l) for trainer, stats_l in zip(self.trainers, stats_ll)]
        if agg_f:
            agg_l = agg_f(agg_l)                

        return agg_l

    def val_1epoch(self, dl, agg_f=lambda agg_l: map(list, zip(*agg_l))):
        if dl is None:
            return None

        [trainer.network.eval() for trainer in self.trainers]

        with torch.no_grad():
            stats_ll = [[] for _ in self.trainers]
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                for i, trainer in enumerate(self.trainers):
                    stats = trainer.val_1batch(inputs, labels)
                    stats_ll[i].append(stats)
            agg_l = [trainer.val_agg(stats_l) for trainer, stats_l in zip(self.trainers, stats_ll)]
        if agg_f:
            agg_l = agg_f(agg_l)                

        return agg_l     

    @property
    def networks(self):
        return Networks([trainer.network for trainer in self.trainers])

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, trainer in enumerate(self.trainers):
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.trainers) else arg for arg in args]
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.trainers) else v for k, v in kwargs.items()}
                return_l.append(getattr(trainer, attr)(*new_args, **new_kwargs))
            return return_l
        return wrapper


class MergeEnsemble(Trainer):
    def __init__(self, networks, criterion=None, optimizer=None, scheduler=(None, None), device=None):
        super().__init__(None, criterion, optimizer, scheduler, device)
        self.trainers = [Trainer(network, criterion, device=device) for network in networks]
            
    def train_1epoch(self, dl, ens_f=lambda outputs_l: torch.stack(outputs_l, dim=0).mean(dim=0), incl_members=False):
        total_losses = [0.0] * len(self.trainers)
        total_corrs = [0] * len(self.trainers)

        ens_total_loss = 0.0
        ens_total_corr = 0

        [trainer.network.train() for trainer in self.trainers]

        for inputs, labels in dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs_l = []
            for i, trainer in enumerate(self.trainers):
                outputs, loss = trainer.forward_flow(inputs, labels)
                preds, corr = trainer.eval_flow(outputs, labels)

                total_losses[i] += loss.item() * len(inputs)
                total_corrs[i] += corr

                outputs_l.append(outputs)

            ens_outputs = ens_f(outputs_l)  # デフォルトでは平均をとる
            ens_loss = self.criterion(ens_outputs, labels)
            self.update_grad(ens_loss)
            ens_preds, ens_corr = self.eval_flow(ens_outputs, labels)

            ens_total_loss += ens_loss.item() * len(inputs)
            ens_total_corr += ens_corr

            if self.scheduler[1] == "iter":
                self.scheduler[0].step()

        if self.scheduler[1] == "epoch":
            self.scheduler[0].step()

        train_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        train_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        ens_train_loss = ens_total_loss / len(dl.dataset)
        ens_train_acc = ens_total_corr / len(dl.dataset)

        if incl_members:
            return (ens_train_loss, train_losses), (ens_train_acc, train_accs)
        else:
            return ens_train_loss, ens_train_acc

    def val_1epoch(self, dl, ens_f=lambda outputs_l: torch.stack(outputs_l, dim=0).mean(dim=0), incl_members=False):
        if dl is None:
            return None, None

        total_losses = [0.0] * len(self.trainers)
        total_corrs = [0] * len(self.trainers)

        ens_total_loss = 0.0
        ens_total_corr = 0

        [trainer.network.eval() for trainer in self.trainers]

        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs_l = []
                for i, trainer in enumerate(self.trainers):
                    outputs, loss = trainer.forward_flow(inputs, labels)
                    preds, corr = trainer.eval_flow(outputs, labels)

                    total_losses[i] += loss.item() * len(inputs)
                    total_corrs[i] += corr

                    outputs_l.append(outputs)

                ens_outputs = ens_f(outputs_l)  # デフォルトでは平均をとる
                ens_loss = self.criterion(ens_outputs, labels)
                ens_preds, ens_corr = self.eval_flow(ens_outputs, labels)

                ens_total_loss += ens_loss.item() * len(inputs)
                ens_total_corr += ens_corr

        val_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        val_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        ens_val_loss = ens_total_loss / len(dl.dataset)
        ens_val_acc = ens_total_corr / len(dl.dataset)

        if incl_members:
            return (ens_val_loss, val_losses), (ens_val_acc, val_accs)
        else:
            return ens_val_loss, ens_val_acc

    @property
    def network(self):
        return Networks([trainer.network for trainer in self.trainers], merge_stat=True)


class MergeEnsembleMeta(Trainer):
    def __init__(self, networks, criterion=None, optimizer=None, scheduler=(None, None), device=None):
        super().__init__(None, criterion, optimizer, scheduler, device)
        self.networks = networks
        self.trainers = [Trainer(network, criterion, device=device) for network in networks]
        self.criterion = nn.CrossEntropyLoss()
        # self.scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")
        
        # params, self.meta_model = self.fetch_meta_model()
        # params, buffers = stack_module_state(self.networks)

        self.base_model = copy.deepcopy(self.networks[0]).to("meta")
        self.params, self.buffers = stack_module_state(self.networks)

        self.optimizer = torch.optim.Adam(self.params.values(), lr=0.005)
        # self.optimizer = torch.optim.SGD(self.params.values(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        
            
    def train_1epoch(self, dl, ens_f=lambda outputs: outputs.mean(dim=0), incl_members=False):
        total_losses = [0.0] * len(self.trainers)
        total_corrs = [0] * len(self.trainers)

        ens_total_loss = 0.0
        ens_total_corr = 0

        # [trainer.network.train() for trainer in self.trainers]
        self.base_model.train()

        for inputs, labels in dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # def forward_fn(params, buffers, inputs):
            #     return vmap(lambda p, b, x: functional_call(self.base_model, (p, b), (x,)), in_dims=(0, 0, None))(params, buffers, inputs)  # [M, B, C]
                
            # def compute_loss(params, buffers, inputs, labels):
            #     outputs_exp = forward_fn(params, buffers, inputs)  # [M, B, C]
            #     ens_outputs = outputs_exp.mean(dim=0) 
            #     # ens_outputs = ens_f(outputs_exp) 
            #     ens_loss = nn.functional.cross_entropy(ens_outputs, labels)
            #     return ens_loss, (outputs_exp, ens_outputs)
            
            # grad_p, (ens_loss, (outputs_exp, ens_outputs)) = grad_and_value(compute_loss, argnums=0, has_aux=True)(self.params, self.buffers, inputs, labels)  # [M, ...]
            
                
            def compute_loss(params, buffers):
                outputs = functional_call(self.base_model, (params, buffers), (inputs,))
                loss = nn.functional.cross_entropy(outputs, labels)
                return loss
            
            grad_p = vmap(grad(compute_loss, has_aux=False), in_dims=(0, 0))(self.params, self.buffers)
            
            # self.optimizer.zero_grad()
            # for p, g in zip(self.params.values(), grad_p.values()):
            #     p.grad = g
            # self.optimizer.step()  # 更新
            
            # flat_outputs = torch.flatten(outputs_exp, start_dim=0, end_dim=1)   # [M*B, C]
            # flat_labels = labels.repeat(len(self.trainers))     # [M*B]
            # flat_loss = self.criterion_meta(flat_outputs, flat_labels)         # [M, B]
            # loss_exp = flat_loss.view(len(self.trainers), -1).mean(dim=1)       # [M]

            # preds_exp = torch.argmax(outputs_exp.detach(), dim=2)
            # corr_exp = torch.sum(preds_exp == labels_exp.data, dim=1).cpu().tolist()

            # for i, trainer in enumerate(self.trainers):
                # total_losses[i] += loss_exp[i].item() * len(inputs)
                # total_corrs[i] += corr_exp[i]

            # ens_outputs = ens_f(outputs_exp)  # デフォルトでは平均をとる
            # ens_loss = self.criterion(ens_outputs, labels)
            
            # self.update_grad(ens_loss)


            # ens_preds, ens_corr = self.eval_flow(ens_outputs, labels)

            # ens_total_loss += ens_loss.item() * len(inputs)
            # ens_total_corr += ens_corr


            if self.scheduler[1] == "iter":
                self.scheduler[0].step()

        if self.scheduler[1] == "epoch":
            self.scheduler[0].step()

        train_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        train_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        ens_train_loss = ens_total_loss / len(dl.dataset)
        ens_train_acc = ens_total_corr / len(dl.dataset)

        if incl_members:
            return (ens_train_loss, train_losses), (ens_train_acc, train_accs)
        else:
            return ens_train_loss, ens_train_acc

    def val_1epoch(self, dl, ens_f=lambda outputs: outputs.mean(dim=0), incl_members=False):
        if dl is None:
            return None, None

        total_losses = [0.0] * len(self.trainers)
        total_corrs = [0] * len(self.trainers)

        ens_total_loss = 0.0
        ens_total_corr = 0

        [trainer.network.eval() for trainer in self.trainers]

        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                def compute_loss(params, buffers):
                    outputs = functional_call(self.base_model, (params, buffers), (inputs,))
                    loss = nn.functional.cross_entropy(outputs, labels)
                    return loss
                
                loss = vmap(compute_loss, in_dims=(0, 0))(self.params, self.buffers)
            



                # outputs_exp = vmap(lambda params, buffers, inputs: functional_call(self.base_model, (params, buffers), inputs), in_dims=(0, 0, None))(self.params, self.buffers, inputs)  # [M, B, C]

                # ens_outputs = ens_f(outputs_exp)  # デフォルトでは平均をとる
                # ens_loss = self.criterion(ens_outputs, labels)
                # ens_preds, ens_corr = self.eval_flow(ens_outputs, labels)

                # ens_total_loss += ens_loss.item() * len(inputs)
                # ens_total_corr += ens_corr

        val_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        val_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        ens_val_loss = ens_total_loss / len(dl.dataset)
        ens_val_acc = ens_total_corr / len(dl.dataset)

        if incl_members:
            return (ens_val_loss, val_losses), (ens_val_acc, val_accs)
        else:
            return ens_val_loss, ens_val_acc

    # def fetch_meta_model(self):
        # base_model = copy.deepcopy(self.networks[0]).to("meta")
        # params, buffers = stack_module_state(self.networks)
        # fmodel = lambda params, buffers, inputs: functional_call(base_model, (params, buffers), inputs)
        # meta_model = lambda inputs: vmap(fmodel, in_dims=(0, 0, None))(params, buffers, inputs)
        
        # return params, meta_model
