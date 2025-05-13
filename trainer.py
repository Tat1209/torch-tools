import sys
import io
from time import time

import torch
from torchinfo import summary

import utils

class Network:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def get_sd(self):
        return self._wrapped.state_dict()

    def load_sd(self, sd_path):
        sd = torch.load(sd_path)
        self._wrapped.load_state_dict(sd)

    def count_params(self):
        return sum(p.numel() for p in self._wrapped.parameters())

    def count_params_trainable(self):
        return sum(p.numel() for p in self._wrapped.parameters() if p.requires_grad)

    def count_params_with_grad(self):
        return sum(p.grad.numel() for p in self._wrapped.parameters() if p.grad is not None)

    def grad_mean(self, abs=True):

        grads = []
        for param in self._wrapped.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        all_grads = torch.cat(grads)
        grad_mean = all_grads.mean().item()      # 平均勾配
        grad_abs_mean = all_grads.abs().mean().item()
        if abs:
            return grad_abs_mean
        else:
            return grad_mean

    def torchinfo(self,
                  dl=None,
                  input_size=(1, 3, 224, 224),
                  batch_dim=None,
                  verbose=1,
                  depth=4,
                  col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds", ],
                  row_settings=["var_names"],
                  ):
        if dl is not None:
            inputs, _ = next(iter(dl))
            input_size = inputs.shape

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        summary(model=self._wrapped, input_size=input_size, batch_dim=batch_dim, depth=depth, verbose=verbose, col_names=col_names, row_settings=row_settings)
        sys.stdout = old_stdout
        summary_str = buf.getvalue()

        return summary_str

    def repr_network(self):
        return repr(self._wrapped)

    def __getattr__(self, name):
        orig = getattr(self._wrapped, name)

        if callable(orig):
            # return a function that, when called, wraps Module results
            def hooked(*args, **kwargs):
                result = orig(*args, **kwargs)
                if isinstance(result, torch.nn.Module):
                    return Network(result)
                return result
            return hooked

        return orig

    def __call__(self, *args, **kwargs):
        return self._wrapped.__call__(*args, **kwargs)

class Networks(list):
    def __init__(self, networks=None, merge=False, merge_f=lambda networks: sum(networks)):
        # Networkインスタンスのリストを想定
        super().__init__(networks if networks is not None else [])
        self.merge = merge
        self.merge_f = merge_f

    def count_params(self):
        if self.merge:
            return self.merge_f(network.count_params() for network in self)
        else:
            return [network.count_params() for network in self]

    def count_params_trainable(self):
        if self.merge:
            return self.merge_f(network.count_params_trainable() for network in self)
        else:
            return [network.count_params_trainable() for network in self]

    def count_params_with_grad(self):
        if self.merge:
            return self.merge_f(network.count_params_with_grad() for network in self)
        else:
            return [network.count_params_with_grad() for network in self]

    def parameters(self):
        if self.merge:
            for network in self:
                for p in network.parameters():
                    yield p
        # return (p for network in self for p in network.parameters())
        else:
            for network in self:
                yield network.parameters()

    def to(self, device):
        moved = [network.to(device) for network in self]
        return Networks(moved)

    def __repr__(self):
        return f"Networks({list(self)})"
    
    def __getattr__(self, attr):
        # ここにないやつは各Networkに投げ，その結果のリストを返す
        def wrapper(*args, **kwargs):
            return_l = []
            for i, network in enumerate(self):
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self) else arg for arg in args]
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self) else v for k, v in kwargs.items()}
                return_l.append(getattr(network, attr)(*new_args, **new_kwargs))
            return return_l
        return wrapper


class TrainerUtils:
    def __init__(self, device):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.time_data = {}
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

    @classmethod
    def time_log(cls, name="time", mode="add"):
        assert mode in ("add", "set"), f"Invalid mode: {mode}. Expected 'add' or 'set'."

        def _deco(func):
            def _wrapper(*args, **kwargs):
                self = args[0]  # インスタンス（self）を取得
                if not hasattr(self, "time_data"):
                    raise AttributeError("Instance must have 'time_data' attribute.")
                if mode == "add":
                    start = time()
                    result = func(*args, **kwargs)
                    elapsed = time() - start
                    self.time_data[name] = self.time_data.get(name, 0) + elapsed
                elif mode == "set":
                    result = func(*args, **kwargs)
                    self.time_data[name] = time()
                return result
            return _wrapper
        return _deco

    def timeinfo(self, style=0):
        current_time = time()
        current_time_fmt = utils.format_time(current_time, style=1)
        duration = current_time - self.created_time
        duration_fmt = utils.format_duration(duration)

        return {"timestamp": current_time, "timestamp_fmt": current_time_fmt, "duration": duration, "duration_fmt": duration_fmt}


class Trainer(TrainerUtils):
    def __init__(self, network=None, criterion=None, optimizer=None, scheduler_t=(None, None), device=None):
        super().__init__(device)
        if network is not None:
            self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        if scheduler_t != (None, None):
            assert isinstance(scheduler_t, tuple), "scheduler_t must be a tuple"
            assert len(scheduler_t) == 2, "scheduler_t must have two elements"
            assert isinstance(scheduler_t[0], torch.optim.lr_scheduler.LRScheduler), "scheduler_t[0] must be a torch.optim.lr_scheduler type"
            assert isinstance(scheduler_t[1], str), "scheduler_t[1] must be a string"
            assert scheduler_t[1] in ["epoch", "iter"], "scheduler_t[1] must be either 'epoch' or 'iter'"
        self.scheduler_t = scheduler_t

    # @TrainerUtils.time_log("train_dur", mode="add")
    # @TrainerUtils.time_log("total_dur", mode="add")
    def train_1epoch(self, dl):
        total_loss = 0.0
        total_corr = 0

        self.network.train()
        for inputs, labels in dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, loss = self.model_flow(inputs, labels)
            preds, corr = self.eval_flow(outputs, labels)

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            self.update_grad(loss)

            if self.scheduler_t[1] == "iter":
                self.scheduler_t[0].step()
        if self.scheduler_t[1] == "epoch":
            self.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)

        return train_loss, train_acc

    # @TrainerUtils.time_log("total_dur", mode="add")
    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        self.network.eval()
        total_loss = 0.0
        total_corr = 0

        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, loss = self.model_flow(inputs, labels)
                preds, corr = self.eval_flow(outputs, labels)

                total_loss += loss.item() * len(inputs)
                total_corr += corr

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)

        return val_loss, val_acc

    def model_flow(self, inputs, labels):
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def eval_flow(self, outputs, labels):
        preds = torch.argmax(outputs.detach(), dim=1)
        corr = torch.sum(preds == labels.data).item()

        return preds, corr

    def update_grad(self, loss):
        # 通常は zero_grad -> bachward -> step
        # 効率を考慮するなら backward -> step -> zero_grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # @TrainerUtils.time_log("pred_dur", mode="add")
    def pred_1iter(self, dl, categorize=True):
        total_outputs = []
        total_labels = []

        self.network.eval()
        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.network(inputs).detach()

                if categorize:
                    outputs = torch.argmax(outputs, dim=1)

                total_outputs.append(outputs)
                total_labels.append(labels)

        outputs = torch.cat(total_outputs, dim=0)
        labels = torch.cat(total_labels, dim=0)

        return outputs, labels

    def timeinfo(self):
        time_info = super().timeinfo()
        # for k, v in self.time_data.items():
        #     time_info[k] = v
        #     k_fmt = k + "_fmt"
        #     if k.endswith("_dur"):
        #         time_info[k_fmt] = utils.format_duration(v, style=0)
        #     elif k.endswith("_time"):
        #         time_info[k_fmt] = utils.format_time(v, style=1)

        return time_info

    def get_lr(self):
        # return self.scheduler_t[0].get_lr()[0]
        return self.scheduler_t[0].get_last_lr()[0] # recommended
    # return self.optimizer.param_groups[0]["lr"]

    def repr_criterion(self):
        return repr(self.criterion)

    def repr_optimizer(self, use_break=False):
        string = repr(self.optimizer)
        if not use_break:
            string = string.replace("\n", " ")
        return string

    def repr_scheduler(self, use_break=False):
        if self.scheduler_t == (None, None):
            return None
        else:
            format_string = self.scheduler_t[0].__class__.__name__ + " (\n"
            for attr in dir(self.scheduler_t[0]):
                if not attr.startswith("_") and not callable(getattr(self.scheduler_t[0], attr)):  # exclude special attributes and methods
                    if attr.startswith("optimizer"):
                        value = f"{getattr(self.scheduler_t[0], attr).__class__.__name__}()"
                    else:
                        value = getattr(self.scheduler_t[0], attr)
                    format_string += f"{attr} = {value}\n"
            format_string += ")"

            if not use_break:
                format_string = format_string.replace("\n", " ")
            return format_string

    def repr_device(self):
        return repr(self.device)


class MultiTrainer(TrainerUtils):
    def __init__(self, trainers=None, device=None):
        super().__init__(device) # アンサンブルとか，params数を合算したいとかならmerge=Trueとする
        self.trainers = trainers

    def train_1epoch(self, dl):
        total_losses = [0.0] * len(self.trainers)
        total_corrs = [0] * len(self.trainers)

        [trainer.network.train() for trainer in self.trainers]

        for inputs, labels in dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            for i, trainer in enumerate(self.trainers):
                outputs, loss = trainer.model_flow(inputs, labels)
                preds, corr = trainer.eval_flow(outputs, labels)

                total_losses[i] += loss.item() * len(inputs)
                total_corrs[i] += corr

                trainer.update_grad(loss)

            for trainer in self.trainers:
                if trainer.scheduler_t[1] == "iter":
                    trainer.scheduler_t[0].step()

        for trainer in self.trainers:
            if trainer.scheduler_t[1] == "epoch":
                trainer.scheduler_t[0].step()

        train_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        train_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        return train_losses, train_accs

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        total_losses = [0.0] * len(self.trainers)
        total_corrs = [0] * len(self.trainers)

        [trainer.network.eval() for trainer in self.trainers]

        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                for i, trainer in enumerate(self.trainers):
                    outputs, loss = trainer.model_flow(inputs, labels)
                    preds, corr = trainer.eval_flow(outputs, labels)

                    total_losses[i] += loss.item() * len(inputs)
                    total_corrs[i] += corr

        val_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        val_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        return val_losses, val_accs

    @property
    def networks(self):
        return Networks([trainer.network for trainer in self.trainers], merge=False)

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
    def __init__(self, networks, criterion=None, optimizer=None, scheduler_t=(None, None), device=None):
        super().__init__(None, criterion, optimizer, scheduler_t, device)
        self.trainers = [Trainer(network, device=device) for network in networks]

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
                outputs, loss = trainer.model_flow(inputs, labels)
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

            if self.scheduler_t[1] == "iter":
                self.scheduler_t[0].step()

        if self.scheduler_t[1] == "epoch":
            self.scheduler_t[0].step()

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
                    outputs, loss = trainer.model_flow(inputs, labels)
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
        val_accs = [total_loss / len(dl.dataset) for total_loss in total_corrs]

        ens_val_loss = ens_total_loss / len(dl.dataset)
        ens_val_acc = ens_total_corr / len(dl.dataset)

        if incl_members:
            return (ens_val_loss, val_losses), (ens_val_acc, val_accs)
        else:
            return ens_val_loss, ens_val_acc

    @property
    def networks(self):
        return Networks([trainer.network for trainer in self.trainers], merge=True, merge_f=lambda networks: sum(networks))

    # def __getattr__(self, attr):
    #     def wrapper(*args, **kwargs):
    #         return_l = []
    #         for i, trainer in enumerate(self.trainers):
    #             new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.trainers) else arg for arg in args]
    #             new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
    #             return_l.append(getattr(trainer, attr)(*new_args, **new_kwargs))
    #         return return_l
    #     return wrapper




