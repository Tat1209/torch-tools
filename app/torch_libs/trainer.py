import sys
import os
import io
import datetime
from time import time

import torch
import numpy as np
import polars as pl
from torchinfo import summary


class TrainerUtils:
    def __init__(self, device):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.start_time = None

    def interval(self, itv=None, step=None, last_step=None):
        return itv is None or (step - 1) % itv >= itv - 1 or step == last_step

    def printmet(self, met_dict, e, epochs, itv: int | float =1, anyval=False, timezone=9):
        """
            met_dict    : 表示内容
            e           : 現在のepoch数(最初は1)
            epochs      : 終了エポック数
            itv         : このインターバルごとに1行stdout それ以外は前回の出力を上書きしつつstdout 最終epochは必ず1行stdout
            anyval      : met_dictの各要素がfloat以外でも表示
            timezone    : etaの表示時間にx時間加算
            
            Ex.)
            # e+1 エポック目の met_dict の内容を 100epochまで出力 5epochごとに出力を残す
            trainer.printmet(met_dict, e + 1, 100, itv=5)

            # e+1 エポック目の met_dict の内容を epochs epochまで出力 epochsのうち4回出力を残す
            trainer.printmet(met_dict, e + 1, epochs, itv=epochs / 4)
        """
        if e == 1:
            self.start_time = time()
        else:
            stop_time = time()
            req_time = (stop_time - self.start_time) / (e - 1) * epochs
            left = self.start_time + req_time - stop_time
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left) + datetime.timedelta(hours=timezone)).strftime("%Y-%m-%d %H:%M")
            t_hour, t_min = divmod(left // 60, 60)
            left = f"{int(t_hour):02d}:{int(t_min):02d}"

        disp_str = ""
        for key, value in met_dict.items():
            try:
                if key == "epoch":
                    disp_str += f"Epoch: {value:>4}/{value - (e-1) - 1 + epochs:>4}"
                else:
                    if anyval:
                        disp_str += f"    {key}: {value}"
                    else:
                        disp_str += f"    {key}: {value:<9.7f}"
            except Exception:
                pass
        if (e - 1) != 0:
            disp_str += f"    eta: {eta} (left: {left})"

        # if (e - 1) % itv >= itv - 1 or e == epochs:
        if self.interval(itv=itv, step=e, last_step=epochs):
            print(disp_str)
        else:
            print(disp_str, end="\r")


class Trainer(TrainerUtils):
    def __init__(
        self,
        network=None,
        loss_func=None,
        optimizer=None,
        scheduler_t=None,
        device=None,
    ):
        super().__init__(device)
        self.network = network.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        if scheduler_t is not None:
            assert isinstance(scheduler_t, tuple), "scheduler_t must be a tuple"
            assert len(scheduler_t) == 2, "scheduler_t must have two elements"
            assert isinstance(scheduler_t[0], torch.optim.lr_scheduler.LRScheduler), "scheduler_t[0] must be a torch.optim.lr_scheduler type"
            assert isinstance(scheduler_t[1], str), "scheduler_t[1] must be a string"
            assert scheduler_t[1] in [
                "epoch",
                "iter",
            ], "scheduler_t[1] must be either 'epoch' or 'iter'"
        self.scheduler_t = scheduler_t

    def train_1epoch(self, dl):
        self.network.train()
        total_loss = 0.0
        total_corr = 0.0

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.network(inputs)
            loss = self.loss_func(outputs, labels)

            _, pred = torch.max(outputs.detach(), dim=1)
            corr = torch.sum(pred == labels.data).item()

            # 通常は zero_grad -> bachward -> step
            # 効率を考慮するなら backward -> step -> zero_grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            if self.scheduler_t is not None and self.scheduler_t[1] == "iter":
                self.scheduler_t[0].step()
        if self.scheduler_t is not None and self.scheduler_t[1] == "epoch":
            self.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)

        return train_loss, train_acc

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        self.network.eval()
        total_loss = 0.0
        total_corr = 0.0

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.network(inputs)
                loss = self.loss_func(outputs, labels)

                _, pred = torch.max(outputs.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                total_loss += loss.item() * len(inputs)
                total_corr += corr

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)

        return val_loss, val_acc

    def pred_1iter(self, dl, categorize=False):
        self.network.eval()
        total_output = None
        total_label = None

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device) # Noneのときどうなるかは分からん
                outputs = self.network(inputs)
                outputs = outputs.detach()

                if categorize:
                    _, pred = torch.max(outputs, dim=1)
                    outputs = pred

                if total_output is None:
                    total_output = outputs
                else:
                    total_output = torch.cat((total_output, outputs), dim=0)

                if total_label is None:
                    total_label = labels
                else:
                    total_label= torch.cat((total_label, labels), dim=0)

        return total_output, total_label

    def get_sd(self):
        return self.network.state_dict()

    def load_sd(self, sd_path):
        sd = torch.load(sd_path)
        self.network.load_state_dict(sd)

    def get_lr(self):
        # return self.scheduler_t[0].get_lr()[0]
        return self.optimizer.param_groups[0]["lr"]

    def count_params(self):
        return sum(p.numel() for p in self.network.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def arc_check(
        self,
        out_file=False,
        fname="arccheck.txt",
        dl=None,
        input_size=(200, 3, 256, 256),
        verbose=1,
        col_names=[
            "input_size",
            "output_size",
            "kernel_size",
            "num_params",
            "mult_adds",
        ],
        row_settings=["var_names"],
    ):
        if dl is not None:
            inputs, _ = next(iter(dl))
            input_size = inputs.shape
        try:
            tmp_out = io.StringIO()
            sys.stdout = tmp_out
            summary(
                model=self.network,
                input_size=input_size,
                verbose=verbose,
                col_names=col_names,
                row_settings=row_settings,
            )
        finally:
            sys.stdout = sys.__stdout__
        summary_str = tmp_out.getvalue()

        if out_file:
            with open(fname, "w") as f:
                f.write(summary_str)

        return summary_str

    def repr_network(self):
        return repr(self.network)

    def repr_loss_func(self):
        return repr(self.loss_func)

    def repr_optimizer(self, use_break=False):
        string = repr(self.optimizer)
        if not use_break:
            string = string.replace("\n", " ")
        return string

    def repr_scheduler(self, use_break=False):
        if self.scheduler_t is None:
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
        super().__init__(device)
        self.trainers = trainers

    def train_1epoch(self, dl):
        total_losses = [0.0 for _ in self.trainers]
        total_corrs = [0.0 for _ in self.trainers]

        for trainer in self.trainers:
            trainer.network.train()

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            for i, trainer in enumerate(self.trainers):
                outputs = trainer.network(inputs)
                loss = trainer.loss_func(outputs, labels)

                _, pred = torch.max(outputs.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()

                total_losses[i] += loss.item() * len(inputs)
                total_corrs[i] += corr

            for trainer in self.trainers:
                if trainer.scheduler_t is not None and trainer.scheduler_t[1] == "iter":
                    trainer.scheduler_t[0].step()
        
        for trainer in self.trainers:
            if trainer.scheduler_t is not None and trainer.scheduler_t[1] == "epoch":
                trainer.scheduler_t[0].step()

        train_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        train_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]
            
        return train_losses, train_accs

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        total_losses = [0.0 for _ in self.trainers]
        total_corrs = [0.0 for _ in self.trainers]

        for trainer in self.trainers:
            trainer.network.eval()

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                for i, trainer in enumerate(self.trainers):
                    outputs = trainer.network(inputs)
                    loss = trainer.loss_func(outputs, labels)

                    _, pred = torch.max(outputs.detach(), dim=1)
                    corr = torch.sum(pred == labels.data).item()

                    total_losses[i] += loss.item() * len(inputs)
                    total_corrs[i] += corr

        val_losses = [total_loss / len(dl.dataset) for total_loss in total_losses]
        val_accs = [total_corr / len(dl.dataset) for total_corr in total_corrs]

        return val_losses, val_accs

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, trainer in enumerate(self.trainers):
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.trainers) else arg for arg in args]
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(trainer, attr)(*new_args, **new_kwargs))
            return return_l
        return wrapper
    
    def __getitem__(self, idx):
        return self.trainers[idx]
