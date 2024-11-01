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

    def printlog(self, log_dict, e, epochs, itv=10, anyval=False, timezone=9):
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
        for key, value in log_dict.items():
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
                "batch",
            ], "scheduler_t[1] must be either 'epoch' or 'batch'"
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

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler_t is not None and self.scheduler_t[1] == "batch":
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

    def ret_ckpt(self):
        ckpt = dict()
        ckpt["network_sd"] = self.network.state_dict()
        ckpt["optimizer_sd"] = self.optimizer.state_dict()
        try:
            ckpt["scheduler_sd"] = self.scheduler_t[0].state_dict()
        except Exception:
            pass
        return ckpt

    def load_ckpt(self, path):
        ckpt = torch.load(path)
        self.network.load_state_dict(ckpt["network_sd"])
        self.optimizer.load_state_dict(ckpt["optimizer_sd"])
        try:
            self.scheduler_t[0].load_state_dict(ckpt["scheduler_sd"])
        except Exception:
            pass

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


# Ensは、一つの同じローダーでアンサンブルするためのもの
class Ens_1Loader(TrainerUtils):
    def __init__(self, models=None, device=None):
        self.models = models
        super().__init__(device)

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        for model in self.models:
            model.network.eval()
        total_loss = 0.0
        total_corr = 0.0

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = [model.network(inputs) for model in self.models]
                outputs = torch.mean(torch.stack(outputs), dim=0)

                loss = self.loss_func(outputs, labels)

                _, pred = torch.max(outputs.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                total_loss += loss.item() * len(inputs)
                total_corr += corr

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)

        return val_loss, val_acc

    def get_sd(self):
        return [model.get_sd() for model in self.models]

    def load_sd(self, sd_list):
        (self.models[i].load_sd(sd) for i, sd in enumerate(sd_list))

    def count_params(self):
        return sum(model.count_params() for model in self.models)

    def count_trainable_params(self):
        return sum(model.count_trainable_params() for model in self.models)


class PureEns_1Loader(Ens_1Loader):
    def __init__(self, models=None, device=None):
        super().__init__(models=models, device=device)

    def train_1epoch(self, dl):
        # この実装では、すべてのModelについて同じデータローダ―を使用している。
        # 正常な動作は、
        for model in self.models:
            model.network.train()
        total_loss = 0.0
        total_corr = 0.0

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = [model.network(inputs) for model in self.models]
            outputs = torch.mean(torch.stack(outputs), dim=0)
            losses = [self.models[m].loss_func(outputs[m], labels) for m in range(len(self.models))]

            _, pred = torch.max(outputs.detach(), dim=1)
            corr = torch.sum(pred == labels.data).item()

            total_loss += sum([loss.item() for loss in losses]) * len(inputs)
            total_corr += corr

            for model in self.models:
                model.optimizer.zero_grad()
            for loss in losses:
                loss.backward()
            for model in self.models:
                model.optimizer.step()

            for model in self.models:
                if model.scheduler_t is not None and model.scheduler_t[1] == "batch":
                    model.scheduler_t[0].step()
        for model in self.models:
            if model.scheduler_t is not None and model.scheduler_t[1] == "epoch":
                model.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)

        return train_loss, train_acc


class MergeEns_1Loader(Ens_1Loader):
    def __init__(self, models=None, device=None):
        super().__init__(models=models, device=device)

    def train_1epoch(self, dl):
        for model in self.models:
            model.network.train()
        total_loss = 0.0
        total_corr = 0.0

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = [model.network(inputs) for model in self.models]
            outputs = torch.mean(torch.stack(outputs), dim=0)
            loss = model.loss_func(outputs, labels)

            _, pred = torch.max(outputs.detach(), dim=1)
            corr = torch.sum(pred == labels.data).item()

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            for model in self.models:
                model.optimizer.zero_grad()
            loss.backward()
            for model in self.models:
                model.optimizer.step()

            for model in self.models:
                if model.scheduler_t is not None and model.scheduler_t[1] == "batch":
                    model.scheduler_t[0].step()
        for model in self.models:
            if model.scheduler_t is not None and model.scheduler_t[1] == "epoch":
                model.scheduler_t[0].step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)

        return train_loss, train_acc


class MultiTrain(TrainerUtils):
    def __init__(self, models=None, device=None):
        self.models = models
        super().__init__(device)

    def train_1epoch(self, dl):
        for model in self.models:
            model.network.train()
        total_losses = [0.0 for _ in self.models]
        total_corrs = [0.0 for _ in self.models]

        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = [model.network(inputs) for model in self.models]

            losses = [self.models[m].loss_func(outputs[m], labels) for m in range(len(self.models))]
            _, preds = zip(*[(torch.max(outputs.detach(), dim=1)) for outputs in outputs])

            losses_add = [loss.item() * len(inputs) for loss in losses]
            corrs_add = []
            for pred in preds:
                corr = torch.sum(pred == labels.data).item()
                corrs_add.append(corr)

            total_losses = [a + b for a, b in zip(total_losses, losses_add)]
            total_corrs = [a + b for a, b in zip(total_corrs, corrs_add)]

            for model in self.models:
                model.optimizer.zero_grad()

            for loss in losses:
                loss.backward()

            for model in self.models:
                model.optimizer.step()

            for model in self.models:
                if model.scheduler_t is not None and model.scheduler_t[1] == "batch":
                    model.scheduler_t[0].step()
        for model in self.models:
            if model.scheduler_t is not None and model.scheduler_t[1] == "epoch":
                model.scheduler_t[0].step()

        total_losses = [loss / (len(dl.dataset) * len(self.models)) for loss in total_losses]
        total_accs = [corr / len(dl.dataset) for corr in total_corrs]

        return total_losses, total_accs
        # return list(zip(total_losses, total_accs))

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        for model in self.models:
            model.network.eval()
        total_losses = [0.0 for _ in self.models]
        total_corrs = [0.0 for _ in self.models]

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = [model.network(inputs) for model in self.models]

                losses = [self.models[m].loss_func(outputs[m], labels) for m in range(len(self.models))]
                _, preds = zip(*[(torch.max(outputs.detach(), dim=1)) for outputs in outputs])

                losses_add = [loss.item() * len(inputs) for loss in losses]
                corrs_add = []
                for pred in preds:
                    corr = torch.sum(pred == labels.data).item()
                    corrs_add.append(corr)

                total_losses = [a + b for a, b in zip(total_losses, losses_add)]
                total_corrs = [a + b for a, b in zip(total_corrs, corrs_add)]

        val_losses = [loss / (len(dl.dataset) * len(self.models)) for loss in total_losses]
        val_accs = [acc / len(dl.dataset) for acc in total_corrs]

        return val_losses, val_accs
        # return list(zip(val_losses, val_accs))

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, model in enumerate(self.models):
                # new_args = []
                # for arg in args:
                #     if isinstance(arg, list) and len(arg) == len(self.models):
                #         new_arg = arg[i]
                #     else:
                #         new_arg = arg
                #     new_args.append(new_arg)
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.models) else arg for arg in args]

                # new_kwargs = dict()
                # for k, v in kwargs.items():
                #     if isinstance(v, list) and len(v) == len(self.models):
                #         new_kwargs[k] = v[i]
                #     else:
                #         new_kwargs[k] = v
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(model, attr)(*new_args, **new_kwargs))
            return return_l

        return wrapper
    
    def __getitem__(self, idx):
        return self.models[idx]


class MyMultiTrain(TrainerUtils):
    def __init__(self, models=None, device=None):
        self.models = models
        super().__init__(device)

    def train_1epoch(self, dl):
        for model in self.models:
            model.network.train()
        total_losses = [0.0 for _ in self.models]
        total_corrs = [0.0 for _ in self.models]

        for inputs, labels in dl:
            # labels = torch.Tensor([0 for _ in range(len(labels))]).long() # 消すこと
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            for i, model in enumerate(self.models):
                outputs = model.network(inputs)
                loss = model.loss_func(outputs, labels)

                _, pred = torch.max(outputs.detach(), dim=1)
                corr = torch.sum(pred == labels.data).item()

                total_losses[i] += loss.item() * len(inputs)
                total_corrs[i] += corr

                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

            for model in self.models:
                if model.scheduler_t is not None and model.scheduler_t[1] == "batch":
                    model.scheduler_t[0].step()
        for model in self.models:
            if model.scheduler_t is not None and model.scheduler_t[1] == "epoch":
                model.scheduler_t[0].step()

        total_losses = [loss / (len(dl.dataset) * len(self.models)) for loss in total_losses]
        total_accs = [corr / len(dl.dataset) for corr in total_corrs]

        return total_losses, total_accs
        # return list(zip(total_losses, total_accs))

    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        for model in self.models:
            model.network.eval()
        total_losses = [0.0 for _ in self.models]
        total_corrs = [0.0 for _ in self.models]

        with torch.no_grad():
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                for i, model in enumerate(self.models):
                    outputs = model.network(inputs)
                    loss = model.loss_func(outputs, labels)

                    _, pred = torch.max(outputs.detach(), dim=1)
                    corr = torch.sum(pred == labels.data).item()

                    total_losses[i] += loss.item() * len(inputs)
                    total_corrs[i] += corr

        val_losses = [loss / (len(dl.dataset) * len(self.models)) for loss in total_losses]
        val_accs = [acc / len(dl.dataset) for acc in total_corrs]

        return val_losses, val_accs
        # return list(zip(val_losses, val_accs))
        

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, model in enumerate(self.models):
                # new_args = []
                # for arg in args:
                #     if isinstance(arg, list) and len(arg) == len(self.models):
                #         new_arg = arg[i]
                #     else:
                #         new_arg = arg
                #     new_args.append(new_arg)
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.models) else arg for arg in args]

                # new_kwargs = dict()
                # for k, v in kwargs.items():
                #     if isinstance(v, list) and len(v) == len(self.models):
                #         new_kwargs[k] = v[i]
                #     else:
                #         new_kwargs[k] = v
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(model, attr)(*new_args, **new_kwargs))
            return return_l

        return wrapper

    def __getitem__(self, idx):
        return self.models[idx]