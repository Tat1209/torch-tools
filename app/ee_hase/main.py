import sys
from pathlib import Path
import math

import torch
from torchvision import transforms

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets
from run_manager import RunManager, RunsManager, RunViewer
from trainer import Trainer, MyMultiTrain
from trans import Trans

from mytrainer import MyTrainer

# from models.resnet_ee import resnet18 as net
from models.gitresnet_ee import resnet18 as net

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

exp_name = "exp_tmp"
# exp_name = "exp_dbg"

base_ndata = 10000
max_lr = 0.005
batch_size = 128

base_epochs = 0
# base_epochs = 100
train_ds_str_l = ["cifar10_train"]
val_ds_str_l = ["cifar10_val"]
ndata_l = [500]

base_fils = 16

for train_ds_str, val_ds_str in zip(train_ds_str_l, val_ds_str_l):
    # train_trans = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 360))]
    # val_trans = [transforms.Resize((32, 32)), transforms.ToTensor()]

    train_trans = Trans.cf_git
    val_trans = Trans.cf_gen

    base_train_ds = ds(train_ds_str, train_trans).balance_label(seed=0)
    val_ds = ds(val_ds_str, val_trans)

    # base_train_ds = ds(train_ds_str, train_trans).limit_class(max_num=10).balance_label(seed=0)
    # val_ds = ds(val_ds_str, val_trans).limit_class(labels=base_train_ds.fetch_classes(listed=True))
    
    for ndata in ndata_l:
        train_ds = base_train_ds.in_ndata(ndata)
        if ndata > len(train_ds):
            break
        # train_ds = train_ds.in_ratio(0.1)

        for fils_l in [[1, 2, 4, 8, 16]]:
        # for fils_l in [[1, 2, 4, 8, 16, 32]]:
        # for fils_l in [[2 ** i for i in range(int(math.log2(base_fils)) + 1)]]:
            runs = [RunManager(exc_path=__file__, exp_name=exp_name) for _ in fils_l]
            runs_mgr = RunsManager(runs)


            runs_mgr.log_param("model_arc", f"{net.__module__} {net.__name__}")

            runs_mgr.log_param("train_dataset", train_ds.ds_str)
            runs_mgr.log_param("val_dataset", val_ds.ds_str)
            runs_mgr.log_param("num_classes", num_classes := train_ds.fetch_classes())

            runs_mgr.log_param("train_trans", repr(train_trans))
            runs_mgr.log_param("val_trans", repr(val_trans))

            runs_mgr.log_param("train_num", len(train_ds))
            runs_mgr.log_param("val_num", len(val_ds))

            runs_mgr.log_param("epochs", epochs := int(base_epochs / ndata * base_ndata + 0.00000001))
            runs_mgr.log_param("max_lr", max_lr)
            runs_mgr.log_param("batch_size", batch_size)

            train_dl = train_ds.loader(batch_size, shuffle=True)
            val_dl = val_ds.loader(batch_size, shuffle=True)

            runs_mgr.log_param("iters/epoch", iters_per_epoch := len(train_dl))
            runs_mgr.log_param("base_fils", base_fils)
            runs_mgr.log_param("fils", fils_l)
            runs_mgr.log_param("ensembles", ensembles_l := [min(int((base_fils / fils) ** 2), 2048) for fils in fils_l])


            trainers = []
            for fils, ensembles in zip(fils_l, ensembles_l):
                network = net(num_classes=num_classes, nb_fils=fils, ee_groups=ensembles)
                loss_func = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)
                scheduler_t = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1), "epoch")

                trainer = Trainer(network, loss_func, optimizer, scheduler_t, device)
                trainers.append(trainer)
            mtrainer = MyMultiTrain(trainers, device)

            runs_mgr.log_param("params", mtrainer.count_params())
            hp_dict = {
                "loss_func": mtrainer.repr_loss_func(),
                "optimizer": mtrainer.repr_optimizer(),
                "scheduler": mtrainer.repr_scheduler(),
            }

            runs_mgr.log_params(hp_dict)
            runs_mgr.log_text(mtrainer.repr_network(), "model_layers.txt")
            runs_mgr.log_text(mtrainer.arc_check(dl=train_dl), "model_structure.txt")

            print(f"{base_fils=}, {fils_l}, {len(train_ds)=}")

            for e in range(epochs):
                runs_mgr.log_metric("lr", mtrainer.get_lr(), step=e + 1)

                train_loss, train_acc = mtrainer.train_1epoch(train_dl)
                met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc}

                if mtrainer.interval(itv=epochs/100, step=e+1, last_step=epochs):
                    val_loss, val_acc = mtrainer.val_1epoch(val_dl)
                    met_dict.update({"val_loss": val_loss, "val_acc": val_acc})
                # met_dict = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}


                runs_mgr.log_metrics(met_dict, step=e + 1)
                mtrainer.printlog(met_dict, e + 1, epochs, itv=epochs / 4)
                runs_mgr.ref_stats(itv=5, step=e + 1, last_step=epochs)

            runs_mgr.log_torch_save(mtrainer.get_sd(), "state_dict.pt")

            for run_mgr, trainer in zip(runs_mgr, mtrainer):
                # encoder = torch.nn.Sequential(*list(trainer.network.children())[:-1])
                etrainer = MyTrainer(network=trainer.network)

                feat = etrainer.fetch_feat(val_dl, flatten=True)

                for k, v in feat.items():
                    run_mgr.log_torch_save(feat[k], f"train_{k}.pt")
                    print(feat[k].shape)

            rv = RunViewer(exc_path=__file__, exp_name=exp_name)
            rv.fetch_results(refresh=True)
