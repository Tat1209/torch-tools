from pathlib import Path
import pickle
from PIL import Image

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

from dataset_handler import DatasetHandler

class FixRandomDataset(Dataset):
    def __init__(self, size):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index):
        data = self.data.detach().clone()
        target = index
        # target = 1
        return data, target

    def __len__(self):
        return 10000

class PklToDataset(Dataset):
    def __init__(self, pkl_path, transform=None, target_transform=None):
        with open(pkl_path, "rb") as f:
            (self.datas, self.targets) = pickle.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        target = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

class TinyImageNet(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        ti_path = root / Path("tiny-imagenet-200")
        self.paths = []
        self.targets = []
        self.dirname_label = {}
        
        if not ti_path.exists():
            raise FileNotFoundError
        
        # wnids.txtを参照し、クラス名と値の対応付けを行う。その後、self.dirname_labelに対応を格納
        with open(ti_path / Path("wnids.txt"), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                key = line.strip()
                value = i
                self.dirname_label[key] = value
        
        if train:
            ti_train_path = ti_path / Path("train")
            for class_dir in ti_train_path.iterdir():
                if class_dir.is_dir():
                    images_dir = class_dir / Path("images")

                    for image_path in images_dir.iterdir():
                        if image_path.is_file():
                            self.paths.append(image_path)
                            self.targets.append(self.dirname_label[class_dir.name])
                            
        else:
            ti_val_path = ti_path / Path("val")
            images_dir = ti_val_path / Path("images")

            with open(ti_val_path / Path("val_annotations.txt"), 'r') as f:
                lines = f.readlines()

            for line in lines:
                elems = line.split()
                image_name = elems[0]
                class_name = elems[1]

                image_path = images_dir / Path(image_name)

                if image_path.is_file():
                    self.paths.append(image_path)
                    self.targets.append(self.dirname_label[class_name])
            
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = Image.open(self.paths[idx]).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

class Datasets:
    def __init__(self, root=None):
        self.root = Path(root)

    def _base_ds(self, ds_str, download=False):
        match (ds_str):
            case "mnist_train":
                return torchvision.datasets.MNIST(root=self.root, train=True, download=download)
            case "mnist_val":
                return torchvision.datasets.MNIST(root=self.root, train=False, download=download)
            case "cifar100_train":
                return torchvision.datasets.CIFAR100(root=self.root, train=True, download=download)
            case "cifar100_val":
                return torchvision.datasets.CIFAR100(root=self.root, train=False, download=download)
            case "cifar10_train":
                return torchvision.datasets.CIFAR10(root=self.root, train=True, download=download)
            case "cifar10_val":
                return torchvision.datasets.CIFAR10(root=self.root, train=False, download=download)
            case "stl10_train":
                return torchvision.datasets.STL10(root=self.root, split="train", download=download)
            case "stl10_val":
                return torchvision.datasets.STL10(root=self.root, split="test", download=download)
            case "caltech101_trainval":
                return torchvision.datasets.Caltech101(root=self.root, target_type="category", download=download)
            case "tiny-imagenet_train":
                return TinyImageNet(root=self.root, train=True)
            case "tiny-imagenet_val":
                return TinyImageNet(root=self.root, train=False)
            case "cars_train":
                return torchvision.datasets.StanfordCars(root=self.root, split="train", download=download)
            case "cars_val":
                return torchvision.datasets.StanfordCars(root=self.root, split="test", download=download)
            case "pets_train":
                return torchvision.datasets.OxfordIIITPet(root=self.root, split="trainval", target_types="category", download=download)
            case "pets_val":
                return torchvision.datasets.OxfordIIITPet(root=self.root, split="test", target_types="category", download=download)
            case "flowers_train":
                return torchvision.datasets.Flowers102(root=self.root, split="train", download=download)
            case "flowers_val":
                return torchvision.datasets.Flowers102(root=self.root, split="val", download=download)
            case "flowers_test":
                return torchvision.datasets.Flowers102(root=self.root, split="test", download=download)
            case "imagenet":
                return torchvision.datasets.ImageNet(root=self.root, split="val")
            case "mnist_ddim":
                return torchvision.datasets.ImageFolder(root=str(Path(self.root) / Path("mnist_ddim")))
            case "ai-step_l":
                return PklToDataset(self.root / Path("fukui_train_32_60_ver2.pkl"))
            case "ai-step_ul":
                return PklToDataset(self.root / Path("kanazawa_test_32_60_ver2.pkl"))
            case "fix_rand":
                return FixRandomDataset((3, 32, 32))
            case _:
                raise ValueError(f"Invalid dataset name: {ds_str}")

    def __call__(self, ds_str, transform_l=[], target_transform_l=[], u_seed=None):
        try:
            base_ds = self._base_ds(ds_str)
        except RuntimeError:
            try:
                base_ds = self._base_ds(ds_str, download=True)
            except (RuntimeError, FileNotFoundError) as e:
                raise e(f"The 'download' argument is not supported for this dataset.")
        base_ds.ds_str = ds_str
        base_ds.ds_name = base_ds.__class__.__name__
        base_ds.u_seed = u_seed

        indices = np.arange(len(base_ds), dtype=np.int32)
        classes = None
        transform = torchvision.transforms.Compose(transform_l)
        target_transform = torchvision.transforms.Compose(target_transform_l)
        
        dsh = DatasetHandler(base_ds, indices, classes, transform, target_transform)
        base_ds.base_dsh = dsh
        
        return dsh

