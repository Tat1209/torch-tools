from enum import Enum
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
# from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2 as transforms

from dataset_handler import DatasetHandler



class FixRandomDataset(Dataset):
    def __init__(self, size):
        super().__init__()

        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index):
        data = self.data.detach().clone()
        target = index
        return data, target

    def __len__(self):
        return 10000

class PklToDataset(Dataset):
    def __init__(self, pkl_path, transform=None, target_transform=None):
        # この実装どうなん？一気にデータ取得するからあまりよくない気がする．改善できればしたい．
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

class TestDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.root = Path(path).parent
        self.img_paths = sorted([p for p in Path(path).iterdir()])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path = self.img_paths[index]
        data = Image.open(path).convert('RGB')

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            path = self.target_transform(path)
        return data, str(path.name)

    def __len__(self):
        return len(self.img_paths)
    
    
def fetch_dataset(root, dataset_name):
    try:
        base_ds = _datasets(root, dataset_name, download=False)
    except RuntimeError:
        try:
            base_ds = _datasets(root, dataset_name, download=True)
        except (RuntimeError, FileNotFoundError) as e:
            raise e("The 'download' argument is not supported for this dataset.")
    return base_ds
        
def fetch_handler(root, dataset_id, base_ds=None):
    if not base_ds:
        base_ds = fetch_dataset(root, dataset_id)
    return DatasetHandler.create(dataset_id, root, base_ds) 
        

def _datasets(root, dataset_name, download=False):
    match (dataset_name):
        case "mnist_train":
            return torchvision.datasets.MNIST(root=root, train=True, download=download)
        case "mnist_val":
            return torchvision.datasets.MNIST(root=root, train=False, download=download)
        case "cifar100_train":
            return torchvision.datasets.CIFAR100(root=root, train=True, download=download)
        case "cifar100_val":
            return torchvision.datasets.CIFAR100(root=root, train=False, download=download)
        case "cifar10_train":
            return torchvision.datasets.CIFAR10(root=root, train=True, download=download)
        case "cifar10_val":
            return torchvision.datasets.CIFAR10(root=root, train=False, download=download)
        case "stl10_train":
            return torchvision.datasets.STL10(root=root, split="train", download=download)
        case "stl10_val":
            return torchvision.datasets.STL10(root=root, split="test", download=download)
        case "caltech101_trainval":
            return torchvision.datasets.Caltech101(root=root, target_type="category", download=download)
        case "tiny-imagenet_train":
            return TinyImageNet(root=root, train=True)
        case "tiny-imagenet_val":
            return TinyImageNet(root=root, train=False)
        case "cars_train":
            return torchvision.datasets.StanfordCars(root=root, split="train", download=download)
        case "cars_val":
            return torchvision.datasets.StanfordCars(root=root, split="test", download=download)
        case "pets_train":
            return torchvision.datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", download=download)
        case "pets_val":
            return torchvision.datasets.OxfordIIITPet(root=root, split="test", target_types="category", download=download)
        case "flowers_train":
            return torchvision.datasets.Flowers102(root=root, split="train", download=download)
        case "flowers_val":
            return torchvision.datasets.Flowers102(root=root, split="val", download=download)
        case "flowers_test":
            return torchvision.datasets.Flowers102(root=root, split="test", download=download)
        case "imagenet":
            return torchvision.datasets.ImageNet(root=root, split="val")
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        
