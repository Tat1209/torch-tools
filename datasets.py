import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torchvision
from dataset_handler import DatasetHandler
from filelock import FileLock
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class PklToDataset(Dataset):
    def __init__(self, pkl_path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        with open(pkl_path, "rb") as f:
            (self.datas, self.targets) = pickle.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        data = self.datas[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        
        return data, target


class TinyImageNet(VisionDataset):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    dataset_folder = "tiny-imagenet-200"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, train: bool = True, download: bool = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        self.train = train
        self.base_path = self.root / self.dataset_folder

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.paths = []
        self.targets = []
        self.dirname_label = {}

        self._load_meta()
        self._load_data()

    def _load_meta(self) -> None:
        wnids_path = self.base_path / "wnids.txt"
        with open(wnids_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.dirname_label[line.strip()] = i

    def _load_data(self) -> None:
        if self.train:
            train_path = self.base_path / "train"
            for class_dir in train_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                if class_name not in self.dirname_label:
                    continue

                label = self.dirname_label[class_name]
                images_dir = class_dir / "images"
                
                for img_path in images_dir.glob("*.JPEG"):
                    self.paths.append(img_path)
                    self.targets.append(label)
        else:
            val_path = self.base_path / "val"
            images_dir = val_path / "images"
            anno_path = val_path / "val_annotations.txt"
            
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    img_name, class_name = parts[0], parts[1]
                    
                    if class_name in self.dirname_label:
                        img_path = images_dir / img_name
                        self.paths.append(img_path)
                        self.targets.append(self.dirname_label[class_name])

    def _check_integrity(self) -> bool:
        return (self.base_path / "wnids.txt").exists()

    def _download(self) -> None:
        if self._check_integrity():
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        img_path = self.paths[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target


class StanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, root: str, split: str = "train", download: bool = False, **kwargs):
        self.root = root
        if download:
            self._custom_download()

        try:
            super().__init__(root, split=split, download=False, **kwargs)
        except RuntimeError as e:
            raise RuntimeError("Dataset not found. Please use download=True.") from e

    def _custom_download(self) -> None:
        resources = [
            ("https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/cars_train.tgz", "cars_train.tgz"),
            ("https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/cars_test.tgz", "cars_test.tgz"),
            ("https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/car_devkit.tgz", "car_devkit.tgz")
        ]
        
        for url, filename in resources:
            check_folder = "devkit" if "devkit" in filename else filename.replace(".tgz", "")
            if os.path.exists(os.path.join(self.root, check_folder)):
                continue

            print(f"Downloading {filename}...")
            download_and_extract_archive(url, download_root=self.root, filename=filename)
            

class CUB200(VisionDataset):
    """
    Caltech-UCSD Birds-200-2011 Dataset
    """
    base_folder = 'CUB_200_2011'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        self.train = train
        self.base_path = self.root / self.base_folder

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it.")

        self.paths = []
        self.targets = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        images_txt = self.base_path / 'images.txt'
        train_test_split_txt = self.base_path / 'train_test_split.txt'
        class_labels_txt = self.base_path / 'image_class_labels.txt'

        # Build dictionaries
        img_id_to_path = {}
        with open(images_txt, 'r') as f:
            for line in f:
                img_id, path = line.strip().split()
                img_id_to_path[img_id] = path

        img_id_to_label = {}
        with open(class_labels_txt, 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                # Labels are 1-indexed in the text file, convert to 0-indexed
                img_id_to_label[img_id] = int(label) - 1

        # Filter by split
        target_split = '1' if self.train else '0'
        with open(train_test_split_txt, 'r') as f:
            for line in f:
                img_id, split = line.strip().split()
                if split == target_split:
                    self.paths.append(self.base_path / 'images' / img_id_to_path[img_id])
                    self.targets.append(img_id_to_label[img_id])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        img_path = self.paths[idx]
        target = self.targets[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        return (self.base_path / 'images.txt').exists()

    def _download(self) -> None:
        if self._check_integrity():
            return
        
        # Note: Caltech server sometimes blocks automated downloads. 
        # If this fails, download manually from https://data.caltech.edu/records/65de6-vp158
        download_and_extract_archive(
            self.url, 
            self.root, 
            filename=self.filename, 
            md5=self.tgz_md5
        )


class CIFAR10C(VisionDataset):
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar.gz"
    filename = "CIFAR-10-C.tar.gz"
    base_folder = "CIFAR-10-C"
    md5 = "561e08f6545a29482931168d9dd78225"

    def __init__(self, root: str, corruption: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        self.corruption = corruption
        self.base_path = self.root / self.base_folder

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.data = np.load(self.base_path / f"{corruption}.npy")
        self.targets = np.load(self.base_path / "labels.npy")

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index % 10000]  # Labels are repeated for each severity (5 levels)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return (self.base_path / "labels.npy").exists()

    def _download(self) -> None:
        if self._check_integrity():
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)


class CIFAR100C(VisionDataset):
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar.gz"
    filename = "CIFAR-100-C.tar.gz"
    base_folder = "CIFAR-100-C"
    md5 = "11f0ed0f1191cb99b231a9809fa73995"

    def __init__(self, root: str, corruption: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        self.corruption = corruption
        self.base_path = self.root / self.base_folder

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.data = np.load(self.base_path / f"{corruption}.npy")
        self.targets = np.load(self.base_path / "labels.npy")

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index % 10000]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return (self.base_path / "labels.npy").exists()

    def _download(self) -> None:
        if self._check_integrity():
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)


class TestDataset(Dataset):
    def __init__(self, path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.root = Path(path).parent
        self.img_paths = sorted(list(Path(path).iterdir()))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> tuple[Any, str]:
        path = self.img_paths[index]
        data = Image.open(path).convert('RGB')

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            path = self.target_transform(path) # type: ignore
            
        return data, str(path.name)

    def __len__(self) -> int:
        return len(self.img_paths)


def fetch_dataset(root: str, dataset_name: str) -> Dataset:
    try:
        return _datasets(root, dataset_name, download=False)
    except (RuntimeError, FileNotFoundError):
        pass

    os.makedirs(root, exist_ok=True)
    lock_file = os.path.join(root, f"{dataset_name.replace('/', '_')}.lock")

    with FileLock(lock_file):
        try:
            return _datasets(root, dataset_name, download=False)
        except (RuntimeError, FileNotFoundError):
            try:
                return _datasets(root, dataset_name, download=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load or download dataset '{dataset_name}'.") from e


def fetch_handler(root: str, dataset_id: str, base_ds: Optional[Dataset] = None) -> Any:
    if not base_ds:
        base_ds = fetch_dataset(root, dataset_id)
    return DatasetHandler.create(dataset_id, root, base_ds)


def _datasets(root: str, dataset_name: str, download: bool = False) -> Dataset:
    match dataset_name:
        case "mnist_train":
            return torchvision.datasets.MNIST(root=root, train=True, download=download)
        case "mnist_test":
            return torchvision.datasets.MNIST(root=root, train=False, download=download)

        case "fashion-mnist_train":
            return torchvision.datasets.FashionMNIST(root=root, train=True, download=download)
        case "fashion-mnist_test":
            return torchvision.datasets.FashionMNIST(root=root, train=False, download=download)

        case "cifar10_train":
            return torchvision.datasets.CIFAR10(root=root, train=True, download=download)
        case "cifar10_test":
            return torchvision.datasets.CIFAR10(root=root, train=False, download=download)

        case "cifar100_train":
            return torchvision.datasets.CIFAR100(root=root, train=True, download=download)
        case "cifar100_test":
            return torchvision.datasets.CIFAR100(root=root, train=False, download=download)

        case "svhn_train":
            return torchvision.datasets.SVHN(root=root, split="train", download=download)
        case "svhn_test":
            return torchvision.datasets.SVHN(root=root, split="test", download=download)

        case "stl10_train":
            return torchvision.datasets.STL10(root=root, split="train", download=download)
        case "stl10_test":
            return torchvision.datasets.STL10(root=root, split="test", download=download)
        case "stl10_unlabeled":
            return torchvision.datasets.STL10(root=root, split="unlabeled", download=download)

        case "stanford-cars_train":
            return torchvision.datasets.StanfordCars(root=root, split="train", download=download)
        case "stanford-cars_test":
            return torchvision.datasets.StanfordCars(root=root, split="test", download=download)

        case "oxford-pet_train":
            return torchvision.datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", download=download)
        case "oxford-pet_test":
            return torchvision.datasets.OxfordIIITPet(root=root, split="test", target_types="category", download=download)

        case "cub200_train":
            return CUB200(root=root, train=True, download=download)
        case "cub200_test":
            return CUB200(root=root, train=False, download=download)

        case "flowers102_train":
            return torchvision.datasets.Flowers102(root=root, split="train", download=download)
        case "flowers102_val":
            return torchvision.datasets.Flowers102(root=root, split="val", download=download)
        case "flowers102_test":
            return torchvision.datasets.Flowers102(root=root, split="test", download=download)

        case "tiny-imagenet_train":
            return TinyImageNet(root=root, train=True, download=download)
        case "tiny-imagenet_test":
            return TinyImageNet(root=root, train=False, download=download)

        case "imagenet_train":
            return torchvision.datasets.ImageNet(root=root, split="train")
        case "imagenet_val":
            return torchvision.datasets.ImageNet(root=root, split="val")
        case "imagenet_test":
            return torchvision.datasets.ImageNet(root=root, split="test")

        case "caltech101_labeled":
            return torchvision.datasets.Caltech101(root=root, target_type="category", download=download)
        
        case _ if dataset_name.startswith("cifar10_c_"):
            return CIFAR10C(root=root, corruption=dataset_name.replace("cifar10_c_", ""), download=download)

        case _ if dataset_name.startswith("cifar100_c_"):
            return CIFAR100C(root=root, corruption=dataset_name.replace("cifar100_c_", ""), download=download)

        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        
