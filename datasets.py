import pickle
import os
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from dataset_handler import DatasetHandler


class FixRandomDataset(Dataset):
    def __init__(self, size: int):
        """固定のランダムデータを生成するデータセット。

        Args:
            size (int): データサイズ（次元数）。
        """
        super().__init__()
        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """データを取得する。

        Args:
            index (int): インデックス。

        Returns:
            tuple[torch.Tensor, int]: データとインデックス（ターゲット）。
        """
        data = self.data.detach().clone()
        target = index
        return data, target

    def __len__(self) -> int:
        """データセットの長さを返す。

        Returns:
            int: 固定値10000。
        """
        return 10000


class PklToDataset(Dataset):
    def __init__(self, pkl_path: str, transform=None, target_transform=None):
        """Pickleファイルからデータを読み込むデータセット。

        Args:
            pkl_path (str): Pickleファイルのパス。
            transform (callable, optional): 入力データへの変換。
            target_transform (callable, optional): ターゲットへの変換。
        """
        with open(pkl_path, "rb") as f:
            (self.datas, self.targets) = pickle.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """データ数を返す。

        Returns:
            int: データ数。
        """
        return len(self.datas)

    def __getitem__(self, idx: int) -> tuple:
        """データを取得する。

        Args:
            idx (int): インデックス。

        Returns:
            tuple: 変換済みのデータとターゲット。
        """
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

    def __init__(self, root: str, transform=None, target_transform=None, train: bool = True, download: bool = False):
        """Tiny ImageNet-200データセット。

        Args:
            root (str): ルートディレクトリ。
            transform (callable, optional): 画像への変換。
            target_transform (callable, optional): ラベルへの変換。
            train (bool): Trueなら学習セット、Falseなら検証セット。
            download (bool): Trueならデータをダウンロードする。
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        ti_path = self.root / self.dataset_folder

        if download:
            self._download()

        if not ti_path.exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.paths = []
        self.targets = []
        self.dirname_label = {}

        # クラスIDとラベル値のマッピング
        with open(ti_path / "wnids.txt", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                key = line.strip()
                self.dirname_label[key] = i

        if train:
            ti_train_path = ti_path / "train"
            for class_dir in ti_train_path.iterdir():
                if class_dir.is_dir():
                    images_dir = class_dir / "images"
                    for image_path in images_dir.iterdir():
                        if image_path.is_file():
                            self.paths.append(image_path)
                            self.targets.append(self.dirname_label[class_dir.name])
        else:
            ti_val_path = ti_path / "val"
            images_dir = ti_val_path / "images"
            with open(ti_val_path / "val_annotations.txt", 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                elems = line.split()
                image_name = elems[0]
                class_name = elems[1]
                image_path = images_dir / image_name
                
                if image_path.is_file():
                    self.paths.append(image_path)
                    self.targets.append(self.dirname_label[class_name])

    def _download(self) -> None:
        """データセットをダウンロード・解凍する。"""
        if (self.root / self.dataset_folder).exists():
            print("Dataset already exists.")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5
        )

    def __len__(self) -> int:
        """データ数を返す。

        Returns:
            int: 画像パスの総数。
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        """画像とラベルを取得する。

        Args:
            idx (int): インデックス。

        Returns:
            tuple: 画像(PIL)とラベル(int)。
        """
        data = Image.open(self.paths[idx]).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target


class TestDataset(Dataset):
    def __init__(self, path: str, transform=None, target_transform=None):
        """テスト用データセット（フォルダ内の画像を読み込む）。

        Args:
            path (str): 画像フォルダのパス。
            transform (callable, optional): 画像への変換。
            target_transform (callable, optional): パスへの変換。
        """
        self.root = Path(path).parent
        self.img_paths = sorted([p for p in Path(path).iterdir()])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> tuple:
        """データを取得する。

        Args:
            index (int): インデックス。

        Returns:
            tuple: 画像とファイル名。
        """
        path = self.img_paths[index]
        data = Image.open(path).convert('RGB')

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            path = self.target_transform(path)
        return data, str(path.name)

    def __len__(self) -> int:
        """画像数を返す。

        Returns:
            int: 画像数。
        """
        return len(self.img_paths)


def fetch_dataset(root: str, dataset_name: str):
    """データセットを取得する。

    Args:
        root (str): データセットのルートディレクトリ。
        dataset_name (str): データセット名。

    Returns:
        Dataset: 指定されたデータセットインスタンス。
    
    Raises:
        RuntimeError: ダウンロード不可のエラー等。
    """
    try:
        base_ds = _datasets(root, dataset_name, download=False)
    except (RuntimeError, FileNotFoundError):
        try:
            base_ds = _datasets(root, dataset_name, download=True)
        except (RuntimeError, FileNotFoundError) as e:
            raise RuntimeError("The 'download' argument is not supported or failed for this dataset.") from e
    return base_ds


def fetch_handler(root: str, dataset_id: str, base_ds=None):
    """データセットハンドラを取得する。

    Args:
        root (str): ルートディレクトリ。
        dataset_id (str): データセットID。
        base_ds (Dataset, optional): 既存のデータセットインスタンス。

    Returns:
        DatasetHandler: ハンドラインスタンス。
    """
    if not base_ds:
        base_ds = fetch_dataset(root, dataset_id)
    return DatasetHandler.create(dataset_id, root, base_ds)


def _datasets(root: str, dataset_name: str, download: bool = False):
    """データセットインスタンスを生成するファクトリ関数。

    Args:
        root (str): 保存先ルート。
        dataset_name (str): データセット識別子。
        download (bool): ダウンロードフラグ。

    Returns:
        Dataset: PyTorchデータセット。

    Raises:
        ValueError: 未対応のデータセット名の場合。
    """
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
            return TinyImageNet(root=root, train=True, download=download)
        case "tiny-imagenet_val":
            return TinyImageNet(root=root, train=False, download=download)
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