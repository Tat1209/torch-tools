import os
import pickle
import shutil
from pathlib import Path
from typing import Optional, Callable, Any

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

# 追加: 排他制御用ライブラリ
from filelock import FileLock

# dataset_handlerモジュールが存在することを前提としています
from dataset_handler import DatasetHandler


class FixRandomDataset(Dataset):
    def __init__(self, size: int):
        """固定シードでランダムデータを生成するデバッグ用データセット。

        Args:
            size (int): データの次元数。
        """
        super().__init__()
        torch.manual_seed(42)
        self.data = torch.rand(size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """データを取得する。

        Args:
            index (int): インデックス。

        Returns:
            tuple[torch.Tensor, int]: 複製されたデータテンソルとインデックス。
        """
        data = self.data.detach().clone()
        target = index
        return data, target

    def __len__(self) -> int:
        """データセット長を返す。

        Returns:
            int: 固定長10000。
        """
        return 10000


class PklToDataset(Dataset):
    def __init__(self, pkl_path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """Pickleファイルからデータを一括ロードするデータセット。

        Note:
            Pickleは全データをメモリに展開するため、大規模データセットには不向きです。
            大容量データの場合はLMDBやHDF5、またはWebDatasetの使用を推奨します。

        Args:
            pkl_path (str): Pickleファイルのパス。
            transform (Callable, optional): データ用変換関数。
            target_transform (Callable, optional): ターゲット用変換関数。
        """
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
    """Tiny ImageNet-200データセットローダー。"""
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    dataset_folder = "tiny-imagenet-200"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, train: bool = True, download: bool = False):
        """
        Args:
            root (str): データセットのルートディレクトリ。
            transform (Callable, optional): 画像変換。
            target_transform (Callable, optional): ラベル変換。
            train (bool): Trueで学習セット、Falseで検証セット。
            download (bool): Trueの場合、データをダウンロードする。
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        self.train = train
        self.base_path = self.root / self.dataset_folder

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it.")

        self.paths = []
        self.targets = []
        self.dirname_label = {}

        self._load_meta()
        self._load_data()

    def _load_meta(self) -> None:
        """クラスIDとラベルのマッピングを読み込む。"""
        wnids_path = self.base_path / "wnids.txt"
        with open(wnids_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.dirname_label[line.strip()] = i

    def _load_data(self) -> None:
        """画像パスとターゲットをリストに格納する。"""
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
            print("Files already downloaded and verified")
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

        # PILで読み込み、RGB変換（グレースケール画像対策）
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target


class StanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, root: str, split: str = "train", download: bool = False, **kwargs):
        """Stanford Carsデータセット（Hugging Faceミラー対応版）。
        
        Args:
            root (str): 保存先ディレクトリ。
            split (str): "train" または "test"。
            download (bool): Trueでダウンロードを実行。
        """
        self.root = root
        if download:
            self._custom_download()

        # 親クラスのdownloadは失敗するためFalse固定
        try:
            super().__init__(root, split=split, download=False, **kwargs)
        except RuntimeError as e:
            raise RuntimeError("Dataset not found. Please use download=True.") from e

    def _custom_download(self) -> None:
        """ミラーサイトからデータをダウンロードし、torchvision準拠の配置を行う。"""
        # Hugging FaceのミラーURL
        resources = [
            ("https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/cars_train.tgz", "cars_train.tgz"),
            ("https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/cars_test.tgz", "cars_test.tgz"),
            ("https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/car_devkit.tgz", "car_devkit.tgz")
        ]
        
        # torchvision.datasets.StanfordCarsはroot直下に展開されたフォルダを期待する
        # (例: root/cars_train, root/devkit)
        
        for url, filename in resources:
            # 簡易チェック: 展開後の主要フォルダが存在すればスキップ
            # 注意: car_devkit.tgz は 'devkit' という名前で展開される可能性があるため確認が必要だが
            # ここではアーカイブファイルの有無と、展開ロジックに任せる
            
            # 展開先チェック用マッピング
            check_folder = "devkit" if "devkit" in filename else filename.replace(".tgz", "")
            if os.path.exists(os.path.join(self.root, check_folder)):
                continue

            print(f"Downloading {filename}...")
            download_and_extract_archive(url, download_root=self.root, filename=filename)


class TestDataset(Dataset):
    def __init__(self, path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """ディレクトリ内の画像をテスト用に読み込むデータセット。

        Args:
            path (str): 画像フォルダのパス。
        """
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
    """データセットを取得する（FileLockを用いた排他的ダウンロード付き）。

    プロセスの競合を防ぐため、以下のロジックを実行します：
    1. ロックなしで読み込み試行（高速化のため）。
    2. 失敗した場合、ロックを取得。
    3. ロック内でもう一度読み込み試行（ロック待ちの間に別プロセスが完了させた可能性があるため）。
    4. それでも失敗した場合のみダウンロードを実行。

    Args:
        root (str): 保存先ルートディレクトリ。
        dataset_name (str): データセット識別子。

    Returns:
        Dataset: 初期化されたデータセット。

    Raises:
        RuntimeError: ダウンロード失敗や未対応の引数エラー。
    """
    # 1. 楽観的チェック: ロックなしでロードを試みる
    try:
        return _datasets(root, dataset_name, download=False)
    except (RuntimeError, FileNotFoundError):
        pass  # 次のロック処理へ進む

    # ロックファイルのパス設定
    os.makedirs(root, exist_ok=True)
    lock_file = os.path.join(root, f"{dataset_name.replace('/', '_')}.lock")

    # 2. 排他制御区間
    with FileLock(lock_file):
        try:
            # 3. 再チェック: ロック取得待ちの間に他プロセスが完了させていないか確認
            return _datasets(root, dataset_name, download=False)
        except (RuntimeError, FileNotFoundError):
            try:
                # 4. 実際にダウンロードを実行
                return _datasets(root, dataset_name, download=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load or download dataset '{dataset_name}'.") from e


def fetch_handler(root: str, dataset_id: str, base_ds: Optional[Dataset] = None) -> Any:
    """DatasetHandlerを生成するラッパー関数。

    Args:
        root (str): ルートディレクトリ。
        dataset_id (str): データセットID。
        base_ds (Dataset, optional): 既存のデータセット。

    Returns:
        DatasetHandler: ハンドラインスタンス。
    """
    if not base_ds:
        base_ds = fetch_dataset(root, dataset_id)
    return DatasetHandler.create(dataset_id, root, base_ds)


def _datasets(root: str, dataset_name: str, download: bool = False) -> Dataset:
    """データセットインスタンス生成の内部ファクトリ。"""
    match dataset_name:
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
            return StanfordCars(root=root, split="train", download=download)
        case "cars_val":
            # 注意: テストセットにはラベルがないため、検証目的には不適切な可能性があります
            return StanfordCars(root=root, split="test", download=download)
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