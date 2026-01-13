import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

# pip install filelock
from filelock import FileLock


@dataclass(frozen=True)
class DatasetState:
    root: Path
    base_ds: Dataset
    dataset_id: str
    dataset_name: str

    indices: np.ndarray
    transform_l: List[Any] = field(default_factory=list, compare=False)
    target_transform_l: List[Any] = field(default_factory=list, compare=False)

    classinfo: Optional[Dict[Any, Any]] = None


class DatasetHandler(Dataset):
    def __init__(self, state: DatasetState, base_dsh: Optional["DatasetHandler"] = None):
        self.state = state
        self.base_dsh = base_dsh if base_dsh is not None else self

        self._transform = transforms.Compose(self.state.transform_l) if self.state.transform_l else None

        # __getitem__ 高速化のため、classinfo変換をTransformに統合
        tt_list = []
        if isinstance(self.state.classinfo, dict):
            mapping = self.state.classinfo
            tt_list.append(transforms.Lambda(lambda t: mapping.get(t, t)))
        
        if self.state.target_transform_l:
            tt_list.extend(self.state.target_transform_l)

        self._target_transform = transforms.Compose(tt_list) if tt_list else None

    @classmethod
    def create(cls, dataset_id: str, root: Union[str, Path], base_ds: Dataset) -> "DatasetHandler":
        """DatasetHandlerの初期インスタンスを作成する。"""
        initial_state = DatasetState(
            root=Path(root),
            base_ds=base_ds,
            dataset_id=dataset_id,
            dataset_name=base_ds.__class__.__name__,
            indices=np.arange(len(base_ds), dtype=np.int32),
        )
        return cls(initial_state, None)

    def _create_new_handler(self, new_state: DatasetState) -> "DatasetHandler":
        return DatasetHandler(new_state, base_dsh=self.base_dsh)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """インデックスに対応するデータを取得する。"""
        real_idx = self.state.indices[idx]
        data, target = self.state.base_ds[real_idx]
        
        if self._transform:
            data = self._transform(data)
            
        if self._target_transform:
            target = self._target_transform(target)
            
        return data, target

    def __len__(self) -> int:
        return len(self.state.indices)

    def loader(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs
    ) -> Optional[DataLoader]:
        if len(self) == 0:
            return None
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs
        )

    def transform(self, transform_l: List[Any], replace_flag: bool = False) -> "DatasetHandler":
        new_transform_l = transform_l if replace_flag else self.state.transform_l + transform_l
        new_state = replace(self.state, transform_l=new_transform_l)
        return self._create_new_handler(new_state)

    def target_transform(self, target_transform_l: List[Any], replace_flag: bool = False) -> "DatasetHandler":
        new_target_transform_l = target_transform_l if replace_flag else self.state.target_transform_l + target_transform_l
        new_state = replace(self.state, target_transform_l=new_target_transform_l)
        return self._create_new_handler(new_state)

    # --- Data Filtering Methods ---

    def in_ndata(self, a: Union[int, Tuple[int, int]], b: Optional[int] = None) -> "DatasetHandler":
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        new_indices = self.state.indices[start:end]
        return self._create_new_handler(replace(self.state, indices=new_indices))

    def ex_ndata(self, a: Union[int, Tuple[int, int]], b: Optional[int] = None) -> "DatasetHandler":
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        mask = np.ones(len(self.state.indices), dtype=bool)
        mask[start:end] = False
        new_indices = self.state.indices[mask]
        return self._create_new_handler(replace(self.state, indices=new_indices))

    def in_ratio(self, a: Union[float, Tuple[float, float]], b: Optional[float] = None) -> "DatasetHandler":
        total = len(self.state.indices)
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        start_idx = int(start * total)
        end_idx = int(end * total)
        new_indices = self.state.indices[start_idx:end_idx]
        return self._create_new_handler(replace(self.state, indices=new_indices))

    def ex_ratio(self, a: Union[float, Tuple[float, float]], b: Optional[float] = None) -> "DatasetHandler":
        total = len(self.state.indices)
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        start_idx = int(start * total)
        end_idx = int(end * total)
        mask = np.ones(total, dtype=bool)
        mask[start_idx:end_idx] = False
        new_indices = self.state.indices[mask]
        return self._create_new_handler(replace(self.state, indices=new_indices))

    def split_ratio(
        self,
        ratio: float,
        balance_class: bool = False,
        seed_bl: Optional[Any] = None
    ) -> Tuple["DatasetHandler", "DatasetHandler"]:
        indices = self.state.indices.copy()
        if balance_class:
            _, label_d = self.fetch_ld()
            a_grp, b_grp = [], []
            for _, idxs_list in label_d.items():
                arr = np.array(idxs_list)
                if seed_bl != "arange":
                    rng = np.random.default_rng(seed=seed_bl)
                    rng.shuffle(arr)
                split = int(len(arr) * ratio)
                a_grp.extend(arr[:split])
                b_grp.extend(arr[split:])
            a_indices = np.array(sorted(a_grp), dtype=np.int32)
            b_indices = np.array(sorted(b_grp), dtype=np.int32)
        else:
            split = int(len(indices) * ratio)
            a_indices = indices[:split]
            b_indices = indices[split:]
        a_state = replace(self.state, indices=a_indices)
        b_state = replace(self.state, indices=b_indices)
        return self._create_new_handler(a_state), self._create_new_handler(b_state)

    def shuffle(self, seed: Optional[Any] = None) -> "DatasetHandler":
        new_indices = self.state.indices.copy()
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(new_indices)
        return self._create_new_handler(replace(self.state, indices=new_indices))

    def __add__(self, other: "DatasetHandler") -> "DatasetHandler":
        new_indices = np.concatenate((self.state.indices, other.state.indices))
        return self._create_new_handler(replace(self.state, indices=new_indices))

    def limit_class(
        self,
        labels: Optional[List[Any]] = None,
        max_num: Optional[int] = None,
        rand_num: Optional[int] = None,
        seed: Optional[Any] = None
    ) -> "DatasetHandler":
        _, label_d = self.fetch_ld()
        if max_num is not None and rand_num is not None:
            raise ValueError("`max_num` and `rand_num` cannot both be specified.")
        if labels is None:
            labels = list(label_d.keys())
        if max_num is not None:
            freq = sorted(label_d.items(), key=lambda x: len(x[1]), reverse=True)
            labels = [k for k, _ in freq[:max_num]]
        if rand_num is not None:
            rng = random.Random(seed)
            labels = sorted(rng.sample(labels, rand_num))
        sel = np.concatenate([label_d[label] for label in labels if label in label_d]) if labels else np.array([], dtype=np.int32)
        new_indices = np.array(sorted(sel), dtype=np.int32)
        new_classinfo = {label: i for i, label in enumerate(labels)}
        return self._create_new_handler(replace(self.state, indices=new_indices, classinfo=new_classinfo))

    def balance_class(self, seed: Optional[Any] = None) -> "DatasetHandler":
        rng = None
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
        _, label_d = self.fetch_ld()
        lens = {k: len(v) for k, v in label_d.items()}
        classes = list(lens.keys())
        counter = lens.copy()
        class_array = []
        while True:
            valid = [c for c in classes if counter[c] > 0]
            if not valid:
                break
            perm = np.arange(len(valid))
            if rng is not None:
                perm = rng.permutation(perm)
            shuffled = [valid[i] for i in perm]
            min_count = min(counter[c] for c in valid)
            for _ in range(min_count):
                for key in shuffled:
                    if counter[key] > 0:
                        class_array.append(key)
                        counter[key] -= 1
        shuffled_label_d = {}
        for k, idxs in label_d.items():
            arr = np.array(idxs)
            if rng is not None:
                arr = rng.permutation(arr)
            shuffled_label_d[k] = list(arr)
        indices_new = [shuffled_label_d[cls].pop(0) for cls in class_array]
        return self._create_new_handler(replace(self.state, indices=np.array(indices_new, dtype=np.int32)))

    def mult_class(self, mult_dict: Optional[Dict[Any, int]] = None, seed: Optional[Any] = None) -> "DatasetHandler":
        _, label_d = self.fetch_ld()
        indices = []
        mult_dict = mult_dict or {}
        for k, v in label_d.items():
            indices.extend(v * mult_dict.get(k, 1))
        indices_np = np.array(indices, dtype=np.int32)
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices_np)
        return self._create_new_handler(replace(self.state, indices=indices_np))

    # --- Stats & Cache Methods ---

    def fetch_classes(self, base: bool = False, listed: bool = False) -> Union[List[Any], int]:
        if base:
            _, blabel_d = self.fetch_ld(base=True)
            keys = blabel_d.keys()
        elif self.state.classinfo is not None:
            keys = self.state.classinfo.keys()
        else:
            _, blabel_d = self.fetch_ld(base=True)
            keys = blabel_d.keys()
        return list(keys) if listed else len(keys)

    def fetch_classweight(self, setting_id: Optional[str] = None, base: bool = True) -> torch.Tensor:
        _, label_d = self.fetch_ld(base=base)
        classes = max(label_d.keys()) + 1 if label_d else 0
        counts = [len(label_d.get(i, [])) for i in range(classes)]
        weights = [0.0 if c == 0 else 1.0 / c for c in counts]
        weight_tsr = torch.tensor(weights, dtype=torch.float)
        sum_weights = weight_tsr.sum()
        if sum_weights == 0:
            return weight_tsr
        return weight_tsr / sum_weights * classes

    def fetch_ld(self, base: bool = False, output: bool = False) -> Tuple[List[Any], Dict[Any, List[int]]]:
        def make_base_ld():
            label_l = []
            label_d = {}
            for idx in self.base_dsh.state.indices:
                _, label = self.base_dsh.state.base_ds[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                label_l.append(label)
                label_d.setdefault(label, []).append(int(idx))
            return {"list": label_l, "dict": dict(sorted(label_d.items()))}

        cache_name = f'{self.base_dsh.state.dataset_id}.ld'
        cached_data = self._make_ds_cache(cache_name, make_base_ld)
        
        blabel_l = cached_data["list"]
        blabel_d_raw = cached_data["dict"]
        try:
            sample_key = next(iter(blabel_d_raw))
            if isinstance(sample_key, str) and sample_key.isdigit():
                 blabel_d = {int(k): v for k, v in blabel_d_raw.items()}
            else:
                 blabel_d = blabel_d_raw
        except StopIteration:
            blabel_d = {}

        if base:
            return blabel_l, blabel_d
        
        label_l_current = []
        current_classes_keys = self.fetch_classes(listed=True)
        label_d_current = {i: [] for i in current_classes_keys}
        for idx in self.state.indices:
            label = blabel_l[idx]
            if label in label_d_current:
                label_l_current.append(label)
                label_d_current[label].append(idx)
        if output:
            for lbl, items in label_d_current.items():
                print(f"{lbl}: {len(items)} items")
            print(f'total: {len(self.state.indices)} items')
        return label_l_current, dict(sorted(label_d_current.items()))

    def normalizer(
        self,
        setting_id: Optional[str] = None,
        base: bool = True,
        inplace: bool = True,
        include_preprocess: bool = True
    ) -> Union[transforms.Compose, transforms.Normalize]:
        if setting_id:
            key = f"{setting_id}.ms"
            ms_getter = lambda: self.calc_mean_std(base=base)
        else:
            key = f'{self.base_dsh.state.dataset_id}.ms'
            ms_getter = lambda: self.calc_mean_std(base=base)

        ms_dict = self._make_ds_cache(key, ms_getter)
        norm_op = transforms.Normalize(mean=ms_dict["mean"], std=ms_dict["std"], inplace=inplace)

        if include_preprocess:
            return transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                norm_op
            ])
        return norm_op

    def calc_mean_std(
        self,
        base: bool = False,
        batch_size: int = 256,
        formatted: bool = False
    ) -> Any:
        if base:
            def make_base_mean_std():
                temp_transform = [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True)
                ]
                return self.base_dsh.transform(temp_transform, replace_flag=True).calc_mean_std(base=False)

            cache_key = f'{self.base_dsh.state.dataset_id}.ms'
            ms_dict = self._make_ds_cache(cache_key, make_base_mean_std)
            mean, std = ms_dict["mean"], ms_dict["std"]
        else:
            sum_channels = None
            sum_sq_channels = None
            total_pixels = 0
            for inputs, _ in self.loader(batch_size, shuffle=False):
                reduce_dims = list(range(inputs.ndim))
                reduce_dims.remove(1)
                batch_sum = inputs.sum(dim=reduce_dims)
                batch_sum_sq = (inputs ** 2).sum(dim=reduce_dims)
                batch_pixels = inputs.numel() // inputs.size(1)
                if sum_channels is None:
                    sum_channels = batch_sum
                    sum_sq_channels = batch_sum_sq
                else:
                    sum_channels += batch_sum
                    sum_sq_channels += batch_sum_sq
                total_pixels += batch_pixels
            mean_tensor = sum_channels / total_pixels
            var_tensor = (sum_sq_channels / total_pixels) - (mean_tensor ** 2)
            std_tensor = torch.sqrt(torch.clamp(var_tensor, min=0.0))
            mean = mean_tensor.tolist()
            std = std_tensor.tolist()

        if formatted:
            return f"transforms.Normalize(mean={mean}, std={std}, inplace=True)"
        return {"mean": mean, "std": std}

    def load_data(self, batch_size: int = 256, one_dim: bool = False) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        all_inputs, all_labels = None, None
        for inputs, labels in self.loader(batch_size, shuffle=True):
            if one_dim:
                inputs = inputs.view(len(inputs), -1)
            all_inputs = inputs if all_inputs is None else torch.cat([all_inputs, inputs], dim=0)
            all_labels = labels if all_labels is None else torch.cat([all_labels, labels], dim=0)
        return all_inputs, all_labels

    def _ds_to_folder(self, num: int, path: str):
        pass

    def _make_ds_cache(self, file_name: str, make_f: Callable[[], Any]) -> Any:
        """キャッシュ作成・読み込み。
        
        FileLockによる排他制御と、アトミック置換（tmp -> rename）を組み合わせて
        堅牢性とパフォーマンスを両立させた実装。
        """
        cache_dir = self.state.root / "ds_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not file_name.endswith(".json"):
            file_name += ".json"
            
        path = cache_dir / file_name
        lock_path = cache_dir / (file_name + ".lock")

        # 1. Fast Path (ロックなし・読み込みのみ)
        # 既に完全なファイルが存在する場合は、ロックコストを支払わずにリターン
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # 破損などの可能性がある場合はロック処理へ進む
                pass

        # 2. 排他制御ブロック
        with FileLock(str(lock_path)):
            # 3. Double-Checked Locking
            # ロック待ちの間に他プロセスが完了させたか確認
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        print(f"Loading cache from {path} (after lock)")
                        return json.load(f)
                except json.JSONDecodeError:
                    print(f"Cache corrupted: {path}. Re-calculating...")

            # 4. データ生成 (最初の1プロセスのみ)
            print(f"Generating cache for {file_name}...")
            data = make_f()

            # 5. アトミック書き込み
            # mkstempで一時ファイルを作成 (同じディレクトリ内にして rename を確実に成功させる)
            fd, temp_path = tempfile.mkstemp(dir=cache_dir, text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno()) # ディスクへの物理書き込みを保証

                # Windows対応: 同名ファイルが存在する場合のケア (os.replaceはWinで上書きエラーになる場合がある)
                # ただしFileLock内なので他プロセスとの競合は基本ない
                if os.path.exists(path) and os.name == 'nt':
                     try:
                         os.remove(path)
                     except OSError:
                         pass

                os.replace(temp_path, path)
                print(f"Saved cache to {path}")
                
            except Exception as e:
                print(f"Failed to save cache: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
            finally:
                # 万が一のリソースリーク防止
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        return data