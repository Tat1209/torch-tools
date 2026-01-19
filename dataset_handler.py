import json
import os
import random
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from filelock import FileLock

from transforms_util import GrayToRGB

class LabelRemapper:
    """ラベル変換用クラス"""
    def __init__(self, mapping: Dict[Any, Any]):
        self.mapping = mapping

    def __call__(self, t: Any) -> Any:
        return self.mapping.get(t, t)


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

        tt_list = []
        if isinstance(self.state.classinfo, dict):
            tt_list.append(transforms.Lambda(LabelRemapper(self.state.classinfo)))
        
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

    # --- Data Manipulation Methods (省略なし) ---

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
        stratify: bool = False,
        shuffle: bool = True,
        seed: Optional[Any] = None
    ) -> Tuple["DatasetHandler", "DatasetHandler"]:
        indices = self.state.indices.copy()
        rng = np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()

        if stratify:
            _, label_d = self.fetch_ld()
            a_grp, b_grp = [], []
            for _, idxs_list in label_d.items():
                arr = np.array(idxs_list)
                if shuffle:
                    rng.shuffle(arr)
                split = int(len(arr) * ratio)
                a_grp.extend(arr[:split])
                b_grp.extend(arr[split:])
            a_indices = np.array(sorted(a_grp), dtype=np.int32)
            b_indices = np.array(sorted(b_grp), dtype=np.int32)
        else:
            if shuffle:
                rng.shuffle(indices)
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

    def in_nshot(
        self,
        ipc: int | Literal["max", "all"],
        mode: Literal["strict", "under", "over"] = "strict",
        seed: Optional[int] = None,
        notall: bool = False,
        notmax: bool = False
    ) -> "DatasetHandler":
        """各クラスから指定された枚数(N-shot)を抽出した新しいデータセットハンドラを作成します。

        `mode` 引数によって、データ不足時の挙動（厳格なチェック、アンダーサンプリング、オーバーサンプリング）を制御できます。
        内部で `get_ipc()` を使用してデータセットの状態を判定します。

        Args:
            ipc (int | str): クラスごとに取得する画像の枚数。
                * 正の整数: 指定された枚数をターゲットとします。
                * 'max': `mode` に応じて自動設定します。
                    (`over`時は最大クラス数、`strict`/`under`時は最小クラス数に合わせます)
                * 'all': フィルタリングを行わず、現在の全データをそのまま返します。
            mode (str, optional): データ不足時および均衡化の挙動。デフォルトは 'strict'。
                * 'strict': 指定枚数が実データ数を超えているクラスがある場合、ValueError を発生させます。
                * 'under': 指定枚数が実データ数を超える場合、全クラスの中で「最小の枚数」に合わせて切り捨て（Undersampling）を行います。
                * 'over': 指定枚数が実データ数を超える場合、不足分を既存データの複製（Oversampling）で補います。
            seed (int, optional): シャッフルおよびサンプリングに使用する乱数シード。
                None の場合はランダムなシードが使用されます。
            notall (bool, optional): Trueの場合、結果が「元の全データセットと同一」になる設定であれば ValueError を発生させます。
                ただし、ipc="all" が明示的に指定された場合はこのフラグは無視され、実行されます。
            notmax (bool, optional): Trueの場合、結果が「ipc='max' と同一」になる設定であれば ValueError を発生させます。
                ただし、ipc="max" が明示的に指定された場合はこのフラグは無視され、実行されます。

        Returns:
            DatasetHandler: 処理適用後の新しいデータセットハンドラ。

        Raises:
            ValueError: `ipc` に無効な値が指定された場合、`mode='strict'` 時にデータ数が不足している場合、
                        あるいは `notall`/`notmax` の条件に抵触した場合。
        """
        # "all" が明示された場合は、notallフラグに関わらず実行
        if ipc == "all":
            return self._create_new_handler(self.state)

        # 乱数生成器の初期化
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # 統計情報を取得
        stats = self.get_ipc()
        min_count = stats["min"]
        max_count = stats["max"]
        is_balanced = stats["is_balanced"]

        # データが空の場合のガード
        if max_count == 0:
            return self._create_new_handler(self.state)

        # ターゲット枚数(target_n)の決定
        target_n = 0
        if ipc == "max":
            # "max" が明示された場合は、notmaxフラグに関わらず実行
            if mode == "over":
                target_n = max_count
            else:
                target_n = min_count
        elif isinstance(ipc, int) and ipc > 0:
            target_n = ipc
            # mode="under" の場合、指定ipcが実データより多ければ、強制的にmin_countに下げる
            if mode == "under" and target_n > min_count:
                print(f"[INFO] in_nshot(mode='under'): Requested {target_n} is larger than min_class_count ({min_count}). Clip to {min_count}.")
                target_n = min_count
        else:
            raise ValueError(f"Invalid ipc value: {ipc}")

        # --- 重複設定チェック (数値指定の場合のみ有効) ---
        if isinstance(ipc, int):
            # notmaxチェック: 結果がipc="max"の挙動と一致するか
            max_equivalent = max_count if mode == "over" else min_count
            if notmax and target_n == max_equivalent:
                 raise ValueError(f"[Skip] Resulting ipc ({target_n}) is identical to ipc='max' behavior ({max_equivalent}).")

            # notallチェック: 結果が元の全データセットと一致するか
            if notall and is_balanced and target_n == min_count:
                raise ValueError(f"[Skip] Resulting ipc ({target_n}) covers the full dataset (identical to ipc='all').")

        # --- 事前チェック (Fail Fast) ---
        # strictモードで、要求枚数が最小クラス枚数を超えている場合は即座にエラー
        if mode == "strict" and target_n > min_count:
            raise ValueError(
                f"[Strict] Requested {target_n} samples, but the smallest class has only {min_count} samples."
            )

        # --- データ抽出処理 ---
        _, label_d = self.fetch_ld()
        sorted_classes = sorted(label_d.keys())
        new_indices = []

        for cls in sorted_classes:
            indices = np.array(label_d[cls])
            n_class = len(indices)

            # Strictチェック (個別)
            if mode == "strict" and target_n > n_class:
                raise ValueError(f"[Strict] Class {cls} has only {n_class} samples, but {target_n} were requested.")

            rng.shuffle(indices)

            if n_class >= target_n:
                # --- Downsampling (通常) ---
                selected = indices[:target_n]
            else:
                # --- Oversampling (mode="over") ---
                # 足りない場合は複製して埋める
                q, r = divmod(target_n, n_class)
                tiled = np.tile(indices, q)
                # 余りは重複なしでランダムに選ぶ
                remainder = rng.choice(indices, size=r, replace=False)
                selected = np.concatenate([tiled, remainder])

            new_indices.extend(selected)

        # 最終的なインデックスをソートして格納
        final_indices = np.array(sorted(new_indices), dtype=np.int32)
        return self._create_new_handler(replace(self.state, indices=final_indices))

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

    def get_ipc(self) -> Dict[str, Union[bool, int, float, None]]:
        """
        IPC (Images Per Class) に関する詳細な統計情報を辞書形式で取得する。

        Returns:
            dict: 以下のキーを含む辞書。
                - "is_balanced" (bool): 全クラスの枚数が一致しているかどうか。
                - "strict" (int | None): 均衡している場合はその枚数、不均衡な場合はNone。
                - "min" (int): 最小クラスの枚数。
                - "max" (int): 最大クラスの枚数。
                - "mean" (float): クラスあたりの平均枚数。
                - "std" (float): クラス枚数の標準偏差。
                - "ratio" (float): 不均衡比 (Max / Min)。1.0なら完全均衡。
        """
        _, label_d = self.fetch_ld()
        
        # データが存在しない場合の初期値
        if not label_d:
            return {
                "is_balanced": True,
                "strict": 0,
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "std": 0.0,
                "ratio": 1.0
            }

        # 各クラスのデータ数リスト
        counts = [len(indices) for indices in label_d.values()]
        
        if not counts:
             return {
                "is_balanced": True,
                "strict": 0,
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "std": 0.0,
                "ratio": 1.0
            }

        # 統計量の計算
        min_c = min(counts)
        max_c = max(counts)
        mean_c = sum(counts) / len(counts)
        
        # 標準偏差 (numpy依存を避けるため手動計算、母標準偏差)
        variance = sum((x - mean_c) ** 2 for x in counts) / len(counts)
        std_c = variance ** 0.5
        
        # 不均衡比 (ゼロ除算回避)
        if min_c == 0:
            ratio_c = float('inf')
        else:
            ratio_c = max_c / min_c

        # 均衡判定
        is_balanced = (min_c == max_c)

        return {
            "is_balanced": is_balanced,
            "strict": min_c if is_balanced else None,
            "min": min_c,
            "max": max_c,
            "mean": mean_c,
            "std": std_c,
            "ratio": ratio_c
        }

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
        base: bool = False,
        use_cache_if_identical: bool = True,
        inplace: bool = True,
        include_preprocess: bool = False
    ) -> Union[transforms.Compose, transforms.Normalize]:
        """Normalize変換を取得する。

        Args:
            setting_id (str, optional): キャッシュ識別子。
            base (bool): 強制的にBaseキャッシュを使用するか。
            use_cache_if_identical (bool): 現在のデータセットがBaseと同一の場合にキャッシュを利用するか。
            inplace (bool): Inplace処理。
            include_preprocess (bool): 前処理を含めるか。
        """
        ms_dict = self.calc_mean_std(
            base=base,
            use_cache_if_identical=use_cache_if_identical,
            setting_id=setting_id
        )

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
        use_cache_if_identical: bool = True,
        batch_size: int = 256,
        formatted: bool = False,
        setting_id: Optional[str] = None,
        size: int = 224
    ) -> Any:
        """平均と標準偏差を計算する（キャッシュ制御・ポリシー決定層）。

        Args:
            base (bool): Baseデータセットの統計量を使用するか。
            use_cache_if_identical (bool): データ内容が同一の場合にキャッシュを利用するか。
            batch_size (int): 計算時のバッチサイズ。
            formatted (bool): 文字列形式で返すか。
            setting_id (str, optional): キャッシュ保存時のID。
            size (int): フォールバック時のリサイズサイズ。

        Returns:
            Any: 辞書 {'mean': ..., 'std': ...} またはフォーマット済み文字列。
        """
        # 1. ターゲットの特定（Baseを使用するか、自分自身か）
        target_dsh = self
        use_base_mode = base

        if not base and use_cache_if_identical:
            # ルートデータセットそのものであるか、インデックスが完全一致する場合
            if self.base_dsh is self:
                use_base_mode = True
            elif len(self.state.indices) == len(self.base_dsh.state.indices):
                if np.array_equal(self.state.indices, self.base_dsh.state.indices):
                    use_base_mode = True
        
        if use_base_mode:
            target_dsh = self.base_dsh

        # 2. キャッシュ利用または計算の実行
        if use_base_mode:
            # Baseモード: キャッシュシステムを経由して計算メソッドを呼ぶ
            if setting_id:
                cache_key = f"{setting_id}.ms"
            else:
                cache_key = f'{target_dsh.state.dataset_id}.ms'

            # ラムダ内で target_dsh._compute_raw_stats を呼ぶ（calc_mean_stdは呼ばない）
            def compute_wrapper():
                return target_dsh._compute_raw_stats(batch_size=batch_size, size=size)

            ms_dict = self._make_ds_cache(cache_key, compute_wrapper)
        
        else:
            # Localモード: キャッシュせず直接計算メソッドを呼ぶ
            ms_dict = target_dsh._compute_raw_stats(batch_size=batch_size, size=size)

        mean, std = ms_dict["mean"], ms_dict["std"]

        if formatted:
            return f"transforms.Normalize(mean={mean}, std={std}, inplace=True)"
        return {"mean": mean, "std": std}

    def _compute_raw_stats(self, batch_size: int, size: int) -> Dict[str, List[float]]:
        """統計量計算の実装層（再帰なし・純粋計算）。"""
        
        # 戦略A: 高精度計算（リサイズなし・オリジナルサイズ）
        transform_high_res = [
            transforms.ToImage(),
            GrayToRGB(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
        
        # 戦略B: 安定計算（リサイズあり・フォールバック用）
        transform_resize = [
            transforms.ToImage(),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            GrayToRGB(),
            transforms.ToDtype(torch.float32, scale=True),
        ]

        # 内部関数: 指定されたTransformでループを実行
        def execute_calculation(trans_list: List[Any]) -> Dict[str, List[float]]:
            # 一時的なハンドラを作成してTransformを適用
            temp_handler = self.transform(trans_list, replace_flag=True)
            
            # メインプロセスで実行（エラーハンドリングのため）
            loader = temp_handler.loader(batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
            
            sum_channels = None
            sum_sq_channels = None
            total_pixels = 0
            
            if loader is None:
                return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

            iterator = iter(loader)
            try:
                for batch in iterator:
                    # バッチのアンパックと検証
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    # リスト型（不揃いバッチ）が来た場合は即座に失敗させる -> 戦略Bへ
                    if not isinstance(inputs, torch.Tensor):
                        raise RuntimeError("Batch is not a Tensor (variable size?).")

                    # inputs: (B, C, H, W) -> [C] に集約
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
            finally:
                del iterator
                del loader

            if total_pixels == 0:
                return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
            
            mean_tensor = sum_channels / total_pixels
            var_tensor = (sum_sq_channels / total_pixels) - (mean_tensor ** 2)
            std_tensor = torch.sqrt(torch.clamp(var_tensor, min=0.0))
            
            return {"mean": mean_tensor.tolist(), "std": std_tensor.tolist()}

        # 実行フロー
        try:
            # まず高精度計算を試みる
            return execute_calculation(transform_high_res)
        except (RuntimeError, ValueError):
            # 失敗（サイズ不揃い等）したらリサイズ版で再実行
            # print(f"Warning: Stats calculation failed with original size. Fallback to Resize({size}).")
            return execute_calculation(transform_resize)

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
        """キャッシュ作成・読み込み。"""
        cache_dir = self.state.root / "ds_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not file_name.endswith(".json"):
            file_name += ".json"
            
        path = cache_dir / file_name
        lock_path = cache_dir / (file_name + ".lock")

        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        with FileLock(str(lock_path)):
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        print(f"Loading cache from {path} (after lock)")
                        return json.load(f)
                except json.JSONDecodeError:
                    print(f"Cache corrupted: {path}. Re-calculating...")

            print(f"Generating cache for {file_name}...")
            data = make_f()

            fd, temp_path = tempfile.mkstemp(dir=cache_dir, text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

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
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        return data
