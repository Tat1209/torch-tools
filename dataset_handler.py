import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms


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
    def __init__(self, state: DatasetState, base_dsh=None):
        self.state = state
        self.base_dsh = base_dsh if base_dsh is not None else self

        self._transform = transforms.Compose(self.state.transform_l) if self.state.transform_l else None
        self._target_transform = transforms.Compose(self.state.target_transform_l) if self.state.target_transform_l else None

    @classmethod
    def create(cls, dataset_id, root, base_ds):
        initial_state = DatasetState(
            root=Path(root),
            base_ds=base_ds,
            dataset_id=dataset_id,
            dataset_name=base_ds.__class__.__name__,
            indices=np.arange(len(base_ds), dtype=np.int32),
            # classinfo, transform_l, target_transform_l は default / default_factory が使われるため省略可能
        )
        return cls(initial_state, None)

    def _create_new_handler(self, new_state: DatasetState) -> "DatasetHandler":
        # 新しい State を持つ Handler を作成するヘルパー。
        return DatasetHandler(new_state, base_dsh=self.base_dsh)

    def __getitem__(self, idx):
        data, target = self.state.base_ds[self.state.indices[idx]]
        
        if isinstance(self.state.classinfo, dict):
            target = self.state.classinfo[target]
            
        if self._transform:
            data = self._transform(data)
            
        if self._target_transform:
            target = self._target_transform(target)
            
        return data, target

    def __len__(self):
        return len(self.state.indices)

    def loader(self, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, **kwargs):
        if len(self) == 0:
            return None
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    def transform(self, transform_l, replace_flag=False):
        new_transform_l = transform_l if replace_flag else self.state.transform_l + transform_l
        new_state = replace(self.state, transform_l=new_transform_l)
        
        return self._create_new_handler(new_state)

    def target_transform(self, target_transform_l, replace_flag=False):
        new_target_transform_l = target_transform_l if replace_flag else self.state.target_transform_l + target_transform_l
        new_state = replace(self.state, target_transform_l=new_target_transform_l)
        
        return self._create_new_handler(new_state)

    def in_ndata(self, a, b=None):
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
            
        new_indices = self.state.indices[start:end]
        new_state = replace(self.state, indices=new_indices)
        
        return self._create_new_handler(new_state)

    def ex_ndata(self, a, b=None):
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
            
        mask = np.ones(len(self.state.indices), dtype=bool)
        mask[start:end] = False
        
        new_indices = self.state.indices[mask]
        new_state = replace(self.state, indices=new_indices)
        
        return self._create_new_handler(new_state)

    def in_ratio(self, a, b=None):
        total = len(self.state.indices)
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
            
        start_idx = int(start * total)
        end_idx = int(end * total)
        
        new_indices = self.state.indices[start_idx:end_idx]
        new_state = replace(self.state, indices=new_indices)
        
        return self._create_new_handler(new_state)

    def ex_ratio(self, a, b=None):
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
        new_state = replace(self.state, indices=new_indices)
        
        return self._create_new_handler(new_state)

    def split_ratio(self, ratio, balance_class=False, seed_bl=None):
        indices = self.state.indices.copy()

        if balance_class:
            label_l, label_d = self.fetch_ld()
            a_grp, b_grp = [], []
            for label, idxs_list in label_d.items():
                arr = np.array(idxs_list) # label_d はリストを返す
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

    def shuffle(self, seed=None):
        new_indices = self.state.indices.copy()
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(new_indices)
            
        new_state = replace(self.state, indices=new_indices)
        return self._create_new_handler(new_state)

    def __add__(self, other: "DatasetHandler"):
        new_indices = np.concatenate((self.state.indices, other.state.indices))
        new_state = replace(self.state, indices=new_indices)
        
        return self._create_new_handler(new_state)

    def limit_class(self, labels=None, max_num=None, rand_num=None, seed=None):
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
        new_state = replace(self.state, indices=new_indices, classinfo=new_classinfo)
        
        return self._create_new_handler(new_state)

    def balance_class(self, seed=None):
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
        
        new_state = replace(self.state, indices=np.array(indices_new, dtype=np.int32))
        return self._create_new_handler(new_state)

    def mult_class(self, mult_dict=None, seed=None):
        _, label_d = self.fetch_ld()
        indices = []
        for k, v in label_d.items():
            indices.extend(v * mult_dict.get(k, 1))
            
        indices_np = np.array(indices, dtype=np.int32)
        
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices_np)
            
        new_state = replace(self.state, indices=indices_np)
        return self._create_new_handler(new_state)

    def fetch_classes(self, base=False, listed=False):
        classes_keys = None
        
        if base:
            # ベースのクラス情報を要求された場合
            _, blabel_d = self.fetch_ld(base=True)
            classes_keys = blabel_d.keys()
        elif self.state.classinfo is not None:
            # classinfo が設定されている (limit_class 済) 場合
            classes_keys = self.state.classinfo.keys()
        else:
            # classinfo が None (未設定) の場合 (ベースと同じ)
            _, blabel_d = self.fetch_ld(base=True)
            classes_keys = blabel_d.keys()

        return list(classes_keys) if listed else len(classes_keys)

    def fetch_classweight(self, setting_id=None, base=True):
        label_l, label_d = self.fetch_ld(base=base)
        classes = max(label_d.keys()) + 1
        
        # 0除算を避けるため、空リストの場合は 1.0 (または inf) にする
        label_count_iv = []
        for i in range(classes):
            count = len(label_d.get(i, []))
            if count == 0:
                label_count_iv.append(0.0) # 重み 0 にする（推奨）
            else:
                label_count_iv.append(1.0 / count)
                
        weight_tsr = torch.tensor(label_count_iv, dtype=torch.float)
        
        # 重みの合計が0の場合、そのまま返す
        sum_weights = weight_tsr.sum()
        if sum_weights == 0:
            return weight_tsr
            
        return weight_tsr / sum_weights * classes

    def fetch_ld(self, base=False, output=False):
        def make_base_ld():
            label_l = []
            label_d = {}
            base_indices = self.base_dsh.state.indices
            for idx in base_indices:
                _, label = self.base_dsh.state.base_ds[idx]
                label_l.append(label)
                label_d.setdefault(label, []).append(idx)
            return label_l, dict(sorted(label_d.items()))

        blabel_l, blabel_d = self._make_ds_cache( f'{self.base_dsh.state.dataset_id}.ld', make_base_ld)

        if base:
            return blabel_l, blabel_d
        
        label_l_current = []
        current_classes_keys = self.fetch_classes(listed=True)
        label_d_current = {i: [] for i in current_classes_keys}

        # blabel_l (全ラベルリスト) から現在の indices に該当するラベルを取得
        for idx in self.state.indices:
            label = blabel_l[idx] # blabel_l は base_ds の idx に対応
            
            if label in label_d_current: # limit_class 適用後を考慮
                label_l_current.append(label)
                label_d_current[label].append(idx)
                
        if output:
            for lbl, items in label_d_current.items():
                print(f"{lbl}: {len(items)} items")
            print(f'total: {len(self.state.indices)} items')
            
        return label_l_current, dict(sorted(label_d_current.items()))

    def normalizer(self, setting_id=None, base=True, inplace=True):
        if setting_id:
            ms_dict = self._make_ds_cache(f"{setting_id}.ms", self.calc_mean_std)
        else:
            ms_dict = self._make_ds_cache(f'{self.base_dsh.state.dataset_id}.ms', lambda: self.calc_mean_std(base=True))

        mean, std = ms_dict["mean"], ms_dict["std"]
        return transforms.Normalize(mean=mean, std=std, inplace=inplace)

    def calc_mean_std(self, base=False, batch_size=256, formatted=False):
        if base:
            def make_base_mean_std():
                return self.base_dsh.transform([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)], replace_flag=True).calc_mean_std(base=False)
            
            ms_dict = self._make_ds_cache(f'{self.base_dsh.state.dataset_id}.ms', make_base_mean_std)
            mean, std = ms_dict["mean"], ms_dict["std"]
        else:
            elems = []
            for inputs, _ in self.loader(batch_size, shuffle=False):
                p = list(range(inputs.ndim))
                p[0], p[1] = 1, 0
                elems.append(inputs.permute(*p).reshape(inputs.shape[1], -1))
            elem = torch.cat(elems, dim=1)
            mean, std = elem.mean(dim=1).tolist(), elem.std(dim=1).tolist()
            
        return (f"transforms.Normalize(mean={mean}, std={std}, inplace=True)"
                if formatted else {"mean": mean, "std": std})

    def load_data(self, batch_size=256, one_dim=False):
        all_inputs, all_labels = None, None
        for inputs, labels in self.loader(batch_size, shuffle=True):
            if one_dim:
                inputs = inputs.view(len(inputs), -1)
            all_inputs = inputs if all_inputs is None else torch.cat([all_inputs, inputs], dim=0)
            all_labels = labels if all_labels is None else torch.cat([all_labels, labels], dim=0)
        return all_inputs, all_labels

    def _ds_to_folder(self, num, path):
        pass  # left unimplemented

    def _make_ds_cache(self, file_name, make_f):
        cache_dir = self.state.root / "ds_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / file_name
        
        if path.exists():
            return torch.load(path, weights_only=False) 
            
        obj = make_f()
        torch.save(obj, path)
        print(f"Saved cache to {path}")
        return obj