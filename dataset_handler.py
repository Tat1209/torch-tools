import itertools
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

class StateDict(dict):
    def copy(self):
        new_dict = StateDict()
        for key, value in self.items():
            new_dict[key] = copy(value)
        return new_dict

def fetch_handler(dataset_id, base_ds: Dataset, root=None):
    state = StateDict()
    state["base_ds"] = base_ds
    state["dataset_id"] = dataset_id
    state["dataset_str"] = base_ds.__class__.__name__
    state["indices"] = np.arange(len(base_ds), dtype=np.int32)
    state["classinfo"] = None
    state["transform_l"] = []
    state["target_transform_l"] = []
    if root:
        state["root"] = Path(root)
    else:
        state["root"] = Path(base_ds.root)

    dsh = DatasetHandler(state)
    dsh.state["base_dsh"] = dsh
    return dsh

class DatasetHandler(Dataset):
    def __init__(self, state: StateDict):
        self.state = state
        self._transform = transforms.Compose(self.state["transform_l"]) if self.state["transform_l"] else None
        self._target_transform = transforms.Compose(self.state["target_transform_l"]) if self.state["target_transform_l"] else None

    def __getitem__(self, idx):
        data, target = self.state["base_ds"][self.state["indices"][idx]]
        if isinstance(self.state["classinfo"], dict):
            target = self.state["classinfo"][target]
        if self._transform:
            data = self._transform(data)
        if self._target_transform:
            target = self._target_transform(target)
        return data, target

    def __len__(self):
        return len(self.state["indices"])

    def loader(self, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, **kwargs):
        if len(self) == 0:
            return None
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    def transform(self, transform_l, replace=False):
        new_state = self.state.copy()
        if replace:
            new_state["transform_l"] = transform_l
        else:
            new_state["transform_l"] += transform_l
        return DatasetHandler(new_state)

    def target_transform(self, target_transform_l, replace=False):
        new_state = self.state.copy()
        if replace:
            new_state["target_transform_l"] = target_transform_l
        else:
            new_state["target_transform_l"] += target_transform_l
        return DatasetHandler(new_state)

    def in_ndata(self, a, b=None):
        new_state = self.state.copy()
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        new_state["indices"] = new_state["indices"][start:end]
        return DatasetHandler(new_state)

    def ex_ndata(self, a, b=None):
        new_state = self.state.copy()
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        mask = np.ones(len(new_state["indices"]), dtype=bool)
        mask[start:end] = False
        new_state["indices"] = new_state["indices"][mask]
        return DatasetHandler(new_state)

    def in_ratio(self, a, b=None):
        new_state = self.state.copy()
        total = len(new_state["indices"])
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        start_idx = int(start * total)
        end_idx = int(end * total)
        new_state["indices"] = new_state["indices"][start_idx:end_idx]
        return DatasetHandler(new_state)

    def ex_ratio(self, a, b=None):
        new_state = self.state.copy()
        total = len(new_state["indices"])
        if isinstance(a, tuple):
            start, end = a
        else:
            start, end = (0, a) if b is None else (a, b)
        start_idx = int(start * total)
        end_idx = int(end * total)
        mask = np.ones(total, dtype=bool)
        mask[start_idx:end_idx] = False
        new_state["indices"] = new_state["indices"][mask]
        return DatasetHandler(new_state)

    def split_ratio(self, ratio, balance_label=False, seed_bl=None):
        a_state = self.state.copy()
        b_state = self.state.copy()
        indices = a_state["indices"].copy()

        if balance_label:
            label_l, label_d = self.fetch_ld()
            a_grp, b_grp = [], []
            for label, idxs in label_d.items():
                arr = idxs.copy()
                if seed_bl != "arange":
                    rng = np.random.default_rng(seed=seed_bl)
                    rng.shuffle(arr)
                split = int(len(arr) * ratio)
                a_grp.extend(arr[:split])
                b_grp.extend(arr[split:])
            a_state["indices"] = np.array(sorted(a_grp), dtype=np.int32)
            b_state["indices"] = np.array(sorted(b_grp), dtype=np.int32)
        else:
            split = int(len(indices) * ratio)
            a_state["indices"] = indices[:split]
            b_state["indices"] = indices[split:]

        return DatasetHandler(a_state), DatasetHandler(b_state)

    def shuffle(self, seed=None):
        new_state = self.state.copy()
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(new_state["indices"])
        return DatasetHandler(new_state)

    def __add__(self, other):
        new_state = self.state.copy()
        new_state["indices"] = np.concatenate((self.state["indices"], other.state["indices"]))
        return DatasetHandler(new_state)

    def limit_class(self, labels=None, max_num=None, rand_num=None, seed=None):
        new_state = self.state.copy()
        _, label_d = self.fetch_ld()
        # mutually exclusive parameters
        if max_num is not None and rand_num is not None:
            raise ValueError("`max_num` and `rand_num` cannot both be specified.")
        # default to all classes
        if labels is None:
            labels = list(label_d.keys())
        if max_num is not None:
            freq = sorted(label_d.items(), key=lambda x: len(x[1]), reverse=True)
            labels = [k for k, _ in freq[:max_num]]
        if rand_num is not None:
            rng = random.Random(seed)
            labels = sorted(rng.sample(labels, rand_num))
        sel = np.concatenate([label_d[label] for label in labels]) if labels else np.array([], dtype=np.int32)
        new_state["indices"] = np.array(sorted(sel), dtype=np.int32)
        new_state["classinfo"] = {label: i for i, label in enumerate(labels)}
        return DatasetHandler(new_state)

    def balance_label(self, seed=None):
        new_state = self.state.copy()
        # unify RNG
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
            # take round-robin
            for _ in range(min(counter[c] for c in valid)):
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
        new_state["indices"] = np.array(indices_new, dtype=np.int32)
        return DatasetHandler(new_state)

    def mult_label(self, mult_dict=None, seed=None):
        new_state = self.state.copy()
        _, label_d = self.fetch_ld()
        indices = []
        for k, v in label_d.items():
            indices.extend(v * mult_dict.get(k, 1))
        indices = np.array(indices, dtype=np.int32)
        if seed != "arange":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
        new_state["indices"] = indices
        return DatasetHandler(new_state)

    def fetch_classes(self, base=False, listed=False):
        classes = None
        if self.state["classinfo"] is None or base:
            blabel_l, blabel_d = self.fetch_ld(base=True)
            if self.state["classinfo"] is None:
                self.state["classinfo"] = list(blabel_d.keys())
            classes = blabel_d.keys() if base else self.state["classinfo"]
        else:
            classes = self.state["classinfo"]
            if isinstance(self.state["classinfo"], dict):
                classes = self.state["classinfo"].keys()
        return list(classes) if listed else len(classes)

    def fetch_classweight(self, setting_id=None, base=True):
        label_l, label_d = self.fetch_ld(base=base)
        classes = max(label_d.keys()) + 1
        label_count_iv = [1.0 / len(label_d.get(i, [])) for i in range(classes)]
        weight_tsr = torch.tensor(label_count_iv, dtype=torch.float)
        return weight_tsr / weight_tsr.sum() * classes

    def fetch_ld(self, base=False, output=False):
        def make_base_ld():
            label_l = []
            label_d = {}
            for idx in self.state["indices"]:
                _, label = self.state["base_ds"][idx]
                label_l.append(label)
                label_d.setdefault(label, []).append(idx)
            return label_l, dict(sorted(label_d.items()))

        blabel_l, blabel_d = self._make_ds_cache(f'{self.state["dataset_id"]}.ld', make_base_ld)
        if base:
            return blabel_l, blabel_d
        label_l = []
        label_d = {i: [] for i in self.fetch_classes(listed=True)}
        for idx in self.state["indices"]:
            label = blabel_l[idx]
            label_l.append(label)
            label_d[label].append(idx)
        if output:
            for lbl, items in label_d.items():
                print(f"{lbl}: {len(items)} items")
            print(f'total: {len(self.state["indices"])} items')
        return label_l, dict(sorted(label_d.items()))

    def normalizer(self, setting_id=None, base=True, inplace=True):
        if setting_id:
            ms_dict = self._make_ds_cache(f"{setting_id}.ms", self.calc_mean_std)
        else:
            ms_dict = self._make_ds_cache(f"{self.state["dataset_id"]}.ms", lambda: self.calc_mean_std(base=True))

        mean, std = ms_dict["mean"], ms_dict["std"]
        return transforms.Normalize(mean=mean, std=std, inplace=inplace)

    def calc_mean_std(self, base=False, batch_size=256, formatted=False):
        if base:
            def make_base_mean_std():
                return self.state["base_dsh"].transform([
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True)
                ], replace=True).calc_mean_std(base=False)
            ms_dict = self._make_ds_cache(f'{self.state["dataset_id"]}.ms', make_base_mean_std)
            mean, std = ms_dict["mean"], ms_dict["std"]
        else:
            elems = []
            for inputs, _ in self.loader(batch_size, shuffle=False):
                p = torch.arange(len(inputs.shape))
                p[0], p[1] = 1, 0
                elems.append(inputs.permute(tuple(p)).reshape(inputs.shape[1], -1))
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

    # def _make_ds_cache(self, file_name, make_f):
    #     path = self.state["base_ds"].root / "ds_cache" / file_name
    #     if path.exists():
    #         obj = torch.load(path, weights_only=False)
    #     else:
    #         obj = make_f()
    #         torch.save(obj, path)
    #         print(f"Saved info to the following path: {path}")
    #     return obj
    
    def _make_ds_cache(self, file_name, make_f):
        cache_dir = self.state["root"] / "ds_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / file_name
        if path.exists():
            return torch.load(path, weights_only=False)
        obj = make_f()
        torch.save(obj, path)
        print(f"Saved cache to {path}")
        return obj