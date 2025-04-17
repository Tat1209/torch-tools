import random
import itertools
from copy import copy

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights
# from umap import UMAP
import matplotlib.pyplot as plt

from trainer import Trainer


# クラスの数を減らすなら、indicesのほかに_classinfo (ラベル名を保管しているリスト) を作ってそれも毎回コピる必要がある。その後、__getitem__のtargetを修正する必要あり
class DatasetHandler(Dataset):
    # self.indicesは、常にnp.array(), label_l, label_dのvalueはlist
    def __init__(self, base_ds, indices, _classinfo, transform, target_transform):
        self.base_ds = base_ds
        self.indices = indices
        self._classinfo = _classinfo
        self._transform = transform
        self._target_transform = target_transform

        self.ds_str = base_ds.ds_str
        self.ds_name = base_ds.ds_name

    def __getitem__(self, idx):
        data, target = self.base_ds[self.indices[idx]]

        if type(self._classinfo) is dict:
            target = self._classinfo[target] # targetをクラス名からインデックスに変換

        if self._transform:
            data = self._transform(data)
        if self._target_transform:
            target = self._target_transform(target)

        return data, target

    def __len__(self):
        return len(self.indices)

    def loader(self, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, **kwargs):
        if len(self) == 0:
            return None
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    def transform(self, transform_l):
        indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        # transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        transform_new = torchvision.transforms.Compose(transform_l)
        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def target_transform(self, target_transform_l):
        indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        # target_transform_new = copy(self._target_transform)

        target_transform_new = torchvision.transforms.Compose(target_transform_l)
        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def in_ndata(self, a, b=None):
        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)
        range_t = (a, b) if b else (0, a)
        if isinstance(a, tuple):
            range_t = (a[0], a[1])
        # data_num = len(self.indices)
        idx_range = (range_t[0], range_t[1])
        indices_new = self.indices[idx_range[0] : idx_range[1]]

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def ex_ndata(self, a, b=None):
        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        range_t = (a, b) if b else (0, a)
        if isinstance(a, tuple):
            range_t = (a[0], a[1])
        # data_num = len(self.indices)
        idx_range = (range_t[0], range_t[1])
        indices_new = list(self.indices[: idx_range[0]]) + list(self.indices[idx_range[1] :])
        indices_new = np.array(indices_new, dtype=np.int32)

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def in_ratio(self, a, b=None):
        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)
        range_t = (a, b) if b else (0, a)
        if isinstance(a, tuple):
            range_t = (a[0], a[1])
        data_num = len(self.indices)
        idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
        indices_new = np.array(self.indices[idx_range[0] : idx_range[1]], dtype=np.int32)

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def ex_ratio(self, a, b=None):
        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        range_t = (a, b) if b else (0, a)
        if isinstance(a, tuple):
            range_t = (a[0], a[1])
        data_num = len(self.indices)
        idx_range = (int(range_t[0] * data_num), int(range_t[1] * data_num))
        indices_new = list(self.indices[: idx_range[0]]) + list(self.indices[idx_range[1] :])
        indices_new = np.array(indices_new, dtype=np.int32)

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def split_ratio(self, ratio, balance_label=False, seed=None):
        # indices_new = self.indices.copy()
        classes_a_new = copy(self._classinfo)
        transform_a_new = copy(self._transform)
        target_transform_a_new = copy(self._target_transform)

        # indices_new = self.indices.copy()
        classes_b_new = copy(self._classinfo)
        transform_b_new = copy(self._transform)
        target_transform_b_new = copy(self._target_transform)

        if balance_label:
            label_l, label_d = self.fetch_ld()
            a_d = {}
            b_d = {}
            for key in label_d:
                lst = label_d[key]
                if seed != "arange":
                    if seed is not None and not isinstance(seed, int):
                        raise TypeError("Variables must be of type int, 'None', or the string 'arange'.")
                    rng = np.random.default_rng(seed=seed)
                    rng.shuffle(lst)
                length = len(lst)
                a_idx = lst[: int(length * ratio)]
                b_idx = lst[int(length * ratio) :]

                a_d[key] = a_idx
                b_d[key] = b_idx

            indices_a_new = np.array(list(itertools.chain(*a_d.values())), dtype=np.int32)
            indices_b_new = np.array(list(itertools.chain(*b_d.values())), dtype=np.int32)

            indices_a_new.sort()
            indices_b_new.sort()

        else:
            indices_new = self.indices.copy()
            if seed != "arange":
                if seed is not None and not isinstance(seed, int):
                    raise TypeError("Variables must be of type int, 'None', or the string 'arange'.")
                rng = np.random.default_rng(seed=seed)
                rng.shuffle(indices_new)
            length = len(indices_new)
            indices_a_new = indices_new[: int(length * ratio)]
            indices_b_new = indices_new[int(length * ratio) :]

        a = DatasetHandler(self.base_ds, indices_a_new, classes_a_new, transform_a_new, target_transform_a_new)
        b = DatasetHandler(self.base_ds, indices_b_new, classes_b_new, transform_b_new, target_transform_b_new)

        return a, b

    def shuffle(self, seed=None):
        # データセットそのものの順序をシャッフル ただし、ロードごとにシャッフルしたいならDataLoaderでシャッフルさせるべき
        indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        if seed != "arange":
            if seed is not None and not isinstance(seed, int):
                raise TypeError("Variables must be of type int, 'None', or the string 'range'.")
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices_new)

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    # classの処理がされていない
    def __add__(self, other):
        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        indices_new = np.concatenate((self.indices, other.indices))

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def limit_class(self, labels: list=None, max_num=None, rand_num=None, seed=None):
        # indices_new = self.indices.copy()
        # classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        label_l, label_d = self.fetch_ld()

        if max_num is not None:
            # if max_num != self.fetch_classes():
                # {k:len(v) for k, v in label_d.items()}
                # labels = list(dict(sorted(label_d.items(), key=lambda item: len(item[1]), reverse=True)).keys())[:max_num]
            labels = sorted(list(dict(sorted(label_d.items(), key=lambda item: len(item[1]), reverse=True)[:max_num]).keys()))
            # else:
                # return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

        if rand_num is not None:
            rng = random.Random(seed)
            labels = sorted(rng.sample(range(self.fetch_classes()), rand_num))

        classinfo_new = labels
        indices_new = np.array([], dtype=np.int32)

        for label in labels:
            indices = label_d[label]
            indices_new = np.concatenate((indices_new, indices))
        indices_new = sorted(indices_new)

        classinfo_new = {label: i for i, label in enumerate(classinfo_new)}

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def balance_label(self, seed=None):
        # len(_classinfo)ごとに取り出したとき、常に要素の数が極力均等になるようにデータセットのindicesを構成
        # seed="arange"で、該当indicesをクラスが若い順から順番に、indicesの小さい順でとってくる

        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        if seed != "arange":
            rng = np.random.default_rng(seed=seed)

        label_l, label_d = self.fetch_ld()

        # 各クラスのラベル数を取得・クラス名を取得
        lens_d = {k: len(v) for k, v in label_d.items()}
        classes = lens_d.keys()

        # クラス名だけで構成されたclass_arrayを作成
        class_array = []
        counter = lens_d.copy()
        while True:
            valid_keys = [c for c in classes if 0 < counter[c]]  # len_dのkeyのうち、まだvalue個格納されていないものを選択する
            if max(counter.values()) == 0:
                break
            for _ in range(min([labels for labels in counter.values() if 0 < labels])):
                m = len(valid_keys)
                perm = np.arange(m)
                if seed != "arange":
                    np.random.shuffle(perm)
                shuffled_keys = [valid_keys[p] for p in perm]

                for key in shuffled_keys:  # シャッフルされたkeyに対してループする
                    class_array.append(key)
                    counter[key] -= 1

        # label_dのすべてのkeyについて、その要素を全て並び替えたshuffled_label_dを作成
        shuffled_label_d = dict()
        for key in label_d.keys():
            perm = np.array(label_d[key])
            if seed != "arange":
                perm = rng.permutation(perm)
                # perm = np.random.permutation(perm)
            shuffled_label_d[key] = perm

        # class_arrayでクラス、shuffled_label_dでindexを取得
        indices_new = []
        for key in class_array:
            value = shuffled_label_d[key][0]  # label_d[key]のリストから先頭の要素を取り出す
            indices_new.append(value)
            shuffled_label_d[key] = shuffled_label_d[key][1:]  # shuffled_label_d[key]のリストから先頭の要素を削除する

        indices_new = np.array(indices_new, dtype=np.int32)

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def mult_label(self, mult_dict=None, seed=None):
        # indices_new = self.indices.copy()
        classinfo_new = copy(self._classinfo)
        transform_new = copy(self._transform)
        target_transform_new = copy(self._target_transform)

        label_l, label_d = self.fetch_ld()
        for k, v in mult_dict.items():
            label_d[k] *= v
        indices_new = np.array([item for sublist in label_d.values() for item in sublist], dtype=np.int32)

        if seed == "arange":
            indices_new.sort()
        else:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices_new)

        return DatasetHandler(self.base_ds, indices_new, classinfo_new, transform_new, target_transform_new)

    def fetch_classes(self, base=False, listed=False):
        classes = None
        if self._classinfo is None  or  base:
            blabel_l, blabel_d = self._fetch_base_info("ld")

            if self._classinfo is None:
                self._classinfo = list(blabel_d.keys())

            if base:
                classes = list(blabel_d.keys())
            else:
                classes = self._classinfo
        else:
            classes = self._classinfo
            # このelseに入らない場合、必ずdictではないため、この場所でok もう一つ上の階層でもOKでそのほうがわかりやすいが、この無駄な処理が増える
            if type(self._classinfo) is dict:
                classes = self._classinfo.keys()

        if listed:
            return classes
        else:
            return len(classes)

    def fetch_ld(self, base=False, output=False):
        """
            base    : もとのデータセットの情報を取得
            output  : stdoutに出力
        """
        blabel_l, blabel_d = self._fetch_base_info("ld")
        if base:
            return blabel_l, blabel_d

        else:
            label_l = []  # label のリストを作成

            # index と対応させ、label を key とし、index を item とした dict を作成
            label_d = {i: [] for i in self.fetch_classes(listed=True)} 
            for idx in self.indices:
                label = blabel_l[idx]
                label_l.append(label)
                label_d[label].append(idx)

            label_d = dict(sorted(label_d.items()))

            if output:
                for label in label_d.keys():
                    print(f"{label}: {len(label_d[label])} items")
                print(f"total: {len(self.indices)} items")

            return label_l, label_d

    def fetch_classweight(self, base=False):
        """
        Ex.)
        loss_func = torch.nn.CrossEntropyLoss(weight=train_ds.fetch_weight(base=True).to(device))
        """
        label_l, label_d = self.fetch_ld()
        if base:
            blabel_l, blabel_d = self.fetch_base_ld()
            classes = max(blabel_d.keys()) + 1

        else:
            classes = max(label_d.keys()) + 1

        label_count_iv = [1.0 / len(label_d.get(i, [])) for i in range(classes)]  # インデックスが数字以外だと機能しない
        weight_tsr = torch.tensor(label_count_iv, dtype=torch.float) / sum(label_count_iv) * classes

        return weight_tsr

    def normalizer(self, base=True, inplace=True):
        ms_dict = self.calc_mean_std(base=base)
        mean, std = ms_dict["mean"], ms_dict["std"]
        return torchvision.transforms.Normalize(mean=mean, std=std, inplace=inplace)

    def calc_mean_std(self, base=False, batch_size=256, formatted=False):
        if base:
            return self._fetch_base_info("msd")
        # elem = None
        # for inputs, _ in self.loader(batch_size, shuffle=False):
        #     # p は、バッチの次元を除いたものが2次元データなら(1, 0, 2, 3)、1次元データなら(1, 0, 2)
        #     p = torch.arange(len(inputs.shape))
        #     p[0], p[1] = 1, 0
        #     p = tuple(p)

        #     elem_b = inputs.permute(p).reshape(inputs.shape[1], -1)
        #     if elem is None:
        #         elem = elem_b
        #     else:
        #         elem = torch.cat([elem, elem_b], dim=1)

        elems = []
        for inputs, _ in self.loader(batch_size, shuffle=False):
            # p は、バッチの次元を除いたものが2次元データなら(1, 0, 2, 3)、1次元データなら(1, 0, 2)
            p = torch.arange(len(inputs.shape))
            p[0], p[1] = 1, 0
            p = tuple(p)

            elems.append(inputs.permute(p).reshape(inputs.shape[1], -1))

        elem = torch.cat(elems, dim=1)

        mean = elem.mean(dim=1).tolist()
        std = elem.std(dim=1).tolist()

        if formatted:
            return f"transforms.Normalize(mean={mean}, std={std}, inplace=True)"
        else:
            return {"mean": mean, "std": std}

    def calc_min_max(self, batch_size=256, formatted=False):
        elem = None
        for inputs, _ in self.loader(batch_size, shuffle=False):
            # p は、バッチの次元を除いたものが2次元データなら(1, 0, 2, 3)、1次元データなら(1, 0, 2)
            p = torch.arange(len(inputs.shape))
            p[0], p[1] = 1, 0
            p = tuple(p)

            elem_b = inputs.permute(p).reshape(inputs.shape[1], -1)
            if elem is None:
                elem = elem_b
            else:
                elem = torch.cat([elem, elem_b], dim=1)

        min = elem.min(dim=1).values.tolist()
        max = elem.max(dim=1).values.tolist()

        if formatted:
            return f"transforms.Normalize(min={min}, max={max}, inplace=True)"
        else:
            return {"min": min, "max": max}

    def calc_classdist(self, plot=False, algorithm=0):
        labels, mapping = self._fetch_base_info("map")

        if plot:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            fig = plt.figure( figsize=(8,8) )
            ax = fig.add_subplot(1, 1, 1)
            i = 0

        points = []
        for class_label in self.fetch_classes(listed=True):
            class_mask = labels == class_label
            x = mapping[class_mask, 0]
            y = mapping[class_mask, 1]

            class_center = np.array([x.mean(), y.mean()])
            points.append(class_center)

            if plot:
                c = i % len(color_cycle)
                ax.scatter(x, y, color=color_cycle[c], label=f'Class {class_label}', alpha=0.25, s=4)
                ax.scatter(x=class_center[0], y=class_center[1], color=color_cycle[c], marker="*", s=125)
                i += 1

        points = np.stack(points)

        distance_matrix = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=-1))

        if algorithm == 0:
            distance = distance_matrix.mean()

        elif algorithm == 1:
            distance = np.sqrt(distance_matrix).mean()

        elif algorithm == 2:
            mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
            tmp_mat = distance_matrix[mask].reshape(distance_matrix.shape[0], -1)
            distance = tmp_mat.min(axis=1).mean()

        if plot:
            ax.set_title(f"class_dist: {distance}")
            ax.legend()
            fig.show()

        return distance

    def load_data(self, batch_size=256, one_dim=False):
        all_inputs = None
        all_labels = None
        for inputs, labels in self.loader(batch_size, shuffle=True):
            if one_dim:
                inputs = inputs.view(len(inputs), -1)
                # inputs = torch.flatten(inputs, start_dim=1)

            if all_inputs is None:
                all_inputs = inputs
                all_labels = labels
            else:
                all_inputs = torch.cat([all_inputs, inputs], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)

        return all_inputs, all_labels

    # 未実装
    def _ds_to_folder(self, num, path):
        #     path = Path(path)
    #     path.mkdir(parents=True, exist_ok=True)
    #     ds_it = iter(ds)
    #     for i in range(num):
    #         img, label = next(ds_it)
    #         img = np.array(img, dtype=np.uint8)
    #         img = Image.fromarray(img)
    #         img_name = f"{i}_label.png"
    #         img.save(path / Path(img_name), format="png")
        pass

    def _fetch_base_info(self, info):
        expected = ("ld", "map", "msd")
        assert info in expected, f"Invalid argument. Expected one of: {', '.join(expected)}."

        obj_path = self.base_ds.root / (self.ds_str + "." + info)
        info_path = self.base_ds.root / (self.ds_str + ".info")

        # 拡張子で検索 -> .infoの[info]を検索
        if obj_path.exists():
            obj = torch.load(obj_path, weights_only=False)

        elif info_path.exists(): # 現仕様では常にelseとなる
            info_dict = torch.load(info_path, weights_only=False)
            obj = info_dict[info]

        # 現仕様では.infoファイルは作成しない
        else:
            obj = getattr(self, f"_make_{info}")()
            torch.save(obj, obj_path)
            print(f"Saved label data to the following path: {obj_path}")

        return obj

    def _make_ld(self):
        # fetch_ldとほぼ同じだが、ところどころ違うから別で定義した方が楽そう

        label_l = []  # label のリストを作成
        label_d = dict()  # index と対応させ、label を key とし、index を item とした dict を作成
        for idx in self.indices:
            _, label = self.base_ds[idx]
            label_l.append(label)
            if label_d.get(label) is None:
                label_d[label] = [idx]
            else:
                label_d[label].append(idx)

        label_d = dict(sorted(label_d.items()))

        return label_l, label_d

    def _make_msd(self):
        ms_dict = self.base_ds.base_dsh.transform([torchvision.transforms.ToTensor()]).calc_mean_std(base=False)

        return ms_dict

    def _make_map(self, transform_l=None, batch_size=256, feat_extracter=None, dim_reducer_f=None):
        if transform_l is None:
            transform_l = [torchvision.transforms.Lambda(lambda image: image.convert("RGB")), torchvision.transforms.ToTensor(), torchvision.transforms.Resize((224, 224), antialias=True)]

        tmp_ds = self.transform(transform_l=transform_l)
        tmp_dl = tmp_ds.loader(batch_size)

        if feat_extracter is None:
            transform_l.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True))
            base_arc = resnet50(weights=ResNet50_Weights.DEFAULT)
            feat_extracter = torch.nn.Sequential(*list(base_arc.children())[:-1])

        if dim_reducer_f is None:
            dim_reducer_f = UMAP(n_neighbors=50, min_dist=0.1).fit_transform

        trainer = Trainer(network=feat_extracter)

        feat, labels = trainer.pred_1iter(tmp_dl)
        feat = feat.view(len(feat), -1)
        feat = feat.cpu()
        labels = labels.cpu()

        mapping = dim_reducer_f(feat)

        return labels, mapping

