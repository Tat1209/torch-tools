import functools
import inspect
import itertools
import math
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np


def interval(step=None, itv=None, last_step=None):
    if step is None  or  itv is None:
        return True
    else:
        return (step - 1) % itv >= itv - 1  or  step == last_step

def get_source(path=None, with_name=False):
    if path is None:
        src_fname = inspect.stack()[1].filename
        path = Path(src_fname)
    src_name = path.name
    src_text = path.read_text(encoding='utf-8')
    
    if with_name:
        return src_text, src_name
    else:
        return src_text

def torch_fix_seed(seed=42):
    import torch
    random.seed(seed) # Python random
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # Pytorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # cuda
    torch.backends.cudnn.deterministic = True # cudnn
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

def format_duration(duration_s, style=0) -> str:
    """
    Converts a duration in seconds to a human-readable format.

    Args:
        duration_s (float): Duration in seconds to be formatted.
        style (int, optional): 
            Formatting style. 
            - 0: Single highest unit (e.g., "1.5ms")
            - 1: Two-level breakdown (e.g., "1m23s")
            Defaults to 0.

    Returns:
        str: Formatted duration string.
    """
    duration_ns = duration_s * 1e9
    units = [("ns", 1), ("μs", 1000), ("ms", 1000), ("s", 1000), ("m", 60), ("h", 60), ("d", 24), ("w", 7), ("x", 10000)]
    # units = [("ns", 1), ("μs", 1000), ("ms", 1000), ("s", 1000), ("m", 60), ("h", 60), ("d", 24), ("y", 365), ("c", 100), ("e", 10)]

    # 最大単位を整数部にしたfloatに変換
    if style == 0:
        unit_b = ""
        t_b = duration_ns
        for unit, divisor in units:
            if t_b < divisor:
                return f"{t_b:.1f}{unit_b}"
            t_b /= divisor
            unit_b = unit
        
    # 最大単位から2段階表示
    elif style == 1:
        def digit_count(n):
            if n == 0:
                return 1
            return int(math.log10(abs(n))) + 1

        quo = int(duration_ns)
        fmt_d = {}
        unit_b = ""
        for unit, divisor in units:
            quo, rem = divmod(quo, divisor)
            fmt_d[unit_b] = (rem, digit_count(divisor - 1))
            unit_b = unit

        fmt_str = ""
        part_count = 0
        for k, v in reversed(fmt_d.items()):
            value, digit = v
            if value > 0:
                if part_count == 0  or  part_count == 1:
                    fmt_str += f"{value:{digit}d}{k}"
                    part_count += 1
                else:
                    break
                    
        return fmt_str
    
def format_time(timestamp: float = None, style: int = 0) -> str:
    """
    タイムスタンプを指定された書式の文字列に変換します。
    
    Args:
        timestamp (float | None): time.time()などで取得したUNIX時間。Noneの場合は現在時刻。
        style (int): フォーマットのスタイル指定
    """
    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromtimestamp(timestamp)
    
    if style == 0:
        return dt.strftime("%Y/%m/%d %H:%M")
    elif style == 1:
        return dt.strftime("%Y/%m/%d %H:%M:%S")
    elif style == 2:
        return dt.strftime("%Y-%m-%d_%H-%M-%S_%f")
    else:
        raise ValueError(f"Unknown format style: {style}")

def is_reached(*args, tag="tmp") -> bool:
    """
    タグによって区別することで、複数のチェックを独立して実行できる SkipManager の関数版。
    
    Args:
        tag (str): チェックを一位に識別するための文字列。
        *args: (現在値, 目標値) のタプルを可変長引数として受け取る。
    
    Returns:
        bool: 条件が一度でも満たされたら True を返し続ける。それ以外は False。
    """
    # 状態を保存するための辞書を関数オブジェクトに持たせる
    if not hasattr(is_reached, "states"):
        is_reached.states = {}

    # このタグ用の状態がなければ初期化する
    if tag not in is_reached.states:
        is_reached.states[tag] = {"reached": False, "skip_count": 0}

    # このタグに紐づく状態を取得
    tag_state = is_reached.states[tag]

    # すでに一度条件を満たしている場合は、常にTrueを返す
    if tag_state["reached"]:
        return True

    # すべての引数ペアが (現在値 == 目標値) になっているか判定
    if all(t[0] == t[1] for t in args):
        tag_state["reached"] = True
        print(f"[{tag}] Skipped {tag_state['skip_count']} times before reaching.")
        return True
    else:
        tag_state['skip_count'] += 1
        return False
    
def get_resource_config(max_process: int = 1, reserve_threads: int = 0) -> dict:
    """
    システムのCPUコア数を取得し、指定された同時実行プロセス数で分割した
    最適なスレッド数・ワーカー数設定を辞書で返します。

    Args:
        max_process (int): 同時に実行するプロセス数（デフォルト: 1）。
        reserve_threads (int): システム全体として残しておく予備スレッド数（デフォルト: 0）。

    Returns:
        dict: 各プロセスに割り当てるべき設定値。
            - 'num_threads': (int) Intra-op並列数
            - 'num_interop_threads': (int) Inter-op並列数
            - 'num_workers': (int) DataLoaderのワーカー数
    """
    if max_process < 1:
        raise ValueError("max_process must be at least 1.")

    # 1. 利用可能な全CPUコア数の取得 (コンテナ対応)
    try:
        available_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = os.cpu_count() or 1

    # 2. 予備スレッドの確保
    # システム全体で reserve_threads を引いた残りを計算
    allocatable_cores = max(1, available_cores - reserve_threads)

    # 3. プロセスごとの割り当て計算
    # 利用可能なコア数をプロセス数で割り、切り捨て(floor)を行うことで
    # 合計が物理リソースを超えないようにする。
    cores_per_process = math.floor(allocatable_cores / max_process)

    # 少なくとも1スレッドは確保する
    cores_per_process = max(1, cores_per_process)

    # 4. 辞書の作成
    # num_workers について:
    # PyTorchのDataLoaderはメインプロセスとは別にプロセスを立ち上げます。
    # ここでは「その学習プロセスに割り当てられたCPU枠」を上限として設定します。
    config = {
        "num_threads": cores_per_process,
        "num_interop_threads": cores_per_process,
        "num_workers": cores_per_process
    }

    return config


class TimeLog:
    """
    説明:
      - クラスデコレータ (@TimeLog(...)):
          インスタンス生成時に `_time_data` と `_time_meta` を初期化し、
          指定した名前のタイム取得メソッドを追加して計測結果を取得できるようにする
      - メソッドデコレータ (@TimeLog("key", mode="add" or "set")):
          ・mode="add": 実行時間を `_time_data[key]` に累積
          ・mode="set": 呼び出し完了時刻（UNIX time）を `_time_data[key]` に保存
      - イテレーション計測ヘルパ:
          - `iter_load`    : 「次の要素取得部分のみ」を計測
          - `iter_process` : 「取得＋その後の処理全体」を計測
          - `iter_profile` : 1 回のイテレーションで両方（load, full-loop）を計測し、別々のキーに累積

    使用方法:
      1. クラスに対して以下のように付与
         - `@TimeLog()`                     → インスタンスに `_time_data`, `_time_meta`, `time_stats()` を追加
         - `@TimeLog(stats_name="get_times")` → `get_times()` を追加 (time_stats と同内容)
      2. 計測したい個別メソッドに付与
         - `@TimeLog("任意のキー", mode="add")`：実行時間を累積
         - `@TimeLog("任意のキー", mode="set")`：呼び出し完了時刻を保存
      3. ループ部分を計測したい場合は、メソッド内で以下のように呼び出す
         ```python
         # データ取得のみを計測
         for inputs, labels in TimeLog.iter_load(self, "fetch_time", dataloader):
             # ここに処理を書くが、TimeLog.iter_load() は「取得部分のみ」のみ計測、
             # この後の処理は計測対象外になる
             _ = some_model(inputs)

         # 取得＋内部処理をまとめて計測
         for inputs, labels in TimeLog.iter_process(self, "full_time", dataloader):
             _ = some_model(inputs)

         # 同じループで「両方を同時に計測し、2 つのキーに格納」したい場合
         for inputs, labels in TimeLog.iter_profile(self, "fetch_time", "full_time", dataloader):
             _ = some_model(inputs)
         ```
      4. クラス定義後、インスタンスで取得メソッドを呼び出す例
         ```python
         @TimeLog(stats_name="get_times")
         class Worker:
             @TimeLog("run", mode="add")
             def run(self, dataloader):
                 # iter_load: 取得のみ
                 for inputs, labels in TimeLog.iter_load(self, "fetch_time", dataloader):
                     _ = some_model(inputs)

                 # iter_process: 取得＋処理全体
                 for inputs, labels in TimeLog.iter_process(self, "full_time", dataloader):
                     _ = some_model(inputs)

                 # iter_profile: 1 ループで両方同時に取得
                 for inputs, labels in TimeLog.iter_profile(self, "fetch_time", "full_time", dataloader):
                     _ = some_model(inputs)

         w = Worker()
         w.run(my_dataloader)
         stats = w.get_times(incl_fmt=True)
         print(stats)
         # 例:
         # {
         #   "run": 0.523,              "run_fmt": "0.523 秒",
         #   "fetch_time": 0.142,       "fetch_time_fmt": "0.142 秒",
         #   "full_time": 0.489,        "full_time_fmt": "0.489 秒"
         # }
         ```
    """

    def __init__(
        self,
        key: Optional[str] = None,
        mode: str = "set",
        stats_name: str = "time_stats",
    ):
        if mode not in ("add", "set"):
            raise ValueError(f"mode must be 'add' or 'set' (got {mode!r})")
        self.key = key
        self.mode = mode
        self.stats_name = stats_name

    def __call__(self, obj: Any) -> Any:
        # ――― クラスデコレータとして振る舞う場合 ―――
        if isinstance(obj, type):
            cls = obj
            orig_init = getattr(cls, "__init__", None)

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._time_data = {}
                self._time_meta = {}
                if orig_init:
                    orig_init(self, *args, **kwargs)
                else:
                    super(cls, self).__init__(*args, **kwargs)

            setattr(cls, "__init__", __init__)

            def generated_stats(
                self,
                incl_fmt: bool = True,
                style_dur: int = 0,
                style_time: int = 1,
            ) -> Dict[str, Any]:
                return TimeLog.time_stats(
                    self, incl_fmt=incl_fmt, style_dur=style_dur, style_time=style_time
                )

            setattr(cls, self.stats_name, generated_stats)
            return cls

        # ――― メソッドデコレータとして振る舞う場合 ―――
        else:
            func: Callable = obj
            if self.key is None:
                raise ValueError(
                    "TimeLog デコレータをメソッドに使う場合は、key を指定してください。"
                )

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                instance = args[0]
                instance._time_meta[self.key] = self.mode
                if self.mode == "add":
                    start = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    instance._time_data[self.key] = (
                        instance._time_data.get(self.key, 0.0) + elapsed
                    )
                    return result
                else:  # mode == "set"
                    result = func(*args, **kwargs)
                    instance._time_data[self.key] = time.time()
                    return result

            return wrapper

    @staticmethod
    def time_stats(
        instance: Any, incl_fmt: bool = True, style_dur: int = 0, style_time: int = 1
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if not hasattr(instance, "_time_data"):
            return stats

        for k, v in instance._time_data.items():
            stats[k] = v
            if incl_fmt and hasattr(instance, "_time_meta"):
                mode = instance._time_meta.get(k)
                fmt_key = f"{k}_fmt"
                if mode == "add":
                    stats[fmt_key] = format_duration(v, style=style_dur)
                elif mode == "set":
                    stats[fmt_key] = format_time(v, style=style_time)
        return stats

    @staticmethod
    def iter_load(
        instance: Any, key: str, iterable: Iterable[Any]
    ) -> Iterator[Any]:
        """
        「データ取得のみ」を計測し、結果を _time_data[key] に累積する。

        Args:
            instance (Any): TimeLog がデコレートされたインスタンス（self に相当）
            key (str)       : 累積時間を記録するキー名
            iterable (Iterable[Any]): DataLoader などのイテレータ

        Yields:
            各 iterable の要素（例: inputs, labels のタプル）
        """
        # メタ情報に「add」を記録
        instance._time_meta[key] = "add"
        iterator = iter(iterable)
        while True:
            start = time.time()
            try:
                item = next(iterator)  # ここで「要素取得」のみを計測
            except StopIteration:
                break
            elapsed = time.time() - start
            instance._time_data[key] = instance._time_data.get(key, 0.0) + elapsed
            # 取得した要素を返す
            yield item

    @staticmethod
    def iter_process(
        instance: Any, key: str, iterable: Iterable[Any]
    ) -> Iterator[Any]:
        """
        「データ取得＋その後の処理全体」を計測し、結果を _time_data[key] に累積する。

        Args:
            instance (Any): TimeLog がデコレートされたインスタンス（self に相当）
            key (str)       : 累積時間を記録するキー名
            iterable (Iterable[Any]): DataLoader などのイテレータ

        Yields:
            各 iterable の要素（例: inputs, labels のタプル）
        """
        instance._time_meta[key] = "add"
        iterator = iter(iterable)
        while True:
            start = time.time()
            try:
                item = next(iterator)  # 取得部分＋ユーザー処理を含む
            except StopIteration:
                break
            # yield 後に、取得から処理終了までの全体時間を計算
            yield item
            elapsed = time.time() - start
            instance._time_data[key] = instance._time_data.get(key, 0.0) + elapsed

    @staticmethod
    def iter_profile(
        instance: Any,
        fetch_key: str,
        full_key: str,
        iterable: Iterable[Any]
    ) -> Iterator[Any]:
        """
        1 回のイテレーションにつき
          1) fetch_key : 「要素取得部分のみ」の時間を計測
          2) full_key  : 「要素取得＋その後の処理全体」の時間を計測
        の両方を _time_data に累積する。

        Args:
            instance  (Any): TimeLog がデコレートされたインスタンス（self に相当）
            fetch_key (str): 取得のみの累積時間を記録するキー
            full_key  (str): フルループ（取得＋内部処理）の累積時間を記録するキー
            iterable  (Iterable[Any]): DataLoader などのイテレータ

        Yields:
            各 iterable の要素（例: inputs, labels のタプル）
        """
        instance._time_meta[fetch_key] = "add"
        instance._time_meta[full_key]  = "add"

        iterator = iter(iterable)
        while True:
            # ─── Step1: 「取得のみ」時間の計測 ───
            t0 = time.time()
            try:
                item = next(iterator)
            except StopIteration:
                break
            t1 = time.time()
            fetch_elapsed = t1 - t0
            instance._time_data[fetch_key] = instance._time_data.get(fetch_key, 0.0) + fetch_elapsed

            # ─── Step2: 「取得＋処理全体」時間の計測開始 ───
            # t0 を起点にして、ユーザー処理までの全体を計測
            full_start = t0

            yield item

            full_end = time.time()
            full_elapsed = full_end - full_start
            instance._time_data[full_key] = instance._time_data.get(full_key, 0.0) + full_elapsed

class SkipManager:
    def __init__(self):
        self.reached = False

    def is_reached(self, tuples) -> bool:
        if self.reached:
            return True
        elif all(all(x == t[0] for x in t) for t in tuples):
            self.reached = True
            return True
        else:
            return False

class LazyPipeline:
    def __init__(self, seed_obj: Any, _steps: List[dict] = None):
        self._seed_obj = seed_obj
        self._steps = _steps or []

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.history()}>"

    def history(self) -> str:
        step_strs = []
        for step in self._steps:
            args = ", ".join(
                [str(a) for a in step['args']] + 
                [f"{k}={v}" for k, v in step['kwargs'].items()]
            )
            step_strs.append(f"{step['func'].__name__}({args})")
        
        base_name = getattr(self._seed_obj, '__name__', 'Base')
        if not isinstance(base_name, str):
            base_name = 'Base'
            
        return ".".join([base_name] + step_strs)

    def pipe(self, func: Callable, *args, **kwargs) -> 'LazyPipeline':
        step_info = {'func': func, 'args': args, 'kwargs': kwargs}
        return self.__class__(self._seed_obj, self._steps + [step_info])

    def _execute(self, target_obj: Any) -> Any:
        current_obj = target_obj
        for step in self._steps:
            new_result = step['func'](current_obj, *step['args'], **step['kwargs'])
            if new_result is not None:
                current_obj = new_result
        return current_obj
