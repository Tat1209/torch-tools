import os
import time
import fcntl
import itertools
import multiprocessing
import traceback
import pynvml
from typing import Any, IO, Callable
from pathlib import Path
from copy import deepcopy

def get_device(
    avoid_used: bool,
    util_th: int,
    free_mem_th: float,
    avoid_locked: bool = True,
    lock_path: Path = Path("~/.gpu_locks"),
    lock: bool = True,
    gpu_ids: list[int] | None = None
) -> tuple[Any, IO] | tuple[None, None]:
    """
    Args:
        avoid_used (bool): Trueの場合，GPUの使用率と空きメモリ容量を監視して空きGPUを探す．
        util_th (int): 許容するGPU使用率の上限(%)．これ以下の使用率のGPUを選択する．
        free_mem_th (float): 必要とする空きメモリの下限(MiB)．これ以上の空きがあるGPUを選択する．
        avoid_locked (bool): Trueの場合，ロックファイルが存在するGPUをスキップする．
        lock_path (Path): 排他制御用のロックファイルを保存するディレクトリ．
        lock (bool): Trueの場合，選択したGPUに対してファイルロックを取得して返す．
        gpu_ids (list[int] | None): 使用を許可するGPU IDのリスト．Noneの場合は全GPUを対象とする．

    Returns:
        lock=False: torch.device or None
        lock=True:  (torch.device, IO) or (None, None)
    """
    lock_path = lock_path.expanduser()
    lock_path.mkdir(parents=True, exist_ok=True)

    try:
        pynvml.nvmlInit()
        
        if gpu_ids is not None:
            candidates = gpu_ids
        else:
            candidates = range(pynvml.nvmlDeviceGetCount())
        
        for i in candidates:
            # 指定されたIDが存在しない場合はスキップ(エラー回避)
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except pynvml.NVMLError:
                continue
            
            if avoid_used:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                free_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 ** 2)

                if util > util_th or free_mem < free_mem_th:
                    continue

            lock_file = lock_path / f"gpu_{i}.lock"
            
            if avoid_locked and lock_file.exists():
                try:
                    with open(lock_file, "w") as f:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f, fcntl.LOCK_UN)
                except OSError:
                    continue

            if lock:
                try:
                    f = open(lock_file, "w")
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    Device = type("Device", (object,), {"index": i})
                    return Device(), f
                except OSError:
                    f.close()
                    continue
            else:
                Device = type("Device", (object,), {"index": i})
                return Device(), None

    except pynvml.NVMLError:
        return None, None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return None, None

def generate_task_grid(config: dict[str | tuple[str, ...], Any]) -> list[dict[str, Any]]:
    """
    設定辞書からタスクリストを生成する。
    リストの値は直積(Grid)の対象となるが、タプルキーを用いることで特定の要素をZip(同期)できる。

    Usage:
        config = {
            # --- Zip (Linked parameters) ---
            # キーをタプルにし、値をタプル(またはリスト)のリストにすると同期して変動する
            ("model", "lr"): [("resnet", 1e-3), ("vit", 1e-4)],

            # --- Grid (Cartesian product) ---
            # 通常の文字列キーは他の要素と直積をとる
            "batch_size": [32, 64],

            # --- Fixed ---
            # リストでない値は固定される
            "epochs": 10
        }
        tasks = generate_task_grid(config)
        # Result: 2 (model/lr) * 2 (batch_size) = 4 configs
    """
    sweep_items = []
    fixed = {}

    for k, v in config.items():
        if isinstance(v, list):
            sweep_items.append((k, v))
        else:
            fixed[k] = v

    if not sweep_items:
        return [deepcopy(fixed)]

    keys, values_list = zip(*sweep_items)
    tasks = []

    for combo in itertools.product(*values_list):
        new_cfg = deepcopy(fixed)
        
        for key_def, val in zip(keys, combo):
            if isinstance(key_def, tuple):
                # Zip展開: タプルキー ("a", "b") -> 値 (1, 2) をそれぞれ代入
                for sub_k, sub_v in zip(key_def, val):
                    new_cfg[sub_k] = sub_v
            else:
                # Grid展開: 通常キー "a" -> 値 1 を代入
                new_cfg[key_def] = val
        
        tasks.append(new_cfg)

    return tasks

def _worker_wrapper(task_func: Callable, cfg: dict, gpu_id: int):
    """[内部用] 環境変数をセットしてタスクを実行するラッパー"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    p_name = multiprocessing.current_process().name
    try:
        print(f"[{p_name}] Start Task on GPU {gpu_id}")
        task_func(cfg)
        print(f"[{p_name}] Done Task on GPU {gpu_id}")
    except Exception:
        print(f"[Error] Task failed on GPU {gpu_id}")
        traceback.print_exc()

def _spawn_worker(task_func: Callable, cfg: dict, gpu_id: int) -> multiprocessing.Process:
    """[ヘルパー] ワーカープロセスの生成と起動"""
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(
        target=_worker_wrapper, 
        args=(task_func, cfg, gpu_id),
        name=f"Worker-GPU{gpu_id}"
    )
    p.start()
    return p

def _cleanup_finished_processes(running_procs: dict[multiprocessing.Process, IO]):
    """[ヘルパー] 終了したプロセスのロック解放とリスト削除"""
    for p in list(running_procs.keys()):
        if not p.is_alive():
            p.join()
            lock_handle = running_procs[p]
            if lock_handle:
                try:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                    lock_handle.close()
                except IOError as e:
                    print(f"[Warn] Failed to close lock: {e}")
            del running_procs[p]


def parallel_run(
    task_func: Callable[[dict[str, Any]], None],
    config: dict[str, Any],
    max_tasks: int | None = None,
    check_interval: float = 1.0,
    avoid_used: bool = False,
    util_th: int = 10,
    free_mem_th: float = 2000.0,
    lock_path: Path = Path("~/.gpu_locks"),
    gpu_ids: list[int] | None = None
):
    """
    GPUリソースを管理しながらタスクを並列実行するスケジューラ.

    Args:
        task_func (Callable): 各タスクを実行する関数．引数として設定辞書を受け取る．
        config (dict): 実験設定を含む辞書．リスト型の値はグリッドサーチ（Sweep）の対象として展開される．
        max_tasks (int | None): 同時に実行するプロセスの最大数．Noneの場合は制限なし．
        check_interval (float): 空きGPUを探すポーリング間隔（秒）．
        avoid_used (bool): Trueの場合，GPUの負荷状況（使用率・メモリ）を考慮して割り当てを行う．
        util_th (int): 割り当て対象とするGPU使用率の上限(%)．
        free_mem_th (float): 割り当てに必要とするGPU空きメモリの下限(MiB)．
        lock_path (Path): GPUの排他制御用ロックファイルを保存するパス．
        gpu_ids (list[int] | None): 使用を許可するGPU IDのリスト．Noneの場合は全GPUを使用候補とする．
    """
    tasks = generate_task_grid(config)
    print(f"[Scheduler] Total tasks: {len(tasks)}")
    
    running_procs: dict[multiprocessing.Process, IO] = {}

    try:
        while tasks or running_procs:
            _cleanup_finished_processes(running_procs)

            if tasks and (max_tasks is None or len(running_procs) < max_tasks):
                res = get_device(
                    avoid_used=avoid_used,
                    util_th=util_th,
                    free_mem_th=free_mem_th,
                    avoid_locked=True,
                    lock_path=lock_path,
                    lock=True,
                    gpu_ids=gpu_ids
                )
                
                if res and res[0] is not None:
                    device, lock_handle = res
                    gpu_id = device.index
                    
                    next_cfg = tasks.pop(0)
                    p = _spawn_worker(task_func, next_cfg, gpu_id)
                    running_procs[p] = lock_handle
                    
                    print(f"[Scheduler] Assigned GPU {gpu_id}. (Running: {len(running_procs)}, Rem: {len(tasks)})")
                else:
                    time.sleep(check_interval)
            else:
                if running_procs:
                    time.sleep(check_interval)

        print("[Scheduler] All tasks completed.")

    except KeyboardInterrupt:
        print("\n[Scheduler] Terminating...")
        for p in list(running_procs.keys()):
            if p.is_alive():
                p.terminate()
                p.join()
                lh = running_procs[p]
                if lh:
                    try:
                        fcntl.flock(lh, fcntl.LOCK_UN)
                        lh.close()
                    except Exception:
                        pass
        print("[Scheduler] Shutdown complete.")